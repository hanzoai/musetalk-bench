#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{Result, Tensor, D};
use hanzo_nn::{Conv2d, GroupNorm, Module};
use hanzo_quant::{Convolution, ShardedVarBuilder};

use crate::layers::{conv2d, group_norm, MatMul};

use super::config::VaeConfig;

const ATTN_EPS: f64 = 1e-6;
const RESNET_EPS: f64 = 1e-6;

fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (dim as f64).sqrt();
    let attn = (MatMul.matmul(q, &k.t()?)? * scale)?;
    MatMul.matmul(&hanzo_nn::ops::softmax_last_dim(&attn)?, v)
}

#[derive(Debug, Clone)]
struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    fn new(in_c: usize, out_c: usize, groups: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let conv_cfg = hanzo_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let norm1 = group_norm(groups, in_c, RESNET_EPS, vb.pp("norm1"))?;
        let conv1 = conv2d(in_c, out_c, 3, conv_cfg, vb.pp("conv1"))?;
        let norm2 = group_norm(groups, out_c, RESNET_EPS, vb.pp("norm2"))?;
        let conv2 = conv2d(out_c, out_c, 3, conv_cfg, vb.pp("conv2"))?;
        let conv_shortcut = if in_c == out_c {
            None
        } else {
            Some(conv2d(in_c, out_c, 1, Default::default(), vb.pp("conv_shortcut"))?)
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            conv_shortcut,
        })
    }
}

impl ResnetBlock {
}

impl Module for ResnetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let mut h = self.norm1.forward(xs)?;
        h = h.silu()?;
        h = Convolution.forward_2d(&self.conv1, &h)?;
        h = self.norm2.forward(&h)?;
        h = h.silu()?;
        h = Convolution.forward_2d(&self.conv2, &h)?;
        match self.conv_shortcut.as_ref() {
            None => h + residual,
            Some(c) => h + Convolution.forward_2d(c, residual)?,
        }
    }
}

#[derive(Debug, Clone)]
struct AttnBlock {
    group_norm: GroupNorm,
    to_q: Conv2d,
    to_k: Conv2d,
    to_v: Conv2d,
    to_out: Conv2d,
}

impl AttnBlock {
    fn new(channels: usize, groups: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let group_norm = group_norm(groups, channels, ATTN_EPS, vb.pp("group_norm"))?;
        let to_q = conv2d(channels, channels, 1, Default::default(), vb.pp("to_q"))?;
        let to_k = conv2d(channels, channels, 1, Default::default(), vb.pp("to_k"))?;
        let to_v = conv2d(channels, channels, 1, Default::default(), vb.pp("to_v"))?;
        let to_out = conv2d(channels, channels, 1, Default::default(), vb.pp("to_out").pp(0))?;
        Ok(Self {
            group_norm,
            to_q,
            to_k,
            to_v,
            to_out,
        })
    }
}

impl Module for AttnBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let normed = self.group_norm.forward(xs)?;
        let q = Convolution.forward_2d(&self.to_q, &normed)?;
        let k = Convolution.forward_2d(&self.to_k, &normed)?;
        let v = Convolution.forward_2d(&self.to_v, &normed)?;
        let (b, c, h, w) = q.dims4()?;
        let q = q.flatten_from(2)?.t()?.unsqueeze(1)?;
        let k = k.flatten_from(2)?.t()?.unsqueeze(1)?;
        let v = v.flatten_from(2)?.t()?.unsqueeze(1)?;
        let out = sdpa(&q, &k, &v)?;
        let out = out.squeeze(1)?.t()?.reshape((b, c, h, w))?;
        Convolution.forward_2d(&self.to_out, &out)? + residual
    }
}

#[derive(Debug, Clone)]
struct Downsample {
    conv: Conv2d,
}

impl Downsample {
    fn new(channels: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let conv_cfg = hanzo_nn::Conv2dConfig {
            stride: 2,
            ..Default::default()
        };
        let conv = conv2d(channels, channels, 3, conv_cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }
}

impl Module for Downsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.pad_with_zeros(D::Minus1, 0, 1)?;
        let xs = xs.pad_with_zeros(D::Minus2, 0, 1)?;
        Convolution.forward_2d(&self.conv, &xs)
    }
}

#[derive(Debug, Clone)]
struct Upsample {
    conv: Conv2d,
}

impl Upsample {
    fn new(channels: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let conv_cfg = hanzo_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = conv2d(channels, channels, 3, conv_cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }
}

impl Module for Upsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        let up = xs.upsample_nearest2d(h * 2, w * 2)?;
        Convolution.forward_2d(&self.conv, &up)
    }
}

#[derive(Debug, Clone)]
struct MidBlock {
    resnet_1: ResnetBlock,
    attn: AttnBlock,
    resnet_2: ResnetBlock,
}

impl MidBlock {
    fn new(channels: usize, groups: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let resnet_1 = ResnetBlock::new(channels, channels, groups, vb.pp("resnets").pp(0))?;
        let attn = AttnBlock::new(channels, groups, vb.pp("attentions").pp(0))?;
        let resnet_2 = ResnetBlock::new(channels, channels, groups, vb.pp("resnets").pp(1))?;
        Ok(Self {
            resnet_1,
            attn,
            resnet_2,
        })
    }
}

impl Module for MidBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.resnet_1.forward(xs)?;
        let h = self.attn.forward(&h)?;
        self.resnet_2.forward(&h)
    }
}

#[derive(Debug, Clone)]
struct EncDownBlock {
    resnets: Vec<ResnetBlock>,
    downsampler: Option<Downsample>,
}

#[derive(Debug, Clone)]
struct Encoder {
    conv_in: Conv2d,
    down_blocks: Vec<EncDownBlock>,
    mid_block: MidBlock,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Encoder {
    fn new(cfg: &VaeConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let conv_cfg = hanzo_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let base = cfg.block_out_channels[0];
        let conv_in = conv2d(cfg.in_channels, base, 3, conv_cfg, vb.pp("conv_in"))?;

        let vb_down = vb.pp("down_blocks");
        let mut down_blocks = Vec::with_capacity(cfg.block_out_channels.len());
        let mut prev_ch = base;
        for (i, &out_ch) in cfg.block_out_channels.iter().enumerate() {
            let vb_i = vb_down.pp(i);
            let is_last = i == cfg.block_out_channels.len() - 1;
            let mut resnets = Vec::with_capacity(cfg.layers_per_block);
            let vb_res = vb_i.pp("resnets");
            for j in 0..cfg.layers_per_block {
                let in_c = if j == 0 { prev_ch } else { out_ch };
                resnets.push(ResnetBlock::new(in_c, out_ch, cfg.norm_num_groups, vb_res.pp(j))?);
            }
            let downsampler = if is_last {
                None
            } else {
                Some(Downsample::new(out_ch, vb_i.pp("downsamplers").pp(0))?)
            };
            down_blocks.push(EncDownBlock {
                resnets,
                downsampler,
            });
            prev_ch = out_ch;
        }

        let mid_ch = *cfg.block_out_channels.last().unwrap();
        let mid_block = MidBlock::new(mid_ch, cfg.norm_num_groups, vb.pp("mid_block"))?;
        let conv_norm_out = group_norm(cfg.norm_num_groups, mid_ch, 1e-6, vb.pp("conv_norm_out"))?;
        let conv_out = conv2d(mid_ch, 2 * cfg.latent_channels, 3, conv_cfg, vb.pp("conv_out"))?;
        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
        })
    }
}

impl Encoder {
    fn dump(t: &Tensor, dir: &str, name: &str) -> Result<()> {
        t.to_dtype(hanzo_ml::DType::F32)?
            .to_device(&hanzo_ml::Device::Cpu)?
            .write_npy(format!("{dir}/{name}.npy"))?;
        Ok(())
    }

    pub fn forward_debug(&self, xs: &Tensor, dir: &str) -> Result<Tensor> {
        let mut h = Convolution.forward_2d(&self.conv_in, xs)?;
        Self::dump(&h, dir, "r_vae_e_conv_in")?;
        for (i, block) in self.down_blocks.iter().enumerate() {
            for r in block.resnets.iter() {
                h = r.forward(&h)?;
            }
            if let Some(ds) = block.downsampler.as_ref() {
                h = ds.forward(&h)?;
            }
            Self::dump(&h, dir, &format!("r_vae_e_down{i}"))?;
        }
        let h2 = self.mid_block.resnet_1.forward(&h)?;
        Self::dump(&h2, dir, "r_vae_e_mid_res0")?;
        let h3 = self.mid_block.attn.forward(&h2)?;
        Self::dump(&h3, dir, "r_vae_e_mid_attn")?;
        let h4 = self.mid_block.resnet_2.forward(&h3)?;
        Self::dump(&h4, dir, "r_vae_e_mid_res1")?;
        let mut h = self.conv_norm_out.forward(&h4)?;
        h = h.silu()?;
        let out = Convolution.forward_2d(&self.conv_out, &h)?;
        Self::dump(&out, dir, "r_vae_e_conv_out")?;
        Ok(out)
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = Convolution.forward_2d(&self.conv_in, xs)?;
        for block in self.down_blocks.iter() {
            for r in block.resnets.iter() {
                h = r.forward(&h)?;
            }
            if let Some(ds) = block.downsampler.as_ref() {
                h = ds.forward(&h)?;
            }
        }
        h = self.mid_block.forward(&h)?;
        h = self.conv_norm_out.forward(&h)?;
        h = h.silu()?;
        Convolution.forward_2d(&self.conv_out, &h)
    }
}

#[derive(Debug, Clone)]
struct DecUpBlock {
    resnets: Vec<ResnetBlock>,
    upsampler: Option<Upsample>,
}

#[derive(Debug, Clone)]
struct Decoder {
    conv_in: Conv2d,
    mid_block: MidBlock,
    up_blocks: Vec<DecUpBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Decoder {
    fn new(cfg: &VaeConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let conv_cfg = hanzo_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let reversed: Vec<usize> = cfg.block_out_channels.iter().rev().copied().collect();
        let mid_ch = reversed[0];
        let conv_in = conv2d(cfg.latent_channels, mid_ch, 3, conv_cfg, vb.pp("conv_in"))?;
        let mid_block = MidBlock::new(mid_ch, cfg.norm_num_groups, vb.pp("mid_block"))?;

        let vb_up = vb.pp("up_blocks");
        let mut up_blocks = Vec::with_capacity(reversed.len());
        let mut prev_ch = mid_ch;
        for (i, &out_ch) in reversed.iter().enumerate() {
            let vb_i = vb_up.pp(i);
            let is_last = i == reversed.len() - 1;
            let mut resnets = Vec::with_capacity(cfg.layers_per_block + 1);
            let vb_res = vb_i.pp("resnets");
            for j in 0..cfg.layers_per_block + 1 {
                let in_c = if j == 0 { prev_ch } else { out_ch };
                resnets.push(ResnetBlock::new(in_c, out_ch, cfg.norm_num_groups, vb_res.pp(j))?);
            }
            let upsampler = if is_last {
                None
            } else {
                Some(Upsample::new(out_ch, vb_i.pp("upsamplers").pp(0))?)
            };
            up_blocks.push(DecUpBlock {
                resnets,
                upsampler,
            });
            prev_ch = out_ch;
        }

        let base = *reversed.last().unwrap();
        let conv_norm_out = group_norm(cfg.norm_num_groups, base, 1e-6, vb.pp("conv_norm_out"))?;
        let conv_out = conv2d(base, cfg.out_channels, 3, conv_cfg, vb.pp("conv_out"))?;
        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
        })
    }
}

impl Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = Convolution.forward_2d(&self.conv_in, xs)?;
        h = self.mid_block.forward(&h)?;
        for block in self.up_blocks.iter() {
            for r in block.resnets.iter() {
                h = r.forward(&h)?;
            }
            if let Some(us) = block.upsampler.as_ref() {
                h = us.forward(&h)?;
            }
        }
        h = self.conv_norm_out.forward(&h)?;
        h = h.silu()?;
        Convolution.forward_2d(&self.conv_out, &h)
    }
}

#[derive(Debug, Clone)]
pub struct AutoencoderKl {
    encoder: Encoder,
    decoder: Decoder,
    quant_conv: Conv2d,
    post_quant_conv: Conv2d,
    scaling_factor: f64,
}

impl AutoencoderKl {
    pub fn new(cfg: &VaeConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let decoder = Decoder::new(cfg, vb.pp("decoder"))?;
        let quant_conv = conv2d(
            2 * cfg.latent_channels,
            2 * cfg.latent_channels,
            1,
            Default::default(),
            vb.pp("quant_conv"),
        )?;
        let post_quant_conv = conv2d(
            cfg.latent_channels,
            cfg.latent_channels,
            1,
            Default::default(),
            vb.pp("post_quant_conv"),
        )?;
        Ok(Self {
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
            scaling_factor: cfg.scaling_factor,
        })
    }

    pub fn encode_mode(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.encoder.forward(xs)?;
        let moments = Convolution.forward_2d(&self.quant_conv, &h)?;
        let mean = moments.chunk(2, 1)?[0].clone();
        Ok((mean * self.scaling_factor)?)
    }

    pub fn encode_mode_debug(&self, xs: &Tensor, dir: &str) -> Result<Tensor> {
        let h = self.encoder.forward_debug(xs, dir)?;
        let moments = Convolution.forward_2d(&self.quant_conv, &h)?;
        Encoder::dump(&moments, dir, "r_vae_e_moments")?;
        let mean = moments.chunk(2, 1)?[0].clone();
        Ok((mean * self.scaling_factor)?)
    }

    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let z = (latents / self.scaling_factor)?;
        let z = Convolution.forward_2d(&self.post_quant_conv, &z)?;
        self.decoder.forward(&z)
    }
}
