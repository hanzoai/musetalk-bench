#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{DType, Result, Tensor, D};
use hanzo_nn::{Conv2d, GroupNorm, LayerNorm, Linear, Module};
use hanzo_quant::{Convolution, ShardedVarBuilder};

use crate::layers::{conv2d, group_norm, layer_norm, linear, linear_no_bias, MatMul};

use super::config::UNetConfig;

const MAX_PERIOD: f64 = 10000.;

fn timestep_freqs(half: usize, dev: &hanzo_ml::Device) -> Result<Tensor> {
    let arange = Tensor::arange(0, half as u32, dev)?.to_dtype(DType::F32)?;
    (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()
}

fn timestep_embedding(t: &Tensor, freqs: &Tensor, flip_sin_to_cos: bool, dtype: DType) -> Result<Tensor> {
    let args = t
        .unsqueeze(1)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    let (a, b) = if flip_sin_to_cos {
        (args.cos()?, args.sin()?)
    } else {
        (args.sin()?, args.cos()?)
    };
    Tensor::cat(&[a, b], D::Minus1)?.to_dtype(dtype)
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (dim as f64).sqrt();
    let attn = (MatMul.matmul(q, &k.t()?)? * scale)?;
    MatMul.matmul(&hanzo_nn::ops::softmax_last_dim(&attn)?, v)
}

#[derive(Debug, Clone)]
struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    fn new(in_dim: usize, time_dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let linear_1 = linear(in_dim, time_dim, vb.pp("linear_1"))?;
        let linear_2 = linear(time_dim, time_dim, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.linear_1.forward(xs)?;
        let xs = xs.silu()?;
        self.linear_2.forward(&xs)
    }
}

#[derive(Debug, Clone)]
struct ResnetBlock2D {
    norm1: GroupNorm,
    conv1: Conv2d,
    time_emb_proj: Linear,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
}

impl ResnetBlock2D {
    fn new(
        in_c: usize,
        out_c: usize,
        time_dim: usize,
        cfg: &UNetConfig,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let conv_cfg = hanzo_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let norm1 = group_norm(cfg.norm_num_groups, in_c, cfg.norm_eps, vb.pp("norm1"))?;
        let conv1 = conv2d(in_c, out_c, 3, conv_cfg, vb.pp("conv1"))?;
        let time_emb_proj = linear(time_dim, out_c, vb.pp("time_emb_proj"))?;
        let norm2 = group_norm(cfg.norm_num_groups, out_c, cfg.norm_eps, vb.pp("norm2"))?;
        let conv2 = conv2d(out_c, out_c, 3, conv_cfg, vb.pp("conv2"))?;
        let conv_shortcut = if in_c == out_c {
            None
        } else {
            Some(conv2d(in_c, out_c, 1, Default::default(), vb.pp("conv_shortcut"))?)
        };
        Ok(Self {
            norm1,
            conv1,
            time_emb_proj,
            norm2,
            conv2,
            conv_shortcut,
        })
    }

    fn forward(&self, xs: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let mut h = self.norm1.forward(xs)?;
        h = h.silu()?;
        h = Convolution.forward_2d(&self.conv1, &h)?;
        let temb = self.time_emb_proj.forward(&temb.silu()?)?;
        h = h.broadcast_add(&temb.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?)?;
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
struct CrossAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: usize,
}

impl CrossAttention {
    fn new(
        query_dim: usize,
        context_dim: usize,
        heads: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let to_q = linear_no_bias(query_dim, query_dim, vb.pp("to_q"))?;
        let to_k = linear_no_bias(context_dim, query_dim, vb.pp("to_k"))?;
        let to_v = linear_no_bias(context_dim, query_dim, vb.pp("to_v"))?;
        let to_out = linear(query_dim, query_dim, vb.pp("to_out").pp(0))?;
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            heads,
        })
    }

    fn reshape_heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq, dim) = xs.dims3()?;
        let hd = dim / self.heads;
        xs.reshape((b, seq, self.heads, hd))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq, dim) = xs.dims3()?;
        let context = context.unwrap_or(xs);
        let q = self.to_q.forward(xs)?;
        let k = self.to_k.forward(context)?;
        let v = self.to_v.forward(context)?;
        let q = self.reshape_heads(&q)?;
        let k = self.reshape_heads(&k)?;
        let v = self.reshape_heads(&v)?;
        let out = scaled_dot_product_attention(&q, &k, &v)?;
        let out = out.transpose(1, 2)?.reshape((b, seq, dim))?;
        self.to_out.forward(&out)
    }
}

#[derive(Debug, Clone)]
struct GeGlu {
    proj: Linear,
}

impl GeGlu {
    fn new(dim_in: usize, dim_out: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let proj = linear(dim_in, dim_out * 2, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.proj.forward(xs)?;
        let chunks = h.chunk(2, D::Minus1)?;
        chunks[0].broadcast_mul(&chunks[1].gelu_erf()?)
    }
}

#[derive(Debug, Clone)]
struct FeedForward {
    geglu: GeGlu,
    proj: Linear,
}

impl FeedForward {
    fn new(dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let inner = dim * 4;
        let geglu = GeGlu::new(dim, inner, vb.pp("net").pp(0))?;
        let proj = linear(inner, dim, vb.pp("net").pp(2))?;
        Ok(Self { geglu, proj })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.geglu.forward(xs)?;
        self.proj.forward(&h)
    }
}

#[derive(Debug, Clone)]
struct BasicTransformerBlock {
    norm1: LayerNorm,
    attn1: CrossAttention,
    norm2: LayerNorm,
    attn2: CrossAttention,
    norm3: LayerNorm,
    ff: FeedForward,
}

impl BasicTransformerBlock {
    fn new(dim: usize, context_dim: usize, heads: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let norm1 = layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let attn1 = CrossAttention::new(dim, dim, heads, vb.pp("attn1"))?;
        let norm2 = layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        let attn2 = CrossAttention::new(dim, context_dim, heads, vb.pp("attn2"))?;
        let norm3 = layer_norm(dim, 1e-5, vb.pp("norm3"))?;
        let ff = FeedForward::new(dim, vb.pp("ff"))?;
        Ok(Self {
            norm1,
            attn1,
            norm2,
            attn2,
            norm3,
            ff,
        })
    }

    fn forward(&self, xs: &Tensor, context: &Tensor) -> Result<Tensor> {
        let xs = (self.attn1.forward(&self.norm1.forward(xs)?, None)? + xs)?;
        let xs = (self.attn2.forward(&self.norm2.forward(&xs)?, Some(context))? + &xs)?;
        self.ff.forward(&self.norm3.forward(&xs)?)? + xs
    }
}

#[derive(Debug, Clone)]
struct Transformer2D {
    norm: GroupNorm,
    proj_in: Linear,
    blocks: Vec<BasicTransformerBlock>,
    proj_out: Linear,
}

impl Transformer2D {
    fn new(channels: usize, cfg: &UNetConfig, vb: ShardedVarBuilder) -> Result<Self> {
        // diffusers UNet2DConditionModel reads `attention_head_dim` as the NUMBER of heads
        // (constant across blocks); per-block head_dim = channels / heads.
        let heads = cfg.attention_head_dim;
        let norm = group_norm(cfg.norm_num_groups, channels, 1e-6, vb.pp("norm"))?;
        let proj_in = linear(channels, channels, vb.pp("proj_in"))?;
        let blocks = vec![BasicTransformerBlock::new(
            channels,
            cfg.cross_attention_dim,
            heads,
            vb.pp("transformer_blocks").pp(0),
        )?];
        let proj_out = linear(channels, channels, vb.pp("proj_out"))?;
        Ok(Self {
            norm,
            proj_in,
            blocks,
            proj_out,
        })
    }

    fn forward(&self, xs: &Tensor, context: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let (b, c, h, w) = xs.dims4()?;
        let mut hidden = self.norm.forward(xs)?;
        hidden = hidden.reshape((b, c, h * w))?.transpose(1, 2)?;
        hidden = self.proj_in.forward(&hidden)?;
        for block in self.blocks.iter() {
            hidden = block.forward(&hidden, context)?;
        }
        hidden = self.proj_out.forward(&hidden)?;
        hidden = hidden.transpose(1, 2)?.reshape((b, c, h, w))?;
        hidden + residual
    }
}

#[derive(Debug, Clone)]
struct Downsample2D {
    conv: Conv2d,
}

impl Downsample2D {
    fn new(channels: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let conv_cfg = hanzo_nn::Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let conv = conv2d(channels, channels, 3, conv_cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Convolution.forward_2d(&self.conv, xs)
    }
}

#[derive(Debug, Clone)]
struct Upsample2D {
    conv: Conv2d,
}

impl Upsample2D {
    fn new(channels: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let conv_cfg = hanzo_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = conv2d(channels, channels, 3, conv_cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        let up = xs.upsample_nearest2d(h * 2, w * 2)?;
        Convolution.forward_2d(&self.conv, &up)
    }
}

#[derive(Debug, Clone)]
struct DownBlock {
    resnets: Vec<ResnetBlock2D>,
    attentions: Vec<Transformer2D>,
    downsampler: Option<Downsample2D>,
}

impl DownBlock {
    fn forward(
        &self,
        xs: &Tensor,
        temb: &Tensor,
        context: &Tensor,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut h = xs.clone();
        let mut residuals = Vec::new();
        for (i, resnet) in self.resnets.iter().enumerate() {
            h = resnet.forward(&h, temb)?;
            if let Some(attn) = self.attentions.get(i) {
                h = attn.forward(&h, context)?;
            }
            residuals.push(h.clone());
        }
        if let Some(ds) = self.downsampler.as_ref() {
            h = ds.forward(&h)?;
            residuals.push(h.clone());
        }
        Ok((h, residuals))
    }
}

#[derive(Debug, Clone)]
struct UpBlock {
    resnets: Vec<ResnetBlock2D>,
    attentions: Vec<Transformer2D>,
    upsampler: Option<Upsample2D>,
}

impl UpBlock {
    fn forward(
        &self,
        xs: &Tensor,
        temb: &Tensor,
        context: &Tensor,
        skips: &mut Vec<Tensor>,
    ) -> Result<Tensor> {
        let mut h = xs.clone();
        for (i, resnet) in self.resnets.iter().enumerate() {
            let skip = skips.pop().expect("up block expects matching skip connection");
            h = Tensor::cat(&[&h, &skip], 1)?;
            h = resnet.forward(&h, temb)?;
            if let Some(attn) = self.attentions.get(i) {
                h = attn.forward(&h, context)?;
            }
        }
        if let Some(us) = self.upsampler.as_ref() {
            h = us.forward(&h)?;
        }
        Ok(h)
    }
}

#[derive(Debug, Clone)]
struct MidBlock {
    resnet_1: ResnetBlock2D,
    attn: Transformer2D,
    resnet_2: ResnetBlock2D,
}

impl MidBlock {
    fn forward(&self, xs: &Tensor, temb: &Tensor, context: &Tensor) -> Result<Tensor> {
        let mut h = self.resnet_1.forward(xs, temb)?;
        h = self.attn.forward(&h, context)?;
        self.resnet_2.forward(&h, temb)
    }
}

#[derive(Debug, Clone)]
pub struct UNet2DConditionModel {
    conv_in: Conv2d,
    time_embedding: TimestepEmbedding,
    down_blocks: Vec<DownBlock>,
    mid_block: MidBlock,
    up_blocks: Vec<UpBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
    cfg: UNetConfig,
    temb: Tensor,
}

impl UNet2DConditionModel {
    pub fn new(cfg: &UNetConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let block_out = &cfg.block_out_channels;
        let base = block_out[0];
        let time_proj_dim = base;
        let time_embed_dim = base * 4;

        let conv_in_cfg = hanzo_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = conv2d(cfg.in_channels, base, 3, conv_in_cfg, vb.pp("conv_in"))?;
        let time_embedding =
            TimestepEmbedding::new(time_proj_dim, time_embed_dim, vb.pp("time_embedding"))?;

        let vb_down = vb.pp("down_blocks");
        let mut down_blocks = Vec::with_capacity(block_out.len());
        let mut prev_ch = base;
        let mut skip_channels: Vec<usize> = vec![base];
        for (i, &out_ch) in block_out.iter().enumerate() {
            let vb_i = vb_down.pp(i);
            let with_attn = cfg.has_cross_attn(&cfg.down_block_types[i]);
            let is_last = i == block_out.len() - 1;
            let mut resnets = Vec::with_capacity(cfg.layers_per_block);
            let mut attentions = Vec::new();
            let vb_res = vb_i.pp("resnets");
            let vb_attn = vb_i.pp("attentions");
            for j in 0..cfg.layers_per_block {
                let in_c = if j == 0 { prev_ch } else { out_ch };
                resnets.push(ResnetBlock2D::new(
                    in_c,
                    out_ch,
                    time_embed_dim,
                    cfg,
                    vb_res.pp(j),
                )?);
                if with_attn {
                    attentions.push(Transformer2D::new(out_ch, cfg, vb_attn.pp(j))?);
                }
                skip_channels.push(out_ch);
                prev_ch = out_ch;
            }
            let downsampler = if is_last {
                None
            } else {
                skip_channels.push(out_ch);
                Some(Downsample2D::new(
                    out_ch,
                    vb_i.pp("downsamplers").pp(0),
                )?)
            };
            down_blocks.push(DownBlock {
                resnets,
                attentions,
                downsampler,
            });
        }

        let mid_ch = *block_out.last().unwrap();
        let vb_mid = vb.pp("mid_block");
        let mid_block = MidBlock {
            resnet_1: ResnetBlock2D::new(
                mid_ch,
                mid_ch,
                time_embed_dim,
                cfg,
                vb_mid.pp("resnets").pp(0),
            )?,
            attn: Transformer2D::new(mid_ch, cfg, vb_mid.pp("attentions").pp(0))?,
            resnet_2: ResnetBlock2D::new(
                mid_ch,
                mid_ch,
                time_embed_dim,
                cfg,
                vb_mid.pp("resnets").pp(1),
            )?,
        };

        let vb_up = vb.pp("up_blocks");
        let mut up_blocks = Vec::with_capacity(block_out.len());
        let reversed: Vec<usize> = block_out.iter().rev().copied().collect();
        let mut prev_up = mid_ch;
        for (i, &out_ch) in reversed.iter().enumerate() {
            let vb_i = vb_up.pp(i);
            let with_attn = cfg.has_cross_attn(&cfg.up_block_types[i]);
            let is_last = i == reversed.len() - 1;
            let mut resnets = Vec::with_capacity(cfg.layers_per_block + 1);
            let mut attentions = Vec::new();
            let vb_res = vb_i.pp("resnets");
            let vb_attn = vb_i.pp("attentions");
            for j in 0..cfg.layers_per_block + 1 {
                let res_skip = skip_channels.pop().expect("up block skip channel underflow");
                let in_c = if j == 0 { prev_up + res_skip } else { out_ch + res_skip };
                resnets.push(ResnetBlock2D::new(
                    in_c,
                    out_ch,
                    time_embed_dim,
                    cfg,
                    vb_res.pp(j),
                )?);
                if with_attn {
                    attentions.push(Transformer2D::new(out_ch, cfg, vb_attn.pp(j))?);
                }
                prev_up = out_ch;
            }
            let upsampler = if is_last {
                None
            } else {
                Some(Upsample2D::new(out_ch, vb_i.pp("upsamplers").pp(0))?)
            };
            up_blocks.push(UpBlock {
                resnets,
                attentions,
                upsampler,
            });
        }

        let conv_norm_out = group_norm(
            cfg.norm_num_groups,
            base,
            cfg.norm_eps,
            vb.pp("conv_norm_out"),
        )?;
        let conv_out_cfg = hanzo_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_out = conv2d(base, cfg.out_channels, 3, conv_out_cfg, vb.pp("conv_out"))?;

        let dtype = conv_in.weight().dtype();
        let freqs = timestep_freqs(time_proj_dim / 2, vb.device())?;
        let zero_t = Tensor::zeros(1, DType::F32, vb.device())?;
        let temb_in = timestep_embedding(&zero_t, &freqs, cfg.flip_sin_to_cos, dtype)?;
        let temb = time_embedding.forward(&temb_in)?;

        Ok(Self {
            conv_in,
            time_embedding,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            cfg: cfg.clone(),
            temb,
        })
    }

    pub fn forward(&self, sample: &Tensor, _timestep: &Tensor, context: &Tensor) -> Result<Tensor> {
        let dtype = self.conv_in.weight().dtype();
        let sample = sample.to_dtype(dtype)?;
        let context = context.to_dtype(dtype)?;
        let temb = &self.temb;

        let mut h = Convolution.forward_2d(&self.conv_in, &sample)?;
        let mut skips: Vec<Tensor> = vec![h.clone()];
        for block in self.down_blocks.iter() {
            let (out, residuals) = block.forward(&h, &temb, &context)?;
            h = out;
            skips.extend(residuals);
        }

        h = self.mid_block.forward(&h, &temb, &context)?;

        for block in self.up_blocks.iter() {
            h = block.forward(&h, &temb, &context, &mut skips)?;
        }

        h = self.conv_norm_out.forward(&h)?;
        h = h.silu()?;
        Convolution.forward_2d(&self.conv_out, &h)
    }
}
