#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! TAESD (Tiny AutoEncoder for Stable Diffusion) decoder, madebyollin/taesd.
//! A distilled 3x3-conv + ReLU stack that replaces the full SD-VAE decoder: no GroupNorm, no
//! attention, ~1000x fewer params. Used as an optional fast decode path for MuseTalk; the encode
//! still uses the full VAE so the UNet latent space is unchanged. Quality parity requires the
//! distilled `taesd_decoder` weights trained against the same VAE latent space.

use hanzo_ml::{DType, Result, Tensor};
use hanzo_nn::{Conv2d, Conv2dConfig, Module};
use hanzo_quant::{Convolution, ShardedVarBuilder};

use crate::customconv;
use crate::layers::{conv2d, conv2d_no_bias};

const HIDDEN: usize = 64;

/// TAESD conv dispatch: the custom high-occupancy SIMT kernel when MUSETALK_CUSTOM_CONV=1 covers
/// the shape (3x3 f16 cuda), else the standard cuDNN/im2col path. `relu` fuses a trailing ReLU.
fn conv_fwd(layer: &Conv2d, x: &Tensor, relu: bool) -> Result<Tensor> {
    if customconv::covers(layer, x) {
        customconv::forward(layer, x, relu)
    } else {
        let h = Convolution.forward_2d(layer, x)?;
        if relu { h.relu() } else { Ok(h) }
    }
}

fn conv3(in_c: usize, out_c: usize, vb: ShardedVarBuilder) -> Result<Conv2d> {
    let cfg = Conv2dConfig {
        padding: 1,
        ..Default::default()
    };
    conv2d(in_c, out_c, 3, cfg, vb)
}

/// TAESD residual block: conv-relu-conv-relu-conv with a 1x1 (or identity) skip, fused with relu.
#[derive(Debug, Clone)]
struct Block {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    skip: Option<Conv2d>,
}

impl Block {
    fn new(in_c: usize, out_c: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let conv = vb.pp("conv");
        let conv1 = conv3(in_c, out_c, conv.pp(0))?;
        let conv2 = conv3(out_c, out_c, conv.pp(2))?;
        let conv3 = conv3(out_c, out_c, conv.pp(4))?;
        let skip = if in_c == out_c {
            None
        } else {
            Some(conv2d_no_bias(
                in_c,
                out_c,
                1,
                Default::default(),
                vb.pp("skip"),
            )?)
        };
        Ok(Self {
            conv1,
            conv2,
            conv3,
            skip,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = conv_fwd(&self.conv1, xs, true)?;
        let h = conv_fwd(&self.conv2, &h, true)?;
        let h = conv_fwd(&self.conv3, &h, false)?;
        let skip = match self.skip.as_ref() {
            None => xs.clone(),
            Some(c) => conv_fwd(c, xs, false)?,
        };
        (h + skip)?.relu()
    }
}

/// TAESD decoder: clamp -> conv(4->64) -> relu -> 3 upsample stages (3 blocks each) -> conv(64->3).
#[derive(Debug, Clone)]
pub struct TaesdDecoder {
    conv_in: Conv2d,
    blocks: Vec<Block>,
    upsamplers: Vec<Conv2d>,
    conv_out: Conv2d,
    scaling_factor: f64,
}

impl TaesdDecoder {
    pub fn new(
        latent_channels: usize,
        out_channels: usize,
        scaling_factor: f64,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        // Flat Sequential layout matching the reference: index 1 conv_in, then per stage 3 blocks +
        // upsample-conv, then a final block + conv_out. We index the Sequential explicitly.
        // diffusers AutoencoderTiny `decoder.layers`: conv_in(0), ReLU(1, paramless),
        // then per stage [3 Blocks, Upsample(paramless), up-conv], final Block, conv_out.
        let conv_in = conv3(latent_channels, HIDDEN, vb.pp(0))?;
        let mut idx = 2usize;
        let mut blocks = Vec::new();
        let mut upsamplers = Vec::new();
        for _stage in 0..3 {
            for _ in 0..3 {
                blocks.push(Block::new(HIDDEN, HIDDEN, vb.pp(idx))?);
                idx += 1;
            }
            idx += 1; // Upsample (parameter-free nn.Upsample)
            upsamplers.push(conv2d_no_bias(
                HIDDEN,
                HIDDEN,
                3,
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp(idx),
            )?);
            idx += 1;
        }
        blocks.push(Block::new(HIDDEN, HIDDEN, vb.pp(idx))?);
        idx += 1;
        let conv_out = conv3(HIDDEN, out_channels, vb.pp(idx))?;
        Ok(Self {
            conv_in,
            blocks,
            upsamplers,
            conv_out,
            scaling_factor,
        })
    }

    /// TAESD decodes RAW SD-VAE latents. The MuseTalk UNet emits latents in the full-VAE scaled
    /// space (encode multiplies by `scaling_factor`), so undo that scale before the tiny decoder.
    /// The 0-1 magnitude/shift transform in the TAESD repo is only for the uint8 preview, not the
    /// decoder input, so it is intentionally not applied here.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let mut h = (latents / self.scaling_factor)?.to_dtype(latents.dtype())?;
        h = conv_fwd(&self.conv_in, &h, true)?;
        let mut bi = 0usize;
        for up in self.upsamplers.iter() {
            for _ in 0..3 {
                h = self.blocks[bi].forward(&h)?;
                bi += 1;
            }
            let (_, _, hh, ww) = h.dims4()?;
            h = h.upsample_nearest2d(hh * 2, ww * 2)?;
            h = conv_fwd(up, &h, false)?;
        }
        h = self.blocks[bi].forward(&h)?;
        // TAESD's final conv emits the RGB image directly in [0,1] (no tanh/denorm needed).
        conv_fwd(&self.conv_out, &h, false)?
            .to_dtype(DType::F32)?
            .clamp(0f32, 1f32)
    }
}

/// TAESD encoder (madebyollin/taesd `taesd_encoder`): the mirror of `TaesdDecoder`.
/// `conv(3->64) -> Block -> 3x[stride-2 conv(64->64, no bias) + 3 Blocks] -> conv(64->4)`.
/// Maps a normalized [N,3,256,256] image to a scaled SD-VAE latent [N,4,32,32] (3 downsamples),
/// emitting `latent * scaling_factor` to match the full-VAE latent space the UNet expects.
/// Replaces the FULL SD-VAE encoder (the measured ~41ms / ~50% dominant per-frame stage) the same
/// way the decoder did for decode. Distilled `taesd_encoder` weights are required for quality
/// parity; the topology here is faithful for the perf measurement (seeded weights are shape-keyed).
#[derive(Debug, Clone)]
pub struct TaesdEncoder {
    conv_in: Conv2d,
    block_in: Block,
    downs: Vec<Conv2d>,
    blocks: Vec<Block>,
    conv_out: Conv2d,
    scaling_factor: f64,
}

impl TaesdEncoder {
    pub fn new(
        in_channels: usize,
        latent_channels: usize,
        scaling_factor: f64,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let conv_in = conv3(in_channels, HIDDEN, vb.pp(0))?;
        let block_in = Block::new(HIDDEN, HIDDEN, vb.pp(1))?;
        let mut idx = 2usize;
        let mut downs = Vec::new();
        let mut blocks = Vec::new();
        let down_cfg = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        for _stage in 0..3 {
            downs.push(conv2d_no_bias(HIDDEN, HIDDEN, 3, down_cfg, vb.pp(idx))?);
            idx += 1;
            for _ in 0..3 {
                blocks.push(Block::new(HIDDEN, HIDDEN, vb.pp(idx))?);
                idx += 1;
            }
        }
        let conv_out = conv3(HIDDEN, latent_channels, vb.pp(idx))?;
        Ok(Self {
            conv_in,
            block_in,
            downs,
            blocks,
            conv_out,
            scaling_factor,
        })
    }

    /// Encode an ALREADY-normalized image to the scaled SD-VAE latent space (mean path).
    pub fn encode(&self, img: &Tensor) -> Result<Tensor> {
        let mut h = conv_fwd(&self.conv_in, img, false)?;
        h = self.block_in.forward(&h)?;
        let mut bi = 0usize;
        for down in self.downs.iter() {
            h = conv_fwd(down, &h, false)?;
            for _ in 0..3 {
                h = self.blocks[bi].forward(&h)?;
                bi += 1;
            }
        }
        let latent = conv_fwd(&self.conv_out, &h, false)?;
        latent * self.scaling_factor
    }
}
