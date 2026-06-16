#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_quant::ShardedVarBuilder;

use super::config::MuseTalkConfig;
use super::taesd::{TaesdDecoder, TaesdEncoder};
use super::unet::UNet2DConditionModel;
use super::vae::AutoencoderKl;

const NORM_MEAN: f64 = 0.5;
const NORM_STD: f64 = 0.5;

/// Cached VAE latent of the static reference face, reused across all frames of a stream.
#[derive(Debug, Clone)]
pub struct RefLatents {
    latent: Tensor,
}

impl RefLatents {
    pub fn latent(&self) -> &Tensor {
        &self.latent
    }
}

pub struct MuseTalk {
    vae: AutoencoderKl,
    unet: UNet2DConditionModel,
    taesd: Option<TaesdDecoder>,
    taesd_enc: Option<TaesdEncoder>,
    cfg: MuseTalkConfig,
    device: Device,
    dtype: DType,
    mask: Tensor,
    timestep: Tensor,
}

impl MuseTalk {
    pub fn new(
        cfg: MuseTalkConfig,
        vae_vb: ShardedVarBuilder,
        unet_vb: ShardedVarBuilder,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vae = AutoencoderKl::new(&cfg.vae, vae_vb)?;
        let unet = UNet2DConditionModel::new(&cfg.unet, unet_vb)?;
        let mask = Self::build_mask(cfg.resized_img, device, dtype)?;
        let timestep = Tensor::zeros(1, DType::F32, device)?;
        Ok(Self {
            vae,
            unet,
            taesd: None,
            taesd_enc: None,
            cfg,
            device: device.clone(),
            dtype,
            mask,
            timestep,
        })
    }

    /// Attach a TAESD decoder to use as the fast decode path. The full VAE encoder is still used,
    /// so the UNet latent space is unchanged; only `vae.decode` is replaced by the tiny decoder.
    pub fn with_taesd(mut self, taesd_vb: ShardedVarBuilder) -> Result<Self> {
        // sf=1.0: AutoencoderTiny consumes the KL-scaled latent directly (swarm-verified).
        let taesd = TaesdDecoder::new(
            self.cfg.vae.latent_channels,
            self.cfg.vae.out_channels,
            1.0,
            taesd_vb,
        )?;
        self.taesd = Some(taesd);
        Ok(self)
    }

    pub fn has_taesd(&self) -> bool {
        self.taesd.is_some()
    }

    /// Attach a TAESD tiny-ENCODER as the fast per-frame masked-face encode path (mirror of
    /// `with_taesd`). Replaces the full SD-VAE encoder, the measured dominant per-frame stage.
    pub fn with_taesd_encoder(mut self, vb: ShardedVarBuilder) -> Result<Self> {
        // TAESD AutoencoderTiny latents already live in the KL scaled (0.18215) space, so the
        // tiny-encoder scale is 1.0 (verified: latent cosine 0.9922 vs the PyTorch KL latent).
        let enc = TaesdEncoder::new(
            self.cfg.vae.out_channels,
            self.cfg.vae.latent_channels,
            1.0,
            vb,
        )?;
        self.taesd_enc = Some(enc);
        Ok(self)
    }

    pub fn has_taesd_encoder(&self) -> bool {
        self.taesd_enc.is_some()
    }

    /// Like `latents_for_unet_with_ref` but encodes the masked frame with the TAESD tiny-encoder
    /// when attached (falls back to the full VAE encode otherwise). Reference latent is reused.
    pub fn latents_for_unet_fast(
        &self,
        face: &Tensor,
        reference: &RefLatents,
    ) -> Result<Tensor> {
        let masked_raw = face.broadcast_mul(&self.mask.unsqueeze(0)?.unsqueeze(0)?)?;
        let masked_latents = match self.taesd_enc.as_ref() {
            // TAESD AutoencoderTiny expects a [0,1] image (verified: [0,1]-in cosine 0.9922 vs
            // [-1,1]-in 0.9145). Feed the raw masked face; the full VAE path needs normalize.
            Some(e) => e.encode(&masked_raw.to_dtype(self.dtype)?)?,
            None => self.vae.encode_mode(&self.normalize(&masked_raw)?)?,
        };
        let ref_latents = if reference.latent.dim(0)? == masked_latents.dim(0)? {
            reference.latent.clone()
        } else {
            reference
                .latent
                .broadcast_as(masked_latents.shape())?
                .contiguous()?
        };
        Tensor::cat(&[&masked_latents, &ref_latents], 1)
    }

    /// Decode latents through TAESD if attached, else fall back to the full VAE decoder.
    /// TAESD already emits the image in [0,1]; the full-VAE path needs denormalize. Both return
    /// an [N,3,H,W] f32 image in [0,1].
    pub fn decode_latents_fast(&self, pred_latents: &Tensor) -> Result<Tensor> {
        match self.taesd.as_ref() {
            Some(t) => t.decode(pred_latents),
            None => self.decode_latents(pred_latents),
        }
    }

    fn build_mask(size: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let top = Tensor::ones((size / 2, size), dtype, device)?;
        let bottom = Tensor::zeros((size - size / 2, size), dtype, device)?;
        Tensor::cat(&[top, bottom], 0)
    }

    fn normalize(&self, img: &Tensor) -> Result<Tensor> {
        ((img - NORM_MEAN)? / NORM_STD)?.to_dtype(self.dtype)
    }

    fn denormalize(&self, img: &Tensor) -> Result<Tensor> {
        ((img.to_dtype(DType::F32)? * NORM_STD)? + NORM_MEAN)?.clamp(0f32, 1f32)
    }

    pub fn latents_for_unet(&self, face: &Tensor) -> Result<Tensor> {
        // PyTorch MuseTalk masks the face in [0,1] space (lower half -> 0) and THEN
        // normalizes ((x-0.5)/0.5), so the masked region becomes -1.0. Masking AFTER
        // normalize (the previous order) left the masked half at 0.0 (mid-gray), which
        // made the UNet inpaint a blank gray mouth (LSE-C ~0.1). Mask raw, then normalize.
        let b = face.dim(0)?;
        let masked_raw = face.broadcast_mul(&self.mask.unsqueeze(0)?.unsqueeze(0)?)?;
        let masked = self.normalize(&masked_raw)?;
        let face = self.normalize(face)?;
        let latents = self.vae.encode_mode(&Tensor::cat(&[&masked, &face], 0)?)?;
        let masked_latents = latents.narrow(0, 0, b)?;
        let ref_latents = latents.narrow(0, b, b)?;
        Tensor::cat(&[masked_latents, ref_latents], 1)
    }

    /// VAE-encode just the (static) reference face into its latent, for caching across a stream.
    /// The reference latent is the unmasked-face half of the UNet input; in streaming it never
    /// changes, so encoding it once and reusing it halves the per-frame VAE-encode work.
    pub fn encode_reference(&self, face: &Tensor) -> Result<RefLatents> {
        let face = self.normalize(face)?;
        let latent = self.vae.encode_mode(&face)?;
        Ok(RefLatents { latent })
    }

    /// Like `latents_for_unet` but encodes only the masked frame (N images instead of 2N),
    /// concatenating the precomputed reference latent. Byte-identical to `latents_for_unet`
    /// for the same reference face (the VAE-encode is per-image, no cross-frame mixing).
    pub fn latents_for_unet_with_ref(
        &self,
        face: &Tensor,
        reference: &RefLatents,
    ) -> Result<Tensor> {
        // mask in [0,1] space then normalize (lower half -> -1.0), matching PyTorch.
        let masked_raw = face.broadcast_mul(&self.mask.unsqueeze(0)?.unsqueeze(0)?)?;
        let masked = self.normalize(&masked_raw)?;
        let masked_latents = self.vae.encode_mode(&masked)?;
        let ref_latents = if reference.latent.dim(0)? == masked_latents.dim(0)? {
            reference.latent.clone()
        } else {
            reference
                .latent
                .broadcast_as(masked_latents.shape())?
                .contiguous()?
        };
        Tensor::cat(&[&masked_latents, &ref_latents], 1)
    }

    /// Streaming forward with a cached reference latent: encodes only the masked frame.
    pub fn forward_streaming(
        &self,
        face: &Tensor,
        audio_feat: &Tensor,
        reference: &RefLatents,
    ) -> Result<Tensor> {
        let latent_input = self.latents_for_unet_with_ref(face, reference)?;
        let pred_latents = self
            .unet
            .forward(&latent_input, &self.timestep, audio_feat)?;
        let image = self.vae.decode(&pred_latents)?;
        self.denormalize(&image)
    }

    pub fn forward(&self, face: &Tensor, audio_feat: &Tensor) -> Result<Tensor> {
        let latent_input = self.latents_for_unet(face)?;
        let pred_latents = self
            .unet
            .forward(&latent_input, &self.timestep, audio_feat)?;
        let image = self.vae.decode(&pred_latents)?;
        self.denormalize(&image)
    }

    /// Process N frames in a single forward. `faces` is [N,3,H,W], `audio_feat` is [N,seq,dim]
    /// (one audio context per frame). The VAE-encode runs on the 2N masked+ref stack, the UNet
    /// single-step and VAE-decode run on the N-frame batch. Output is [N,3,H,W], identical
    /// per-frame to calling `forward` once per frame.
    pub fn forward_batched(&self, faces: &Tensor, audio_feat: &Tensor) -> Result<Tensor> {
        let latent_input = self.latents_for_unet(faces)?;
        let pred_latents = self
            .unet
            .forward(&latent_input, &self.timestep, audio_feat)?;
        let image = self.vae.decode(&pred_latents)?;
        self.denormalize(&image)
    }

    pub fn unet_forward(
        &self,
        latent_input: &Tensor,
        timestep: &Tensor,
        audio_feat: &Tensor,
    ) -> Result<Tensor> {
        self.unet.forward(latent_input, timestep, audio_feat)
    }

    pub fn decode_latents(&self, pred_latents: &Tensor) -> Result<Tensor> {
        let image = self.vae.decode(pred_latents)?;
        self.denormalize(&image)
    }

    /// VAE-encode an ALREADY-normalized image (mean=std=0.5 applied by caller) to its scaled
    /// latent mean. Matches PyTorch `scaling * vae.encode(x).latent_dist.mode()`.
    pub fn vae_encode_mode(&self, normalized_img: &Tensor) -> Result<Tensor> {
        self.vae.encode_mode(&normalized_img.to_dtype(self.dtype)?)
    }

    pub fn vae_encode_mode_debug(&self, normalized_img: &Tensor, dir: &str) -> Result<Tensor> {
        self.vae
            .encode_mode_debug(&normalized_img.to_dtype(self.dtype)?, dir)
    }

    /// VAE-decode latents to the RAW decoder output (pre-denormalize, in ~[-1,1]).
    /// Matches PyTorch `vae.decode(latents / scaling).sample`.
    pub fn vae_decode_raw(&self, pred_latents: &Tensor) -> Result<Tensor> {
        self.vae.decode(pred_latents)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn blend(&self, original: &Tensor, generated: &Tensor) -> Result<Tensor> {
        let mask = self.mask.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::F32)?;
        let lower = (1f64 - &mask)?;
        let orig = original.to_dtype(DType::F32)?;
        let gen = generated.to_dtype(DType::F32)?;
        orig.broadcast_mul(&mask)? + gen.broadcast_mul(&lower)?
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn cross_attention_dim(&self) -> usize {
        self.cfg.unet.cross_attention_dim
    }

    pub fn latent_size(&self) -> usize {
        self.cfg.unet.sample_size
    }

    pub fn resized_img(&self) -> usize {
        self.cfg.resized_img
    }
}

pub fn audio_feature_seq_len() -> usize {
    50
}

pub fn reshape_whisper_chunk(chunk: &Tensor) -> Result<Tensor> {
    let dim = chunk.dim(D::Minus1)?;
    chunk.reshape(((), dim))
}
