//! Env-gated custom CUDA conv path for the TAESD tiny-encoder/decoder convs.
//!
//! The dub's TAESD 3x3 convs are low-channel-count (C<=64) at high resolution (256/128/64). cuDNN
//! routes them to implicit-GEMM, which underfills the tensor-core MMA tiles and runs far below peak.
//! This module dispatches those convs to a hand-written high-occupancy SIMT kernel
//! (`taesd_conv3x3_*` in hanzo-kernels/src/taesd_conv.cuh) that stages the input tile in shared
//! memory and fuses the per-channel bias + ReLU into the same launch.
//!
//! Gated by MUSETALK_CUSTOM_CONV=1. Falls back to the standard cuDNN/im2col path otherwise, and for
//! any shape the kernel doesn't cover (non-3x3, dtype != f16, CPU).
//!
//! The custom kernel is CUDA-only (cudarc + hanzo-kernels). On non-CUDA backends (Metal, Vulkan,
//! CPU) this module compiles to a stub: `covers()` is always false and `forward()` bails, so the
//! TAESD conv dispatch in `taesd.rs` always takes the standard `hanzo_quant::Convolution` path.

#[cfg(feature = "cuda")]
use half::f16;
#[cfg(feature = "cuda")]
use hanzo_ml::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use hanzo_ml::backend::BackendStorage;
#[cfg(feature = "cuda")]
use hanzo_ml::cuda_backend::{kernels, CudaStorage};
#[cfg(feature = "cuda")]
use hanzo_ml::{CpuStorage, CustomOp2, Layout, Shape};
use hanzo_ml::{DType, Result, Tensor};
use hanzo_nn::Conv2d;

#[cfg(feature = "cuda")]
const TH: u32 = 8;
#[cfg(feature = "cuda")]
const TW: u32 = 16;
#[cfg(feature = "cuda")]
const SMEM_STATIC_CAP: usize = 48 * 1024;

pub fn enabled() -> bool {
    std::env::var("MUSETALK_CUSTOM_CONV").is_ok()
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Variant {
    Simple,
    Ocb4,
    Reg,
}

#[cfg(feature = "cuda")]
fn pick_variant(c_out: usize, stride: usize) -> Variant {
    // OCB4 register-blocks 4 output channels per shared read; best when C_out is a multiple of 4
    // and large enough to amortize (the dominant 64->64 convs). Fall back to the simple kernel
    // otherwise (e.g. C_out=3 conv_in or any non-multiple-of-4 width).
    match std::env::var("MUSETALK_CONV_VARIANT").as_deref() {
        Ok("simple") => Variant::Simple,
        Ok("ocb4") => Variant::Ocb4,
        Ok("reg") => Variant::Reg,
        _ => {
            // Register-blocked v2 only covers stride-1; needs C_out multiple of 8 for OCB8.
            if false {
                Variant::Reg // DISABLED: OOB crash (illegal addr); SIMT loses to cuDNN regardless
            } else if c_out >= 16 && c_out % 4 == 0 {
                Variant::Ocb4
            } else {
                Variant::Simple
            }
        }
    }
}

#[cfg(feature = "cuda")]
struct TaesdConv2d {
    bias: Option<Tensor>,
    stride: usize,
    relu: bool,
}

#[cfg(feature = "cuda")]
impl CustomOp2 for TaesdConv2d {
    fn name(&self) -> &'static str {
        "taesd_conv3x3"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        hanzo_ml::bail!("taesd_conv3x3 has no cpu implementation (cuda-only fast path)")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        w: &CudaStorage,
        w_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let (n, c_in, h, w_in) = x_l.shape().dims4()?;
        let (c_out, c_in_k, k_h, k_w) = w_l.shape().dims4()?;
        debug_assert_eq!(c_in, c_in_k);
        debug_assert_eq!((k_h, k_w), (3, 3));
        let stride = self.stride;
        let h_out = (h + 2 - 3) / stride + 1;
        let w_out = (w_in + 2 - 3) / stride + 1;

        let x_s = x.as_cuda_slice::<f16>()?;
        let x_s = x_s.slice(x_l.start_offset()..);
        let w_s = w.as_cuda_slice::<f16>()?;
        let w_s = w_s.slice(w_l.start_offset()..);

        let out_el = n * c_out * h_out * w_out;
        let out = unsafe { dev.alloc::<f16>(out_el)? };

        let halo_h = TH as usize * stride + 2;
        let halo_w = TW as usize * stride + 2;
        let smem_bytes = c_in * halo_h * halo_w * std::mem::size_of::<f16>();

        let variant = pick_variant(c_out, stride);
        let fname = match variant {
            Variant::Simple => "taesd_conv3x3_f16",
            Variant::Ocb4 => "taesd_conv3x3_ocb4_f16",
            Variant::Reg => "taesd_conv3x3_reg_f16",
        };
        let func = dev.get_or_load_func(fname, &kernels::CONV)?;
        if smem_bytes > SMEM_STATIC_CAP {
            use hanzo_ml::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum as Attr;
            func.set_attribute(
                Attr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem_bytes as i32,
            )
            .map_err(hanzo_ml::Error::wrap)?;
        }

        let grid = (
            (w_out as u32).div_ceil(TW),
            (h_out as u32).div_ceil(TH),
            n as u32,
        );
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (TW, TH, 1),
            shared_mem_bytes: smem_bytes as u32,
        };

        let has_bias = self.bias.is_some();
        let bias_storage = self.bias.as_ref().map(|b| b.storage_and_layout());
        // Keep the read guard alive across the launch; extract the slice through it.
        let bias_slice = match &bias_storage {
            Some((s, _)) => match &**s {
                hanzo_ml::Storage::Cuda(cs) => Some(cs.as_cuda_slice::<f16>()?.slice(..)),
                _ => hanzo_ml::bail!("taesd_conv3x3 bias must be a cuda f16 tensor"),
            },
            None => None,
        };
        // A non-null placeholder for the no-bias case (kernel guards on the do-bias flag via ptr).
        let dummy = unsafe { dev.alloc::<f16>(c_out)? };

        let mut builder = func.builder();
        builder.arg(&x_s);
        builder.arg(&w_s);
        match &bias_slice {
            Some(b) => builder.arg(b),
            None => builder.arg(&dummy),
        };
        builder.arg(&out);
        let n_i = n as i32;
        let ci_i = c_in as i32;
        let co_i = c_out as i32;
        let h_i = h as i32;
        let w_i = w_in as i32;
        let ho_i = h_out as i32;
        let wo_i = w_out as i32;
        let stride_i = stride as i32;
        let relu_i = if self.relu { 1i32 } else { 0i32 };
        let bias_flag = if has_bias { 1i32 } else { 0i32 };
        builder.arg(&n_i);
        builder.arg(&ci_i);
        builder.arg(&co_i);
        builder.arg(&h_i);
        builder.arg(&w_i);
        builder.arg(&ho_i);
        builder.arg(&wo_i);
        builder.arg(&stride_i);
        builder.arg(&relu_i);
        builder.arg(&bias_flag);
        unsafe { builder.launch(cfg) }.map_err(hanzo_ml::Error::wrap)?;

        let storage = CudaStorage::wrap_cuda_slice(out, dev);
        Ok((storage, Shape::from((n, c_out, h_out, w_out))))
    }
}

/// Whether the custom kernel covers this conv (3x3, f16, cuda, groups=1, dilation=1).
#[cfg(feature = "cuda")]
pub fn covers(layer: &Conv2d, x: &Tensor) -> bool {
    if !enabled() || !x.device().is_cuda() || x.dtype() != DType::F16 {
        return false;
    }
    let cfg = layer.config();
    let (_, _, k_h, k_w) = match layer.weight().dims4() {
        Ok(d) => d,
        Err(_) => return false,
    };
    cfg.groups == 1 && cfg.dilation == 1 && k_h == 3 && k_w == 3 && (cfg.stride == 1 || cfg.stride == 2)
}

/// Non-CUDA stub: the custom SIMT kernel never applies, so always take the standard conv path.
#[cfg(not(feature = "cuda"))]
pub fn covers(_layer: &Conv2d, _x: &Tensor) -> bool {
    let _ = DType::F16; // keep DType import used on all backends
    false
}

/// Run the custom conv. `relu` fuses a ReLU into the epilogue. Caller must check `covers` first.
#[cfg(feature = "cuda")]
pub fn forward(layer: &Conv2d, x: &Tensor, relu: bool) -> Result<Tensor> {
    let op = TaesdConv2d {
        bias: layer.bias().cloned(),
        stride: layer.config().stride,
        relu,
    };
    x.apply_op2_no_bwd(layer.weight(), &op)
}

/// Non-CUDA stub: `covers()` is always false here, so this is never reached on Metal/Vulkan/CPU.
#[cfg(not(feature = "cuda"))]
pub fn forward(_layer: &Conv2d, _x: &Tensor, _relu: bool) -> Result<Tensor> {
    hanzo_ml::bail!("customconv::forward is a cuda-only fast path (no metal/vulkan/cpu impl)")
}
