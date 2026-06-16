//! MPSGraph-backed 2D convolution for the Metal backend.
//!
//! The default Metal `conv2d` (see `mod.rs`) is im2col + simdgroup-matmul + an NHWC->NCHW
//! transpose: three kernel launches and a ~(k_h*k_w)x materialization of the input column buffer
//! per conv. For the low-channel high-resolution convs that dominate MuseTalk's VAE/TAESD/UNet
//! this im2col bandwidth tax is the e2e bottleneck (the same problem CUDA had before cuDNN's
//! implicit-GEMM). This module routes such convs to Apple's MPSGraph `convolution2D`, a fused,
//! Apple-tuned conv that never materializes the im2col buffer.
//!
//! Gated by `HANZO_METAL_MPS=1` (alias `MUSETALK_METAL_MPS=1`) so it can be A/B'd against the
//! im2col path. It only covers groups==1, f16/f32, contiguous-from-offset-0 NCHW input + OIHW
//! weights; anything else returns `Ok(None)` and the caller falls back to im2col.
//!
//! Numerics: MPSGraph conv is a direct/winograd conv, the im2col path is a GEMM; both are
//! IEEE-accumulated so outputs match to >0.999 cosine (validated by the `convbench-metal`
//! and `realverify` paths).

use crate::backend::BackendStorage;
use crate::conv::ParamsConv2D;
use crate::metal_backend::{MetalError, MetalStorage};
use crate::{DType, Layout, Result};

use std::os::raw::c_ulong;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::{NSArray, NSDictionary, NSNumber};
use hanzo_metal_kernels::metal::Buffer;
use objc2_metal::MTLBuffer;
use objc2_metal_performance_shaders::{MPSDataType, MPSShape};
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphConvolution2DOpDescriptor, MPSGraphPaddingStyle, MPSGraphTensor,
    MPSGraphTensorData, MPSGraphTensorNamedDataLayout,
};

/// `HANZO_METAL_MPS=1` (or the `MUSETALK_METAL_MPS=1` alias) enables the MPSGraph conv path.
pub fn enabled() -> bool {
    std::env::var("HANZO_METAL_MPS").is_ok() || std::env::var("MUSETALK_METAL_MPS").is_ok()
}

fn mps_dtype(dt: DType) -> Option<MPSDataType> {
    match dt {
        DType::F16 => Some(MPSDataType::Float16),
        DType::F32 => Some(MPSDataType::Float32),
        _ => None,
    }
}

/// Build an `MPSShape` (NSArray<NSNumber>) from usize dims.
fn shape(dims: &[usize]) -> Retained<MPSShape> {
    let nums: Vec<Retained<NSNumber>> = dims
        .iter()
        .map(|&d| unsafe { NSNumber::numberWithUnsignedLong(d as c_ulong) })
        .collect();
    let refs: Vec<&NSNumber> = nums.iter().map(|n| n.as_ref()).collect();
    NSArray::from_slice(&refs)
}

/// Get the raw `MTLBuffer` protocol object behind a candle `Buffer`.
fn mtl_buffer(buf: &Buffer) -> &ProtocolObject<dyn MTLBuffer> {
    AsRef::<ProtocolObject<dyn MTLBuffer>>::as_ref(buf)
}

/// Wrap an existing candle MTLBuffer (contiguous, offset 0) as an MPSGraphTensorData of `dims`.
fn tensor_data(
    buf: &ProtocolObject<dyn MTLBuffer>,
    dims: &[usize],
    dt: MPSDataType,
) -> Retained<MPSGraphTensorData> {
    let sh = shape(dims);
    unsafe {
        MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
            MPSGraphTensorData::alloc(),
            buf,
            &sh,
            dt,
        )
    }
}

/// Attempt an MPSGraph conv2d. Returns `Ok(None)` if this conv isn't covered (caller falls back
/// to the im2col path). Output is contiguous NCHW `[b, c_out, h_out, w_out]`, same as the
/// im2col path's final transpose.
#[allow(clippy::too_many_arguments)]
pub fn try_conv2d(
    input: &MetalStorage,
    layout: &Layout,
    kernel: &MetalStorage,
    kernel_l: &Layout,
    params: &ParamsConv2D,
) -> Result<Option<MetalStorage>> {
    if !enabled() {
        return Ok(None);
    }
    // candle's ParamsConv2D carries no groups field: the backend conv2d is always a plain
    // (groups==1) convolution, so no grouped-conv guard is needed here.
    let dt = input.dtype();
    if kernel.dtype() != dt {
        return Ok(None);
    }
    let mdt = match mps_dtype(dt) {
        Some(d) => d,
        None => return Ok(None),
    };
    // Inputs must be contiguous from offset 0: MPSGraphTensorData reads the raw buffer linearly.
    if !layout.is_contiguous()
        || layout.start_offset() != 0
        || !kernel_l.is_contiguous()
        || kernel_l.start_offset() != 0
    {
        return Ok(None);
    }

    let dims = layout.shape().dims();
    if dims.len() != 4 {
        return Ok(None);
    }
    let (b, c_in, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let kdims = kernel_l.shape().dims(); // [c_out, c_in, k_h, k_w]
    let (c_out, k_h, k_w) = (kdims[0], kdims[2], kdims[3]);
    let h_out = params.out_h();
    let w_out = params.out_w();

    let device = input.device().clone();

    // Build the graph: placeholders for src (NCHW) + weights (OIHW), one convolution2D node.
    let graph = unsafe { MPSGraph::new() };
    let desc = unsafe {
        MPSGraphConvolution2DOpDescriptor::descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout(
            params.stride,
            params.stride,
            params.dilation,
            params.dilation,
            1, // groups
            params.padding,
            params.padding,
            params.padding,
            params.padding,
            MPSGraphPaddingStyle::Explicit,
            MPSGraphTensorNamedDataLayout::NCHW,
            MPSGraphTensorNamedDataLayout::OIHW,
        )
    }
    .ok_or_else(|| MetalError::Message("MPSGraph conv descriptor alloc failed".into()))?;

    let src_ph: Retained<MPSGraphTensor> = unsafe {
        graph.placeholderWithShape_dataType_name(Some(&shape(&[b, c_in, h, w])), mdt, None)
    };
    let w_ph: Retained<MPSGraphTensor> = unsafe {
        graph.placeholderWithShape_dataType_name(
            Some(&shape(&[c_out, c_in, k_h, k_w])),
            mdt,
            None,
        )
    };
    let out_t: Retained<MPSGraphTensor> = unsafe {
        graph.convolution2DWithSourceTensor_weightsTensor_descriptor_name(
            &src_ph, &w_ph, &desc, None,
        )
    };

    // Flush any pending candle work that produced these input buffers, so MPSGraph (which runs on
    // its own command queue) reads up-to-date contents.
    device.wait_until_completed()?;

    // Bind the existing candle input/weight buffers as feeds, and a fresh output buffer as the
    // result target -> MPSGraph writes straight into our candle buffer (stays on-GPU, no copy).
    let src_buf = input.buffer();
    let w_buf = kernel.buffer();
    let src_td = tensor_data(mtl_buffer(src_buf), &[b, c_in, h, w], mdt);
    let w_td = tensor_data(mtl_buffer(w_buf), &[c_out, c_in, k_h, k_w], mdt);

    let out_el = b * c_out * h_out * w_out;
    let out_buf = device.new_buffer(out_el, dt, "conv2d_mps")?;
    let out_td = tensor_data(mtl_buffer(&out_buf), &[b, c_out, h_out, w_out], mdt);

    let src_key: &MPSGraphTensor = &src_ph;
    let w_key: &MPSGraphTensor = &w_ph;
    let feeds: Retained<NSDictionary<MPSGraphTensor, MPSGraphTensorData>> =
        NSDictionary::from_slices(&[src_key, w_key], &[&*src_td, &*w_td]);
    let out_key: &MPSGraphTensor = &out_t;
    let results: Retained<NSDictionary<MPSGraphTensor, MPSGraphTensorData>> =
        NSDictionary::from_slices(&[out_key], &[&*out_td]);

    // A dedicated command queue for MPSGraph on the same device. Cached per-device (queue creation
    // is not free); MPSGraph serializes against candle's queue via the wait_until_completed() above.
    let queue = device.mps_command_queue()?;
    unsafe {
        graph.runWithMTLCommandQueue_feeds_targetOperations_resultsDictionary(
            queue.as_ref(),
            &feeds,
            None,
            &results,
        );
    }

    Ok(Some(MetalStorage::new(out_buf, device, out_el, dt)))
}
