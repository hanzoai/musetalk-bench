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
//! The compiled graph is cached per (shape, dtype, conv-params): MPSGraph build+compile is not
//! free, and re-paying it on every one of the UNet's many small convs is a net regression. With
//! caching only the placeholder feeds + output buffer are rebuilt per call.
//!
//! Numerics: MPSGraph conv is a direct/winograd conv, the im2col path is a GEMM; both are
//! IEEE-accumulated so outputs match to >0.999 cosine (validated by the `convbench-metal`
//! and `realverify` paths; observed cosine 1.000000).

use crate::backend::BackendStorage;
use crate::conv::ParamsConv2D;
use crate::metal_backend::{MetalError, MetalStorage};
use crate::{DType, Layout, Result};

use std::collections::HashMap;
use std::os::raw::c_ulong;

use hanzo_metal_kernels::metal::Buffer;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::{NSArray, NSDictionary, NSNumber};
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
        MPSGraphTensorData::initWithMTLBuffer_shape_dataType(MPSGraphTensorData::alloc(), buf, &sh, dt)
    }
}

/// Identity of a compiled conv graph: everything that changes its structure.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConvKey {
    b: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    k_h: usize,
    k_w: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    dt: DType,
}

/// A compiled MPSGraph conv plus its placeholder/result tensor handles. Reused across calls with
/// the same `ConvKey`; only the feed/result `MPSGraphTensorData` (which wrap the live candle
/// buffers) are rebuilt per call.
pub struct ConvGraph {
    graph: Retained<MPSGraph>,
    src_ph: Retained<MPSGraphTensor>,
    w_ph: Retained<MPSGraphTensor>,
    out_t: Retained<MPSGraphTensor>,
}

/// Per-device cache of compiled conv graphs. The handles are objc2 Metal objects; like candle's
/// own `Commands`, they are used behind the device's locks, so we assert Send+Sync (Metal protocol
/// objects are internally thread-safe and we never mutate them after build).
#[derive(Default)]
pub struct ConvGraphCache {
    map: HashMap<ConvKey, ConvGraph>,
}
unsafe impl Send for ConvGraphCache {}
unsafe impl Sync for ConvGraphCache {}

fn build_graph(key: &ConvKey, mdt: MPSDataType) -> Result<ConvGraph> {
    let graph = unsafe { MPSGraph::new() };
    let desc = unsafe {
        MPSGraphConvolution2DOpDescriptor::descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout(
            key.stride,
            key.stride,
            key.dilation,
            key.dilation,
            1, // groups (candle conv2d is always groups==1)
            key.padding,
            key.padding,
            key.padding,
            key.padding,
            MPSGraphPaddingStyle::Explicit,
            MPSGraphTensorNamedDataLayout::NCHW,
            MPSGraphTensorNamedDataLayout::OIHW,
        )
    }
    .ok_or_else(|| MetalError::Message("MPSGraph conv descriptor alloc failed".into()))?;

    let src_ph = unsafe {
        graph.placeholderWithShape_dataType_name(
            Some(&shape(&[key.b, key.c_in, key.h, key.w])),
            mdt,
            None,
        )
    };
    let w_ph = unsafe {
        graph.placeholderWithShape_dataType_name(
            Some(&shape(&[key.c_out, key.c_in, key.k_h, key.k_w])),
            mdt,
            None,
        )
    };
    let out_t = unsafe {
        graph.convolution2DWithSourceTensor_weightsTensor_descriptor_name(&src_ph, &w_ph, &desc, None)
    };
    Ok(ConvGraph {
        graph,
        src_ph,
        w_ph,
        out_t,
    })
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
    let kdims = kernel_l.shape().dims(); // [c_out, c_in, k_h, k_w]
    let key = ConvKey {
        b: dims[0],
        c_in: dims[1],
        h: dims[2],
        w: dims[3],
        c_out: kdims[0],
        k_h: kdims[2],
        k_w: kdims[3],
        stride: params.stride,
        padding: params.padding,
        dilation: params.dilation,
        dt,
    };
    let h_out = params.out_h();
    let w_out = params.out_w();
    let device = input.device().clone();

    // Flush any pending candle work that produced these input buffers, so MPSGraph (which runs on
    // its own command queue) reads up-to-date contents.
    device.wait_until_completed()?;

    let src_buf = input.buffer();
    let w_buf = kernel.buffer();
    let src_td = tensor_data(mtl_buffer(src_buf), &[key.b, key.c_in, key.h, key.w], mdt);
    let w_td = tensor_data(mtl_buffer(w_buf), &[key.c_out, key.c_in, key.k_h, key.k_w], mdt);

    let out_el = key.b * key.c_out * h_out * w_out;
    let out_buf = device.new_buffer(out_el, dt, "conv2d_mps")?;
    let out_td = tensor_data(mtl_buffer(&out_buf), &[key.b, key.c_out, h_out, w_out], mdt);

    let queue = device.mps_command_queue()?;

    // get-or-build the compiled graph for this shape, then run it. Hold the cache lock only across
    // the (cheap) feed/result dict build + run; the graph objects are immutable once built.
    {
        let mut cache = device.mps_conv_cache.write().map_err(MetalError::from)?;
        let entry = match cache.map.get(&key) {
            Some(e) => e,
            None => {
                let g = build_graph(&key, mdt)?;
                cache.map.insert(key, g);
                cache.map.get(&key).unwrap()
            }
        };

        let feeds: Retained<NSDictionary<MPSGraphTensor, MPSGraphTensorData>> =
            NSDictionary::from_slices(&[&*entry.src_ph, &*entry.w_ph], &[&*src_td, &*w_td]);
        let results: Retained<NSDictionary<MPSGraphTensor, MPSGraphTensorData>> =
            NSDictionary::from_slices(&[&*entry.out_t], &[&*out_td]);

        unsafe {
            entry.graph.runWithMTLCommandQueue_feeds_targetOperations_resultsDictionary(
                queue.as_ref(),
                &feeds,
                None,
                &results,
            );
        }
    }

    Ok(Some(MetalStorage::new(out_buf, device, out_el, dt)))
}
