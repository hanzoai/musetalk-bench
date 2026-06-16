use hanzo_ml::Result;
use hanzo_nn::{Conv2d, Conv2dConfig, GroupNorm, LayerNorm, LayerNormConfig, Linear};
use hanzo_quant::ShardedVarBuilder;

pub use hanzo_quant::{Convolution, MatMul};

pub fn layer_norm<C: Into<LayerNormConfig>>(
    size: usize,
    config: C,
    vb: ShardedVarBuilder,
) -> Result<LayerNorm> {
    let config = config.into();
    let weight = vb.get(size, "weight")?;
    if config.affine {
        let bias = vb.get(size, "bias")?;
        Ok(LayerNorm::new(weight, bias, config.eps))
    } else {
        Ok(LayerNorm::new_no_bias(weight, config.eps))
    }
}

pub fn group_norm(
    num_groups: usize,
    num_channels: usize,
    eps: f64,
    vb: ShardedVarBuilder,
) -> Result<GroupNorm> {
    let weight = vb.get(num_channels, "weight")?;
    let bias = vb.get(num_channels, "bias")?;
    GroupNorm::new(weight, bias, num_channels, num_groups, eps)
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    mut cfg: Conv2dConfig,
    vb: ShardedVarBuilder,
) -> Result<Conv2d> {
    #[cfg(feature = "cudnn")]
    {
        let is_3x3_s1 = kernel_size == 3 && cfg.stride == 1;
        cfg.cudnn_fwd_algo = match std::env::var("MUSETALK_CUDNN_ALGO").as_deref() {
            Ok("none") => None,
            Ok("implicit") => Some(hanzo_ml::conv::CudnnFwdAlgo::ImplicitGemm),
            // Winograd only legal/beneficial for 3x3 stride-1; others use the heuristic.
            Ok("winograd") if is_3x3_s1 => Some(hanzo_ml::conv::CudnnFwdAlgo::WinogradNonFused),
            Ok("winograd") => None,
            _ => Some(hanzo_ml::conv::CudnnFwdAlgo::ImplicitPrecompGemm),
        };
    }
    let ws = vb.get(
        (out_channels, in_channels / cfg.groups, kernel_size, kernel_size),
        "weight",
    )?;
    let bs = vb.get(out_channels, "bias")?;
    Ok(Conv2d::new(ws, Some(bs), cfg))
}

pub fn conv2d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    mut cfg: Conv2dConfig,
    vb: ShardedVarBuilder,
) -> Result<Conv2d> {
    #[cfg(feature = "cudnn")]
    {
        let is_3x3_s1 = kernel_size == 3 && cfg.stride == 1;
        cfg.cudnn_fwd_algo = match std::env::var("MUSETALK_CUDNN_ALGO").as_deref() {
            Ok("none") => None,
            Ok("implicit") => Some(hanzo_ml::conv::CudnnFwdAlgo::ImplicitGemm),
            // Winograd only legal/beneficial for 3x3 stride-1; others use the heuristic.
            Ok("winograd") if is_3x3_s1 => Some(hanzo_ml::conv::CudnnFwdAlgo::WinogradNonFused),
            Ok("winograd") => None,
            _ => Some(hanzo_ml::conv::CudnnFwdAlgo::ImplicitPrecompGemm),
        };
    }
    let ws = vb.get(
        (out_channels, in_channels / cfg.groups, kernel_size, kernel_size),
        "weight",
    )?;
    Ok(Conv2d::new(ws, None, cfg))
}

pub fn linear(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Linear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    let bs = vb.get(out_dim, "bias")?;
    Ok(Linear::new(ws, Some(bs)))
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Linear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(ws, None))
}
