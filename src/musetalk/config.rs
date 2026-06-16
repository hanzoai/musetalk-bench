use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct UNetConfig {
    pub sample_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub layers_per_block: usize,
    pub block_out_channels: Vec<usize>,
    pub down_block_types: Vec<String>,
    pub up_block_types: Vec<String>,
    pub cross_attention_dim: usize,
    pub attention_head_dim: usize,
    pub norm_num_groups: usize,
    pub norm_eps: f64,
    pub flip_sin_to_cos: bool,
    pub freq_shift: f64,
}

impl Default for UNetConfig {
    fn default() -> Self {
        Self {
            sample_size: 32,
            in_channels: 8,
            out_channels: 4,
            layers_per_block: 2,
            block_out_channels: vec![320, 640, 1280, 1280],
            down_block_types: vec![
                "CrossAttnDownBlock2D".to_string(),
                "CrossAttnDownBlock2D".to_string(),
                "CrossAttnDownBlock2D".to_string(),
                "DownBlock2D".to_string(),
            ],
            up_block_types: vec![
                "UpBlock2D".to_string(),
                "CrossAttnUpBlock2D".to_string(),
                "CrossAttnUpBlock2D".to_string(),
                "CrossAttnUpBlock2D".to_string(),
            ],
            cross_attention_dim: 384,
            attention_head_dim: 8,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            flip_sin_to_cos: true,
            freq_shift: 0.0,
        }
    }
}

impl UNetConfig {
    pub fn has_cross_attn(&self, block_type: &str) -> bool {
        block_type.starts_with("CrossAttn")
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    pub scaling_factor: f64,
    pub sample_size: usize,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
            scaling_factor: 0.18215,
            sample_size: 256,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MuseTalkConfig {
    pub unet: UNetConfig,
    pub vae: VaeConfig,
    pub resized_img: usize,
}

impl Default for MuseTalkConfig {
    fn default() -> Self {
        Self {
            unet: UNetConfig::default(),
            vae: VaeConfig::default(),
            resized_img: 256,
        }
    }
}
