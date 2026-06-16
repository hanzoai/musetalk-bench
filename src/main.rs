mod customconv;
mod layers;
mod musetalk;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::io::Write as _;
use std::process::{Command, Stdio};
use std::time::Instant;

use hanzo_ml::{DType, Device, Result, Shape, Tensor};
use hanzo_nn::var_builder::SimpleBackend;
use hanzo_nn::Init;
use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};

use musetalk::{MuseTalk, MuseTalkConfig, RefLatents};

/// mmap a real safetensors checkpoint into a ShardedVarBuilder (no sharding, predicate=true).
fn real_vb(path: &str, dtype: DType, dev: &Device) -> Result<ShardedVarBuilder> {
    let paths = [std::path::PathBuf::from(path)];
    unsafe { ShardedSafeTensors::sharded(&paths, dtype, dev, None, Arc::new(|_| true)) }
}

fn weight_std() -> f32 {
    std::env::var("MUSETALK_WSTD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.02)
}

struct SeededBackend {
    seed: AtomicU64,
}

impl SeededBackend {
    fn new(seed: u64) -> Self {
        Self {
            seed: AtomicU64::new(seed),
        }
    }
    fn next(&self) -> f32 {
        let mut x = self.seed.load(Ordering::Relaxed);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.seed.store(x, Ordering::Relaxed);
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

impl SimpleBackend for SeededBackend {
    fn get(&self, s: Shape, name: &str, _h: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        let n: usize = s.elem_count();
        if name.ends_with("bias") {
            return Tensor::zeros(s, dtype, dev);
        }
        if name.ends_with("weight") && s.rank() == 1 {
            return Tensor::ones(s, dtype, dev);
        }
        let wstd = weight_std();
        let data: Vec<f32> = (0..n).map(|_| self.next() * wstd).collect();
        Tensor::from_vec(data, s, &Device::Cpu)?
            .to_device(dev)?
            .to_dtype(dtype)
    }
    fn get_unchecked(&self, _name: &str, _dtype: DType, _dev: &Device) -> Result<Tensor> {
        hanzo_ml::bail!("SeededBackend requires an explicit shape")
    }
    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

fn seeded_vb(seed: u64, dtype: DType, dev: &Device) -> ShardedVarBuilder {
    ShardedSafeTensors::wrap(Box::new(SeededBackend::new(seed)), dtype, dev.clone())
}

fn seeded_input(seed: u64, shape: &[usize], dev: &Device) -> Result<Tensor> {
    let b = SeededBackend::new(seed);
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| (b.next() + 1.0) * 0.5).collect();
    Tensor::from_vec(data, shape, &Device::Cpu)?.to_device(dev)
}

fn psnr_cosine(a: &Tensor, b: &Tensor) -> Result<(f64, f64)> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len());
    let mut mse = 0f64;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64, y as f64);
        mse += (x - y) * (x - y);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    mse /= a.len() as f64;
    let psnr = if mse <= 1e-12 { 120.0 } else { 10.0 * (1.0 / mse).log10() };
    let cosine = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    Ok((psnr, cosine))
}

struct Stage {
    encode: f64,
    unet: f64,
    decode: f64,
}

/// Build the MuseTalk model. If `taesd` is set, attach a seeded TAESD tiny-decoder for the fast
/// decode path. The VAE/UNet weights are seeded identically regardless, so the encode + UNet
/// numerics are unchanged between framework-only and combined runs.
fn build_model(cfg: &MuseTalkConfig, dtype: DType, dev: &Device, taesd: bool) -> Result<MuseTalk> {
    let model = MuseTalk::new(
        cfg.clone(),
        seeded_vb(0x1234_5678, dtype, dev),
        seeded_vb(0x9abc_def0, dtype, dev),
        dev,
        dtype,
    )?;
    if taesd {
        model
            .with_taesd(seeded_vb(0x7a35_d000, dtype, dev))?
            .with_taesd_encoder(seeded_vb(0x7a35_e000, dtype, dev))
    } else {
        Ok(model)
    }
}

/// Framework-only per-frame path: full VAE-encode of (masked+ref) stack, framework UNet step,
/// full VAE-decode. This is the 4.5fps baseline.
fn time_frame(model: &MuseTalk, face: &Tensor, audio: &Tensor, dev: &Device) -> Result<Stage> {
    dev.synchronize()?;
    let t0 = Instant::now();
    let latents = model.latents_for_unet(face)?;
    dev.synchronize()?;
    let t1 = Instant::now();
    let b = latents.dim(0)?;
    let ts = Tensor::zeros(b, DType::F32, dev)?;
    let pred = model.unet_forward(&latents, &ts, audio)?;
    dev.synchronize()?;
    let t2 = Instant::now();
    let _img = model.decode_latents(&pred)?;
    dev.synchronize()?;
    let t3 = Instant::now();
    Ok(Stage {
        encode: (t1 - t0).as_secs_f64() * 1e3,
        unet: (t2 - t1).as_secs_f64() * 1e3,
        decode: (t3 - t2).as_secs_f64() * 1e3,
    })
}

/// Combined per-frame path with BOTH VAE keystones active:
///   encode = cached static-reference latent (lossless, halves the encode: only the masked frame
///            is VAE-encoded; the precomputed ref latent is concatenated)
///   unet   = framework UNet single step (pinned mempool + fused f16 GroupNorm + fused conv-bias)
///   decode = TAESD tiny-VAE decode (the FLOP cut on the ~45% decode stage)
/// The reference latent is encoded ONCE outside the timed loop, mirroring streaming where the
/// reference face is static across all frames of a stream.
fn time_frame_combined(
    model: &MuseTalk,
    face: &Tensor,
    audio: &Tensor,
    reference: &RefLatents,
    dev: &Device,
) -> Result<Stage> {
    dev.synchronize()?;
    let t0 = Instant::now();
    let latents = model.latents_for_unet_with_ref(face, reference)?;
    dev.synchronize()?;
    let t1 = Instant::now();
    let b = latents.dim(0)?;
    let ts = Tensor::zeros(b, DType::F32, dev)?;
    let pred = model.unet_forward(&latents, &ts, audio)?;
    dev.synchronize()?;
    let t2 = Instant::now();
    let _img = model.decode_latents_fast(&pred)?;
    dev.synchronize()?;
    let t3 = Instant::now();
    Ok(Stage {
        encode: (t1 - t0).as_secs_f64() * 1e3,
        unet: (t2 - t1).as_secs_f64() * 1e3,
        decode: (t3 - t2).as_secs_f64() * 1e3,
    })
}

fn time_frame_fast(
    model: &MuseTalk,
    face: &Tensor,
    audio: &Tensor,
    reference: &RefLatents,
    dev: &Device,
) -> Result<Stage> {
    dev.synchronize()?;
    let t0 = Instant::now();
    let latents = model.latents_for_unet_fast(face, reference)?;
    dev.synchronize()?;
    let t1 = Instant::now();
    let b = latents.dim(0)?;
    let ts = Tensor::zeros(b, DType::F32, dev)?;
    let pred = model.unet_forward(&latents, &ts, audio)?;
    dev.synchronize()?;
    let t2 = Instant::now();
    let _img = model.decode_latents_fast(&pred)?;
    dev.synchronize()?;
    let t3 = Instant::now();
    Ok(Stage {
        encode: (t1 - t0).as_secs_f64() * 1e3,
        unet: (t2 - t1).as_secs_f64() * 1e3,
        decode: (t3 - t2).as_secs_f64() * 1e3,
    })
}

fn pick_dtype() -> DType {
    match std::env::var("MUSETALK_DTYPE").as_deref() {
        Ok("f16") => DType::F16,
        Ok("bf16") => DType::BF16,
        _ => DType::F32,
    }
}

/// Backend selection via MUSETALK_DEV: "cuda" | "metal" | (anything else / unset -> CPU).
/// `Device::new_metal`/`new_cuda` are only compiled in when the matching backend feature is on,
/// so the arms are cfg-gated; an unavailable backend bails with a clear message.
fn pick_device() -> Result<Device> {
    match std::env::var("MUSETALK_DEV").as_deref() {
        Ok("cuda") => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0)
            }
            #[cfg(not(feature = "cuda"))]
            {
                hanzo_ml::bail!("MUSETALK_DEV=cuda but binary not built with --features cuda")
            }
        }
        Ok("metal") => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0)
            }
            #[cfg(not(feature = "metal"))]
            {
                hanzo_ml::bail!("MUSETALK_DEV=metal but binary not built with --features metal")
            }
        }
        _ => Ok(Device::Cpu),
    }
}

struct Agg {
    e: f64,
    u: f64,
    d: f64,
    total_min: f64,
}

fn run_loop<F>(iters: usize, bsz: usize, mut f: F) -> Result<Agg>
where
    F: FnMut() -> Result<Stage>,
{
    // warmup
    for _ in 0..3 {
        let _ = f()?;
    }
    let (mut e, mut u, mut d) = (0f64, 0f64, 0f64);
    let mut total_min = f64::MAX;
    for _ in 0..iters {
        let s = f()?;
        e += s.encode;
        u += s.unet;
        d += s.decode;
        total_min = total_min.min(s.encode + s.unet + s.decode);
    }
    let n = iters as f64;
    Ok(Agg {
        e: e / n / bsz as f64,
        u: u / n / bsz as f64,
        d: d / n / bsz as f64,
        total_min: total_min / bsz as f64,
    })
}

fn print_agg(label: &str, enc_label: &str, dec_label: &str, a: &Agg) {
    let total = a.e + a.u + a.d;
    println!("\n-- {label} --");
    println!("{:<16}{:8.3} ms", enc_label, a.e);
    println!("{:<16}{:8.3} ms", "unet-1step:", a.u);
    println!("{:<16}{:8.3} ms", dec_label, a.d);
    println!("{:<16}{:8.3} ms  (best {:.3} ms)", "total/frame:", total, a.total_min);
    println!("{:<16}{:8.2}", "fps(mean):", 1000.0 / total);
    println!("{:<16}{:8.2}", "fps(best):", 1000.0 / a.total_min);
}

fn run_bench(dev: &Device) -> Result<()> {
    let dtype = pick_dtype();
    let iters: usize = std::env::var("MUSETALK_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let cfg = MuseTalkConfig::default();
    let bsz: usize = std::env::var("MUSETALK_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let sz = cfg.resized_img;
    let face = seeded_input(0x55, &[bsz, 3, sz, sz], dev)?.to_dtype(dtype)?;
    let audio = seeded_input(0xAA, &[bsz, 50, cfg.unet.cross_attention_dim], dev)?.to_dtype(dtype)?;

    println!(
        "\n==== MuseTalk COMBINED bench  dev={:?} dtype={:?} iters={} batch={} (per-frame) ====",
        dev.location(),
        dtype,
        iters,
        bsz
    );

    // --- Framework-only baseline: full VAE-encode + framework UNet + full VAE-decode ---
    let fw = build_model(&cfg, dtype, dev, false)?;
    let base = run_loop(iters, bsz, || time_frame(&fw, &face, &audio, dev))?;
    print_agg(
        "framework-only (full VAE enc + UNet + full VAE dec)",
        "vae-encode(x2):",
        "vae-decode:",
        &base,
    );

    // --- Combined: cached-ref encode + framework UNet + TAESD decode ---
    let combo = build_model(&cfg, dtype, dev, true)?;
    // Reference latent of the static face, encoded ONCE (streaming reuse, not in the timed loop).
    let reference = combo.encode_reference(&face)?;
    let comb = run_loop(iters, bsz, || {
        time_frame_combined(&combo, &face, &audio, &reference, dev)
    })?;
    print_agg(
        "COMBINED (cached-ref enc + UNet + TAESD dec)",
        "vae-encode(x1):",
        "taesd-decode:",
        &comb,
    );

    // --- FULL-FAST: TAESD tiny-encoder + framework UNet + TAESD tiny-decoder ---
    let fast = run_loop(iters, bsz, || {
        time_frame_fast(&combo, &face, &audio, &reference, dev)
    })?;
    print_agg(
        "FULL-FAST (TAESD enc + UNet + TAESD dec)",
        "taesd-encode:",
        "taesd-decode:",
        &fast,
    );

    // --- Summary delta ---
    let base_total = base.e + base.u + base.d;
    let comb_total = comb.e + comb.u + comb.d;
    println!("\n==== SUMMARY (per-frame, mean) ====");
    println!(
        "encode:  {:8.3} -> {:8.3} ms   ({:.2}x)",
        base.e,
        comb.e,
        base.e / comb.e.max(1e-9)
    );
    println!("unet:    {:8.3} -> {:8.3} ms   (framework, unchanged)", base.u, comb.u);
    println!(
        "decode:  {:8.3} -> {:8.3} ms   ({:.2}x)",
        base.d,
        comb.d,
        base.d / comb.d.max(1e-9)
    );
    println!(
        "total:   {:8.3} -> {:8.3} ms   ({:.2}x)",
        base_total,
        comb_total,
        base_total / comb_total.max(1e-9)
    );
    println!(
        "fps:     {:8.2} -> {:8.2}        (target 30.0; gap {:.1} ms / {:.2}x to go)",
        1000.0 / base_total,
        1000.0 / comb_total,
        comb_total - 1000.0 / 30.0,
        comb_total / (1000.0 / 30.0)
    );
    // dominant remaining stage in the combined path
    let (dom, dom_ms) = [("encode", comb.e), ("unet", comb.u), ("decode", comb.d)]
        .into_iter()
        .fold(("", 0f64), |acc, (n, v)| if v > acc.1 { (n, v) } else { acc });
    println!(
        "dominant remaining stage: {} ({:.3} ms, {:.0}% of frame)",
        dom,
        dom_ms,
        100.0 * dom_ms / comb_total
    );
    Ok(())
}

fn run_verify() -> Result<()> {
    let cpu = Device::Cpu;
    let cfg = MuseTalkConfig::default();
    let sz = cfg.resized_img;

    let model_cpu = MuseTalk::new(
        cfg.clone(),
        seeded_vb(0x1234_5678, DType::F32, &cpu),
        seeded_vb(0x9abc_def0, DType::F32, &cpu),
        &cpu,
        DType::F32,
    )?;
    let face_cpu = seeded_input(0x55, &[1, 3, sz, sz], &cpu)?;
    let audio_cpu = seeded_input(0xAA, &[1, 50, cfg.unet.cross_attention_dim], &cpu)?;
    let ref_img = model_cpu.forward(&face_cpu, &audio_cpu)?;

    let gpu = pick_device()?; // honors MUSETALK_DEV (cuda|metal); CPU-vs-CPU if unset
    let dtype = pick_dtype();
    let model_gpu = MuseTalk::new(
        cfg.clone(),
        seeded_vb(0x1234_5678, dtype, &gpu),
        seeded_vb(0x9abc_def0, dtype, &gpu),
        &gpu,
        dtype,
    )?;
    let face_gpu = seeded_input(0x55, &[1, 3, sz, sz], &gpu)?.to_dtype(dtype)?;
    let audio_gpu = seeded_input(0xAA, &[1, 50, cfg.unet.cross_attention_dim], &gpu)?.to_dtype(dtype)?;
    let gpu_img = model_gpu.forward(&face_gpu, &audio_gpu)?;

    let lat_cpu = model_cpu.latents_for_unet(&face_cpu)?;
    let lat_gpu = model_gpu.latents_for_unet(&face_gpu)?;
    let (lp, lc) = psnr_cosine(&lat_cpu, &lat_gpu.to_device(&cpu)?)?;

    let ts_cpu = Tensor::zeros(1, DType::F32, &cpu)?;
    let ts_gpu = Tensor::zeros(1, DType::F32, &gpu)?;
    let lat_gpu_id = lat_cpu.to_device(&gpu)?.to_dtype(dtype)?;
    let pred_cpu = model_cpu.unet_forward(&lat_cpu, &ts_cpu, &audio_cpu)?;
    let pred_gpu = model_gpu.unet_forward(&lat_gpu_id, &ts_gpu, &audio_gpu)?;
    let (up, uc) = psnr_cosine(&pred_cpu, &pred_gpu.to_device(&cpu)?)?;

    let pred_gpu_id = pred_cpu.to_device(&gpu)?.to_dtype(dtype)?;
    let dec_cpu = model_cpu.decode_latents(&pred_cpu)?;
    let dec_gpu = model_gpu.decode_latents(&pred_gpu_id)?;
    let (dp, dc) = psnr_cosine(&dec_cpu, &dec_gpu.to_device(&cpu)?)?;

    let (psnr, cosine) = psnr_cosine(&ref_img, &gpu_img.to_device(&cpu)?)?;
    println!("\n==== MuseTalk correctness  gpu_dtype={:?} vs cpu-f32 ====", dtype);
    println!("stage latents_for_unet: PSNR {:7.3} dB  cosine {:.6}", lp, lc);
    println!("stage unet (on same lat): PSNR {:7.3} dB  cosine {:.6}", up, uc);
    println!("stage decode (on same pred): PSNR {:7.3} dB  cosine {:.6}", dp, dc);
    println!("full forward:           PSNR {:7.3} dB  cosine {:.6}", psnr, cosine);
    Ok(())
}

// GPU-vs-GPU same-dtype fidelity gate: save the per-stage GPU outputs as the pre-lever
// reference (MUSETALK_REF_SAVE=1), then after a lever re-run and compare. Any divergence is
// purely from the kernel change, not from the f16-vs-f32 cross-device gap that `verify` mixes in.
fn run_selfcheck() -> Result<()> {
    let dir = std::env::var("MUSETALK_REF_DIR").unwrap_or_else(|_| "/tmp/musetalk_ref".to_string());
    let save = std::env::var("MUSETALK_REF_SAVE").is_ok();
    let gpu = pick_device()?;
    let dtype = pick_dtype();
    let cfg = MuseTalkConfig::default();
    let sz = cfg.resized_img;
    let model = MuseTalk::new(
        cfg.clone(),
        seeded_vb(0x1234_5678, dtype, &gpu),
        seeded_vb(0x9abc_def0, dtype, &gpu),
        &gpu,
        dtype,
    )?;
    let face = seeded_input(0x55, &[1, 3, sz, sz], &gpu)?;
    let audio = seeded_input(0xAA, &[1, 50, cfg.unet.cross_attention_dim], &gpu)?.to_dtype(dtype)?;

    let lat = model.latents_for_unet(&face)?;
    let ts = Tensor::zeros(1, DType::F32, &gpu)?;
    let pred = model.unet_forward(&lat, &ts, &audio)?;
    let dec = model.decode_latents(&pred)?;
    let stages = [("encode", &lat), ("unet", &pred), ("decode", &dec)];

    if save {
        std::fs::create_dir_all(&dir).ok();
        for (name, t) in stages.iter() {
            t.to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .write_npy(format!("{dir}/{name}.npy"))?;
        }
        println!("saved GPU reference (dtype={dtype:?}) to {dir}");
    } else {
        println!("\n==== MuseTalk self-check  gpu_dtype={dtype:?} vs saved GPU ref ====");
        let mut worst = f64::MAX;
        for (name, t) in stages.iter() {
            let r = Tensor::read_npy(format!("{dir}/{name}.npy"))?.to_device(&Device::Cpu)?;
            let (p, c) = psnr_cosine(&r, &t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?)?;
            worst = worst.min(p);
            println!("stage {name:7}: PSNR {p:8.3} dB  cosine {c:.6}");
        }
        println!("worst-stage PSNR: {worst:.3} dB  (>=60 dB = numerically faithful)");
    }
    Ok(())
}

/// Real-weight numerical verification against PyTorch MuseTalk.
///
/// Loads the converted real `unet.safetensors` + `vae.safetensors`, then for each stage feeds
/// the SAME PyTorch-dumped input tensor (from refdump/*.npy) and compares our output to the
/// PyTorch output for that stage. This isolates each stage so an upstream f32 rounding diff
/// can't cascade. Stages: VAE-encode (mode), UNet pred, VAE-decode. Also runs end-to-end.
fn run_realverify() -> Result<()> {
    let refdir = std::env::var("MUSETALK_REFDIR")
        .unwrap_or_else(|_| "/home/z/work/zen-dub-run/refdump".to_string());
    let wdir = std::env::var("MUSETALK_WDIR")
        .unwrap_or_else(|_| "/home/z/work/zen-dub-run/rustweights".to_string());
    let dev = pick_device()?;
    let dtype = pick_dtype();
    let load = |n: &str| -> Result<Tensor> {
        Tensor::read_npy(format!("{refdir}/{n}.npy"))?.to_device(&dev)
    };

    let cfg = MuseTalkConfig::default();
    let model = MuseTalk::new(
        cfg.clone(),
        real_vb(&format!("{wdir}/vae.safetensors"), dtype, &dev)?,
        real_vb(&format!("{wdir}/unet.safetensors"), dtype, &dev)?,
        &dev,
        dtype,
    )?;
    println!(
        "\n==== MuseTalk REAL-WEIGHT verify  dev={:?} dtype={:?} vs PyTorch(f32) ====",
        dev.location(),
        dtype
    );

    // --- Stage 1: VAE-encode MODE on the exact ref image PyTorch used ---
    let face_crop = load("face_crop")?.to_dtype(dtype)?; // already normalized [1,3,256,256]
    let enc = if std::env::var("MUSETALK_DEBUG").is_ok() {
        model.vae_encode_mode_debug(&face_crop, &refdir)?
    } else {
        model.vae_encode_mode(&face_crop)?
    };
    let enc_ref = load("enc_mode")?;
    let (ep, ec) = psnr_cosine(&enc_ref, &enc.to_device(&Device::Cpu)?)?;
    println!("VAE-encode(mode): PSNR {ep:8.3} dB  cosine {ec:.6}");

    // also verify the masked-half encode + full 8ch unet input assembly
    let masked_img = load("masked_img")?.to_dtype(dtype)?;
    let masked_lat = model.vae_encode_mode(&masked_img)?;
    let unet_in = Tensor::cat(&[&masked_lat, &enc], 1)?;
    let unet_in_ref = load("unet_in")?;
    let (uip, uic) = psnr_cosine(&unet_in_ref, &unet_in.to_device(&Device::Cpu)?)?;
    println!("UNet-input(8ch): PSNR {uip:8.3} dB  cosine {uic:.6}");

    // --- Stage 2: UNet pred on the EXACT PyTorch unet_in + post-PE audio_feat ---
    let unet_in_id = unet_in_ref.to_device(&dev)?.to_dtype(dtype)?;
    let audio_feat = load("audio_feat")?.to_dtype(dtype)?;
    let ts = Tensor::zeros(1, DType::F32, &dev)?;
    let pred = model.unet_forward(&unet_in_id, &ts, &audio_feat)?;
    let pred_ref = load("unet_pred")?;
    let (pp, pc) = psnr_cosine(&pred_ref, &pred.to_device(&Device::Cpu)?)?;
    println!("UNet-pred:        PSNR {pp:8.3} dB  cosine {pc:.6}");

    // --- Stage 3: VAE-decode on the EXACT PyTorch pred (compare raw decoder out, pre-denorm) ---
    let pred_id = pred_ref.to_device(&dev)?.to_dtype(dtype)?;
    let dec = model.vae_decode_raw(&pred_id)?;
    let dec_ref = load("vae_dec")?;
    let (dp, dc) = psnr_cosine(&dec_ref, &dec.to_device(&Device::Cpu)?)?;
    println!("VAE-decode:       PSNR {dp:8.3} dB  cosine {dc:.6}");

    // --- End-to-end: our encode -> our unet -> our decode, vs PyTorch final decode ---
    let e2e_pred = model.unet_forward(&unet_in, &ts, &audio_feat)?;
    let e2e_dec = model.vae_decode_raw(&e2e_pred)?;
    let (e2ep, e2ec) = psnr_cosine(&dec_ref, &e2e_dec.to_device(&Device::Cpu)?)?;
    println!("end-to-end:       PSNR {e2ep:8.3} dB  cosine {e2ec:.6}");
    println!("(>=40 dB / cosine>0.999 vs PyTorch f32 = numerically matching)");
    Ok(())
}

/// Native-MODEL dub inference: run OUR Rust MuseTalk (VAE-encode + UNet + VAE-decode) over a
/// directory of per-frame inputs produced by the Python CV pre-stage, writing decoded mouths.
///   in:  {indir}/face_{i:06}.npy   [1,3,256,256] normalized full-face crop (mean=std=0.5)
///        {indir}/audio_{i:06}.npy  [1,50,384]    post-PositionalEncoding whisper feature
///   out: {outdir}/mouth_{i:06}.npy [3,256,256]   RGB image in [0,1] (Python -> BGR uint8 + blend)
fn run_pipebench() -> Result<()> {
    let wdir = std::env::var("MUSETALK_WDIR")
        .unwrap_or_else(|_| "/home/z/work/zen-dub-run/rustweights".to_string());
    let nframes: usize = std::env::var("MUSETALK_NFRAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(500);
    let bsz: usize = std::env::var("MUSETALK_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let pipe = std::env::var("MUSETALK_PIPE").is_ok();
    let outdir = std::env::var("MUSETALK_DUBOUT").unwrap_or_else(|_| "/tmp/native_dub".to_string());
    let outmp4 = std::env::var("MUSETALK_OUT").unwrap_or_else(|_| "/tmp/native_dub.mp4".to_string());
    let dev = pick_device()?;
    let dtype = pick_dtype();
    let cfg = MuseTalkConfig::default();
    let sz = cfg.resized_img;
    let fast = std::env::var("MUSETALK_FAST").is_ok();
    let mut model = MuseTalk::new(
        cfg.clone(),
        real_vb(&format!("{wdir}/vae.safetensors"), dtype, &dev)?,
        real_vb(&format!("{wdir}/unet.safetensors"), dtype, &dev)?,
        &dev,
        dtype,
    )?;
    if fast {
        let taesd_path = std::env::var("TAESD_PATH").unwrap_or_else(|_| {
            "/home/z/.cache/huggingface/hub/models--madebyollin--taesd/snapshots/614f76814bbe30edbe2e627ace1c2234c81a2c0e/diffusion_pytorch_model.safetensors".to_string()
        });
        let tvb = real_vb(&taesd_path, dtype, &dev)?;
        let tvb2 = real_vb(&taesd_path, dtype, &dev)?;
        model = model
            .with_taesd_encoder(tvb.pp("encoder").pp("layers"))?
            .with_taesd(tvb2.pp("decoder").pp("layers"))?;
    }
    // Fixed (cached-avatar) seeded face+audio looped; isolates the per-frame I/O path. RGB in [0,1].
    let face = seeded_input(0x55, &[bsz, 3, sz, sz], &dev)?.to_dtype(dtype)?;
    let audio = seeded_input(0xAA, &[bsz, 50, cfg.unet.cross_attention_dim], &dev)?.to_dtype(dtype)?;
    // Cache the static reference latent (KL, one-time) for the fast streaming encode.
    let reference = model.encode_reference(&face.narrow(0, 0, 1)?)?;
    let render = |f: &Tensor, a: &Tensor| -> Result<Tensor> {
        if fast {
            let lat = model.latents_for_unet_fast(f, &reference)?;
            let ts = Tensor::zeros(f.dim(0)?, DType::F32, &dev)?;
            let pred = model.unet_forward(&lat, &ts, a)?;
            model.decode_latents_fast(&pred)
        } else {
            model.forward_batched(f, a)
        }
    };
    println!(
        "==== MuseTalk NATIVE pipebench  dev={:?} dtype={:?} frames={} batch={} sink={} ====",
        dev.location(), dtype, nframes, bsz, if pipe { "ffmpeg-pipe" } else { "npy-dump" }
    );
    let mut child = if pipe {
        Some(
            Command::new("ffmpeg")
                .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", &format!("{sz}x{sz}"), "-r", "25", "-i", "pipe:0",
                       "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", &outmp4])
                .stdin(Stdio::piped())
                .spawn()?,
        )
    } else {
        std::fs::create_dir_all(&outdir).ok();
        None
    };
    // warmup (model + ffmpeg startup excluded from the timed region)
    let _ = render(&face, &audio)?;
    dev.synchronize()?;
    let t0 = Instant::now();
    let mut done = 0usize;
    while done < nframes {
        let n = bsz.min(nframes - done);
        let f = if n == bsz { face.clone() } else { face.narrow(0, 0, n)? };
        let a = if n == bsz { audio.clone() } else { audio.narrow(0, 0, n)? };
        let img = render(&f, &a)?; // [n,3,sz,sz] RGB [0,1]
        // -> u8 RGB24, NHWC, on CPU
        let u8s = (img.clamp(0f32, 1f32)? * 255.0)?
            .round()?
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .to_dtype(DType::U8)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let frame_bytes = 3 * sz * sz;
        if let Some(ch) = child.as_mut() {
            ch.stdin.as_mut().unwrap().write_all(&u8s)?;
        } else {
            for k in 0..n {
                let off = k * frame_bytes;
                Tensor::from_slice(&u8s[off..off + frame_bytes], (sz, sz, 3), &Device::Cpu)?
                    .write_npy(format!("{outdir}/frame_{:06}.npy", done + k))?;
            }
        }
        done += n;
    }
    dev.synchronize()?;
    if let Some(mut ch) = child {
        drop(ch.stdin.take());
        ch.wait()?;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "[pipebench] {nframes} frames in {dt:.2}s -> {:.2} fps (e2e incl. I/O, sink={})",
        nframes as f64 / dt,
        if pipe { "ffmpeg-pipe" } else { "npy-dump" }
    );
    Ok(())
}

fn run_dub() -> Result<()> {
    let indir = std::env::var("MUSETALK_DUBIN").expect("set MUSETALK_DUBIN");
    let outdir = std::env::var("MUSETALK_DUBOUT").expect("set MUSETALK_DUBOUT");
    let wdir = std::env::var("MUSETALK_WDIR")
        .unwrap_or_else(|_| "/home/z/work/zen-dub-run/rustweights".to_string());
    let nframes: usize = std::env::var("MUSETALK_NFRAMES")
        .expect("set MUSETALK_NFRAMES")
        .parse()
        .unwrap();
    let bsz: usize = std::env::var("MUSETALK_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let dev = pick_device()?;
    let dtype = pick_dtype();
    std::fs::create_dir_all(&outdir).ok();
    let cfg = MuseTalkConfig::default();
    let model = MuseTalk::new(
        cfg,
        real_vb(&format!("{wdir}/vae.safetensors"), dtype, &dev)?,
        real_vb(&format!("{wdir}/unet.safetensors"), dtype, &dev)?,
        &dev,
        dtype,
    )?;
    println!(
        "==== MuseTalk NATIVE dub  dev={:?} dtype={:?} frames={} batch={} ====",
        dev.location(),
        dtype,
        nframes,
        bsz
    );
    let t0 = Instant::now();
    let mut done = 0usize;
    while done < nframes {
        let n = bsz.min(nframes - done);
        let mut faces = Vec::with_capacity(n);
        let mut auds = Vec::with_capacity(n);
        for k in 0..n {
            let i = done + k;
            faces.push(Tensor::read_npy(format!("{indir}/face_{i:06}.npy"))?.to_device(&dev)?);
            auds.push(Tensor::read_npy(format!("{indir}/audio_{i:06}.npy"))?.to_device(&dev)?);
        }
        let faces = Tensor::cat(&faces, 0)?.to_dtype(dtype)?;
        let auds = Tensor::cat(&auds, 0)?.to_dtype(dtype)?;
        let img = model.forward_batched(&faces, &auds)?; // [n,3,256,256] RGB [0,1]
        let img = img.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        for k in 0..n {
            let i = done + k;
            img.narrow(0, k, 1)?
                .squeeze(0)?
                .write_npy(format!("{outdir}/mouth_{i:06}.npy"))?;
        }
        done += n;
        if done % 32 == 0 || done == nframes {
            println!("  {done}/{nframes}");
        }
    }
    dev.synchronize()?;
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "native model inference done: {nframes} frames in {dt:.2}s ({:.2} fps)",
        nframes as f64 / dt
    );
    Ok(())
}

fn run_taesd_verify() -> Result<()> {
    use musetalk::taesd::TaesdEncoder;
    let refdir = std::env::var("MUSETALK_REFDIR")
        .unwrap_or_else(|_| "/home/z/work/zen-dub-run/refdump".to_string());
    let taesd_path = std::env::var("TAESD_PATH").unwrap_or_else(|_| {
        "/home/z/.cache/huggingface/hub/models--madebyollin--taesd/snapshots/614f76814bbe30edbe2e627ace1c2234c81a2c0e/diffusion_pytorch_model.safetensors".to_string()
    });
    let dev = pick_device()?;
    let dtype = pick_dtype();
    let load = |n: &str| -> Result<Tensor> {
        Tensor::read_npy(format!("{refdir}/{n}.npy"))?.to_device(&dev)
    };
    // Real TAESD encoder (diffusers AutoencoderTiny `encoder.layers` Sequential). sf=1.0: TAESD
    // latents already live in the KL scaled (0.18215) space, and cosine is scale-invariant anyway.
    let taesd_vb = real_vb(&taesd_path, dtype, &dev)?;
    let enc = TaesdEncoder::new(3, 4, 1.0, taesd_vb.pp("encoder").pp("layers"))?;

    let masked_img = load("masked_img")?.to_dtype(dtype)?;
    let masked_lat = load("masked_lat")?;
    let face_crop = load("face_crop")?.to_dtype(dtype)?;
    let enc_mode = load("enc_mode")?;

    println!(
        "\n==== TAESD-encoder correctness (REAL weights) dev={:?} dtype={:?} ====",
        dev.location(),
        dtype
    );
    println!("cosine vs PyTorch KL latent (scale-invariant). >0.90 = faithful drop-in.");
    for (name, img, refl) in [
        ("masked", &masked_img, &masked_lat),
        ("full-face", &face_crop, &enc_mode),
    ] {
        let la = enc.encode(img)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let (pa, ca) = psnr_cosine(refl, &la)?;
        let img01 = ((img.to_dtype(DType::F32)? + 1.0)? * 0.5)?.to_dtype(dtype)?;
        let lb = enc.encode(&img01)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let (pb, cb) = psnr_cosine(refl, &lb)?;
        // best-fit scale k = <ref,taesd>/<taesd,taesd>; scaled psnr shows residual after pure-scale
        let r = refl.flatten_all()?.to_vec1::<f32>()?;
        let t = lb.flatten_all()?.to_vec1::<f32>()?;
        let dot: f64 = r.iter().zip(&t).map(|(a,b)| *a as f64 * *b as f64).sum();
        let tt: f64 = t.iter().map(|b| *b as f64 * *b as f64).sum();
        let k = dot / tt.max(1e-12);
        let lk = (lb.to_dtype(DType::F32)? * k)?;
        let (pk, _) = psnr_cosine(refl, &lk)?;
        println!("{name:9}: [-1,1]-in cos {ca:.4}   [0,1]-in cos {cb:.4} (psnr {pb:.2})  scale-fit k={k:.4} -> psnr {pk:.2}");
    }
    // ---- decoder drop-in check: TAESD.decode(unet_pred) vs KL decode (denormalized [0,1]) ----
    {
        use musetalk::taesd::TaesdDecoder;
        let dec = TaesdDecoder::new(4, 3, 1.0, taesd_vb.pp("decoder").pp("layers"))?;
        let pred = load("unet_pred")?.to_dtype(dtype)?;
        let kl_img = ((load("vae_dec")?.to_dtype(DType::F32)? + 1.0)? * 0.5)?
            .clamp(0f32, 1f32)?
            .to_device(&Device::Cpu)?; // KL raw out (~[-1,1]) -> [0,1]
        let td = dec.decode(&pred)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?; // already [0,1]
        let (pd, cd) = psnr_cosine(&kl_img, &td)?;
        println!("decoder  : TAESD.decode vs KL image  cosine {cd:.4} (psnr {pd:.2})");
    }
    Ok(())
}


/// Microbench: time the custom TAESD conv kernel vs the standard (cuDNN/im2col) path on the exact
/// TAESD encoder/decoder conv shapes, and verify numeric parity (cosine). Forces f16 cuda.
/// Each row: (C_in, C_out, H=W, stride). 3x3, pad 1.
/// CUDA-only: it benchmarks the custom SIMT conv kernel (`customconv`), which has no Metal impl.
#[cfg(feature = "cuda")]
fn run_convbench() -> Result<()> {
    use hanzo_nn::{Conv2d, Conv2dConfig};
    let dev = Device::new_cuda(0)?;
    let dtype = DType::F16;
    let bsz: usize = std::env::var("MUSETALK_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let iters: usize = std::env::var("MUSETALK_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(50);

    // (C_in, C_out, HW, stride, label) covering the TAESD enc+dec convs.
    let shapes: &[(usize, usize, usize, usize, &str)] = &[
        (3, 64, 256, 1, "conv_in 3->64 @256"),
        (64, 64, 256, 1, "block 64->64 @256"),
        (64, 64, 256, 2, "down 64->64 @256->128"),
        (64, 64, 128, 1, "block 64->64 @128"),
        (64, 64, 128, 2, "down 64->64 @128->64"),
        (64, 64, 64, 1, "block 64->64 @64"),
        (64, 64, 64, 2, "down 64->64 @64->32"),
        (64, 64, 32, 1, "block 64->64 @32"),
        (64, 4, 32, 1, "conv_out 64->4 @32"),
    ];

    println!("\n==== TAESD conv microbench  dev=cuda dtype=F16 batch={bsz} iters={iters} ====");
    println!("(3x3 pad1; times in us per conv; std = cuDNN when HW^2>=2048 else native im2col)");
    println!("{:<28} {:>10} {:>10} {:>8} {:>9}", "shape", "std us", "custom us", "speedup", "cosine");

    let mut tot_std = 0f64;
    let mut tot_custom = 0f64;
    let mut worst_cos = 2f64;
    for &(c_in, c_out, hw, stride, label) in shapes {
        let cfg = Conv2dConfig { padding: 1, stride, dilation: 1, groups: 1, cudnn_fwd_algo: None };
        let x = seeded_input(0x123, &[bsz, c_in, hw, hw], &dev)?.to_dtype(dtype)?;
        let w = seeded_input(0x456, &[c_out, c_in, 3, 3], &dev)?.to_dtype(dtype)?;
        let b = seeded_input(0x789, &[c_out], &dev)?.to_dtype(dtype)?;
        let layer = Conv2d::new(w.clone(), Some(b.clone()), cfg);

        // Reference: standard path with explicit cuDNN algo so HW>=2048 convs use implicit-GEMM.
        let mut cfg_cudnn = cfg;
        #[cfg(feature = "cudnn")]
        {
            cfg_cudnn.cudnn_fwd_algo = Some(hanzo_ml::conv::CudnnFwdAlgo::ImplicitPrecompGemm);
        }
        let layer_std = Conv2d::new(w.clone(), Some(b.clone()), cfg_cudnn);

        let std_out = hanzo_quant::Convolution.forward_2d(&layer_std, &x)?;
        let cust_out = customconv::forward(&layer, &x, false)?;
        let (_p, cos) = psnr_cosine(&std_out, &cust_out)?;
        worst_cos = worst_cos.min(cos);

        // warmup
        for _ in 0..5 {
            let _ = hanzo_quant::Convolution.forward_2d(&layer_std, &x)?;
            let _ = customconv::forward(&layer, &x, false)?;
        }
        dev.synchronize()?;
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = hanzo_quant::Convolution.forward_2d(&layer_std, &x)?;
        }
        dev.synchronize()?;
        let std_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let t1 = Instant::now();
        for _ in 0..iters {
            let _ = customconv::forward(&layer, &x, false)?;
        }
        dev.synchronize()?;
        let cust_us = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;

        tot_std += std_us;
        tot_custom += cust_us;
        println!(
            "{:<28} {:>10.1} {:>10.1} {:>7.2}x {:>9.5}",
            label, std_us, cust_us, std_us / cust_us.max(1e-9), cos
        );
    }
    println!("{:<28} {:>10.1} {:>10.1} {:>7.2}x", "TOTAL (one of each)", tot_std, tot_custom, tot_std / tot_custom.max(1e-9));
    println!("worst-shape cosine: {worst_cos:.6}  (target > 0.999)");
    Ok(())
}

fn main() -> Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "bench".to_string());
    let dev = pick_device()?;
    match mode.as_str() {
        "verify" => run_verify()?,
        "selfcheck" => run_selfcheck()?,
        "realverify" => run_realverify()?,
        "taesdverify" => run_taesd_verify()?,
        "dub" => run_dub()?,
        "pipebench" => run_pipebench()?,
        #[cfg(feature = "cuda")]
        "convbench" => run_convbench()?,
        #[cfg(not(feature = "cuda"))]
        "convbench" => hanzo_ml::bail!("convbench is cuda-only (custom SIMT conv kernel)"),
        _ => run_bench(&dev)?,
    }
    Ok(())
}
