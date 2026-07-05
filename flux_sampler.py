"""Flux2 sampling core - pipeline completo CLIP encode -> guidance -> sample -> VAE decode."""

import gc
import torch


def build_flux_conditioning(clip_obj, text: str, guidance: float, ref_latents: list) -> list:
    """Build conditioning con text + guidance + ref_latents keys para Flux."""
    tokenized = clip_obj.tokenize(text)
    pos_cond = clip_obj.encode(tokenized)

    result = []
    for c in pos_cond:
        if isinstance(c, list) and len(c) >= 2:
            part1 = c[0]
            part2 = dict(c[1])  # copy mutable dict
            part2["guidance"] = torch.tensor([guidance], dtype=torch.float32)
            part2["ref_latents"] = ref_latents.copy()
            result.append([part1, part2])
        else:
            result.append(c)
    return result


def build_negative_conditioning(pos_cond: list) -> list:
    """Zero-out negative conditioning para CFG > 1."""
    result = []
    for c in pos_cond:
        if isinstance(c, list) and len(c) >= 2:
            part1 = c[0]
            part2 = {}
            for k, v in c[1].items():
                if k in ("guidance", "ref_latents"):
                    pass 
                else:
                    if hasattr(v, "zeros_like"):
                        part2[k] = v.zeros_like()
            result.append([part1, part2])
        else:
            result.append(c)
    return result


def flux_sample(
    model_patcher, clip_obj, vae_obj,
    text: str, guidance: float, cfg_scale: float,
    ref_samples: torch.Tensor, width: int, height: int,
    steps: int, seed: int,
) -> torch.Tensor:
    """Flux2 sampling con Reference Latent - una sola escena.

    Replica exacto del workflow visual:
      CLIPTextEncode(text) + FluxGuidance(guidance) 
      ConditioningZeroOut(negative) + CFGGuider(cfg)
      EmptyFLUX2Latent -> Noise(seed)
      SamplerCustomAdvanced(euler, steps)
      VAEDecode

    Returns [1, H, W, C] float32 en rango [0, 1].
    """
    
    from comfy.samplers import CFGGuider, KSAMPLER
    import comfy.k_diffusion.sampling as kdsamp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Build conditioning ──────────────
    pos_cond = build_flux_conditioning(clip_obj, text, guidance, [ref_samples])
    
    if cfg_scale > 1.0:
        neg_cond = build_negative_conditioning(pos_cond)
    else:
        neg_cond = None

    # ── 2. Empty latent ──────────────
    # Flux latents usan channels del latent format
    inner_model = model_patcher.get_inner_model()
    latent_ch = inner_model.latent_format.latent_channels
    h_lat = height // 8
    w_lat = width // 8

    empty_latent = torch.zeros(
        (1, latent_ch, h_lat, w_lat), 
        device=device, dtype=torch.float32,
    )

    # ── 3. Noise por escena ────────
    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        (1, latent_ch, h_lat, w_lat), generator=gen, device=device,
    )

    # ── 4. CFGGuider con model clone ─
    model_clone = model_patcher.clone()
    guider = CFGGuider(model_clone)

    if cfg_scale > 1.0:
        guider.set_conds(pos_cond, neg_cond)
        guider.set_cfg(cfg_scale)
    else:
        # Flux native guidance sin CFG negative
        guider.set_conds(pos_cond, None)
        guider.set_cfg(1.0)

    # ── 5. Sigmas + KSAMPLER euler ──
    sigmas = inner_model.model_sampling.get_sigmas(steps).to(device)
    ksampler = KSAMPLER(kdsamp.sample_euler)

    # ── 6. Sample ────────────────
    def noop_callback(step, x0, x_pred):
        pass

    samples_out = guider.outter_sample(
        noise=noise,
        latent_image=empty_latent,
        sampler=ksampler,
        sigmas=sigmas,
        disable_pbar=False,
        callback=noop_callback,
        seed=seed,
    )

    # ── 7. VAE decode -> imagen ──
    decoded_out = vae_obj.decode(samples_out)
    # vae.decode returns list of dicts with 'samples' key
    if isinstance(decoded_out, list):
        decoded = decoded_out[0]["samples"]
    else:
        decoded = decoded_out

    # [B, C, H, W] -> [B, H, W, C] clamp 0-1
    img_float = decoded.to("cpu").float().clamp(0, 1)
    return img_float.permute(0, 2, 3, 1)
