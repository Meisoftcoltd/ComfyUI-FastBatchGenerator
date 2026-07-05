"""Carga de modelos Flux para keyframe pipeline."""

from typing import Any, Tuple

import folder_paths


def find_model(checkpoint_name: str) -> str | None:
    """Buscar en checkpoints/ y unet/, retorna full path o None."""
    for subdir in ["checkpoints", "unet"]:
        fp = folder_paths.get_full_path(subdir, checkpoint_name)
        if fp:
            return fp
    return None


def load_flux_models() -> Tuple[Any, Any, Any]:
    """Cargar UNet + CLIP + VAE via comfy.sd API real.

    Modelos hardcodeados del workflow:
      UNet:   flux-2-klein-9b.safetensors
      CLIP:   qwen_3_8b_fp8mixed.safetensors
      VAE:    flux2-vae.safetensors

    Returns (model_patcher, clip, vae).
    """
    # ── UNet via comfy.sd.load_diffusion_model ──
    unet_path = find_model("flux-2-klein-9b.safetensors")
    if not unet_path:
        raise FileNotFoundError(
            "flux-2-klein-9b.safetensors no encontrado en checkpoints/ ni en unet/"
        )

    from comfy.sd import load_diffusion_model
    model_patcher = load_diffusion_model(unet_path)

    # ── CLIP via comfy.sd.load_clip (type=flux2) ──
    clip_file = "qwen_3_8b_fp8mixed.safetensors"
    clip_path = folder_paths.get_full_path("clip", clip_file)
    if not clip_path:
        raise FileNotFoundError(f"{clip_file} no encontrado en clip/")

    from comfy.sd import load_clip
    clip_obj = load_clip(clip_path, type="flux2")

    # ── VAE via comfy.sd.load_vae ──
    vae_file = "flux2-vae.safetensors"
    vae_path = folder_paths.get_full_path("vae", vae_file)
    if not vae_path:
        raise FileNotFoundError(f"{vae_file} no encontrado en vae/")

    from comfy.sd import load_vae
    vae_obj = load_vae(vae_path)

    return model_patcher, clip_obj, vae_obj


def unload_flux(model_patcher: Any, clip: Any, vae: Any) -> None:
    """Liberar modelos para liberar VRAM."""
    import gc
    if hasattr(model_patcher, "to"):
        model_patcher.to("cpu")
    del model_patcher
    if hasattr(clip, "reset"):
        clip.reset()
    del clip
    if vae is not None and hasattr(vae, "reset"):
        vae.reset()
    del vae
    import torch
    gc.collect()
    torch.cuda.empty_cache()
