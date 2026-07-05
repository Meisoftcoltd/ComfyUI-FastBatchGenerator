"""Flux2KeyframeGenerator — Genera N keyframes batch desde JSON storyboard + imagen de referencia."""

import gc
import json
from typing import Any, List, Optional, Tuple

import torch

# Módulos propios del paquete
from .flux_helpers import get_output_dir, resolve_aspect, save_image
from .flux_ollama import connect_ollama, query_agent, AGENTE_2_SYSTEM_PROMPT
from .flux_models import load_flux_models, unload_flux
from .flux_clip import clip_tokenize_encode, clip_encode_zeros


# ═══════════════ CLASE PRINCIPAL ═══════════════

class Flux2KeyframeGenerator:

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "agent_json_string": ("STRING", {"forceInput": True}),
                "reference_image": ("IMAGE",),
                "file_name": ("STRING", {"default": "project_audio"}),
                "aspect_ratio": (["9:16", "16:9", "1:1", "4:3", "3:4"], {"default": "9:16"}),
                "base_resolution": ("INT", {"default": 1024, "min": 512, "max": 2048}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 20.6, "min": 1.0, "step": 0.1}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "step": 0.1}),
                "seed": ("INT", {"default": 42}),
                "ollama_url": ("STRING", {"default": "http://localhost:11434"}),
                "ollama_model": ("STRING", {"default": "devstral-small-2-256k:latest"}),
            },
            "optional": {
                "ollama_options": ("OLLAMA_OPT", {"forceInput": True}),
                "system_prompt_engineer": (
                    "STRING", {
                        "default": AGENTE_2_SYSTEM_PROMPT,
                        "multiline": True, "forceInput": True,
                    }),
                "use_ollama_refinement": ("BOOLEAN", {"default": True}),
                "trigger_in": ("*", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE_LIST", "STRING", "STRING", "*")
    RETURN_NAMES = ("keyframes", "batch_log", "keyframes_dir", "trigger_out")
    FUNCTION = "generate_keyframes"
    CATEGORY = "Video/Keyframes"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    # ── Función principal del nodo ────────────────

    def generate_keyframes(
        self, agent_json_string, reference_image, file_name, aspect_ratio="9:16",
        base_resolution=1024, steps=35, guidance=20.6, cfg_scale=1.0, seed=42,
        ollama_url="http://localhost:11434", ollama_model="devstral-small-2-256k:latest",
        ollama_options=None, system_prompt_engineer=None,
        use_ollama_refinement=True, trigger_in=None,
    ):
        logs: List[str] = []
        def _log(msg: str):
            print(f"[Flux2KG] {msg}")
            logs.append(str(msg))

        # ── 0. Resolución ────────────────────────────────
        width, height = resolve_aspect(aspect_ratio, base_resolution)
        _log(f"Resolucion: {width}x{height} ({aspect_ratio})")

        if reference_image.ndim == 3:
            reference_image = reference_image.unsqueeze(0)

        # ── 1. Parsear JSON del storyboard ───────────────
        try:
            parsed = json.loads(agent_json_string)
        except json.JSONDecodeError as e:
            _log(f"JSON invalido: {e}")
            return ([], "\n".join(logs), "", trigger_in)

        scenes = parsed.get("scenes", [])
        n_scenes = len(scenes)
        if n_scenes == 0:
            _log("No se encontraron escenas en el JSON")
            return ([], "\n".join(logs), "", trigger_in)

        _log(f"Generando {n_scenes} keyframe(s)")

        # ── 2. Config Ollama AGT 2 ─────────────────────
        eff_opts = ollama_options if isinstance(ollama_options, dict) else {}
        ollama_cfg = {
            "num_ctx": eff_opts.get("num_ctx", 32768),
            "temperature": eff_opts.get("temperature", 1.0),
            "top_k": eff_opts.get("top_k", 64),
            "top_p": eff_opts.get("top_p", 0.95),
            "min_p": eff_opts.get("min_p", 0.05),
        }
        server_url = eff_opts.get("url", ollama_url)
        model_name = eff_opts.get("model", ollama_model)

        sys_prompt = (
            system_prompt_engineer
            if system_prompt_engineer and system_prompt_engineer.strip()
            else AGENTE_2_SYSTEM_PROMPT
        )

        ollama_client = None
        if use_ollama_refinement:
            client, err = connect_ollama(server_url, model_name)
            if err:
                _log(f"Ollama no disponible ({err}), usando prompts sin refinamiento")
                use_ollama_refinement = False
            else:
                ollama_client = client
                _log(f"Ollama conectado ({server_url}/{model_name})")

        # ── 3. Cargar modelos Flux2 ────────────────────
        _log("Cargando modelos Flux2...")
        model_patcher, clip_obj, vae_obj = load_flux_models()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _log(f"Modelos cargados en {device}")

        # ── 4. Pre-encode referencia image -> latents
        ref_gpu = reference_image.to(device).float()
        if ref_gpu.ndim == 4 and ref_gpu.shape[1] in (3, 4):
            ref_pivoted = ref_gpu.permute(0, 3, 1, 2)

        _log(f"Encoding reference image shape={ref_gpu.shape}")
        try:
            ref_latent_out = vae_obj.encode(ref_pivoted)
        except Exception as e:
            _log(f"Error encoding reference: {e}")
            return ([], "\n".join(logs), "", trigger_in)

        actual_latents = []
        if isinstance(ref_latent_out, list):
            for item in ref_latent_out:
                if isinstance(item, dict) and "samples" in item:
                    actual_latents.append(item["samples"])
        
        if not actual_latents:
            _log("Sin samples reference latent")
            return ([], "\n".join(logs), "", trigger_in)

        ref_samples = actual_latents[0]
        _log(f"Ref latents shape: {ref_samples.shape}")

        # ── 5. Directorio de salida ────────────────────
        proj_dir = get_output_dir(file_name)
        keyframes_dir = proj_dir / "keyframes"
        keyframes_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Keyframes en: {keyframes_dir}")

        # ── 6. Loop escenas -> generar keyframes ───────
        generated_images: List[torch.Tensor] = []

        for idx, scene in enumerate(scenes):
            scene_id = scene.get("scene_id", idx)
            prompt_text = scene.get("flux_prompt", "").strip()

            if not prompt_text:
                _log(f"   Escena {idx}: prompt vacio, saltando")
                continue

            _log(f"Escena {idx + 1}/{n_scenes} (id={scene_id})")

            # ── Refinar prompt con Ollama AGT 2 ───
            final_prompt = prompt_text
            if use_ollama_refinement and ollama_client is not None:
                try:
                    refined, err = query_agent(
                        client=ollama_client, model_name=model_name,
                        system_prompt=sys_prompt,
                        user_prompt=f"Scene {scene_id}: {prompt_text}",
                        options_dict=ollama_cfg,
                    )
                    if err:
                        _log(f"   Ollama fallo en escena {idx}: {err}")
                    else:
                        final_prompt = refined
                        _log(f"   Prompt refinado ({len(final_prompt)} chars)")
                except Exception as e:
                    _log(f"   Error refinando prompt: {e}")

            # ── Generar keyframe via Flux2 sampling ──
            try:
                scene_seed = seed + idx * 137
                img = self._sample_one_scene(
                    model_patcher=model_patcher, clip_obj=clip_obj, vae_obj=vae_obj,
                    device=device, text=final_prompt, guidance=guidance,
                    cfg_scale=cfg_scale, ref_samples=ref_samples,
                    width=width, height=height, steps=steps, seed=scene_seed,
                )

                # ── Guardar PNG ───────
                png_name = f"scene_{scene_id:04d}.png"
                png_path = str(keyframes_dir / png_name)
                save_image(img, png_path)
                generated_images.append(img)
                _log(f"   OK {png_name}")

            except Exception as e:
                _log(f"   ERROR escena {idx}: {e}")
                import traceback
                _log(traceback.format_exc())

            # VRAM cleanup entre escenas
            gc.collect()

        # ── 7. Retorno final ──────────────────────
        n_ok = len(generated_images)
        _log(f"Lote completado: {n_ok}/{n_scenes} keyframes")

        if not generated_images:
            return ([], "\n".join(logs), "", trigger_in)

        batch_stack = torch.cat(generated_images, dim=0)

        # Liberar modelos
        _log("Liberando modelos...")
        unload_flux(model_patcher, clip_obj, vae_obj)

        return (batch_stack, "\n".join(logs), str(keyframes_dir), trigger_in)

    # ── Helpers internos de sampling ────────────────

    def _build_flux_cond(self, clip_obj, text, guidance, ref_samples):
        tokens = clip_obj.tokenize(text)
        pos = clip_obj.encode(tokens)
        out = []
        for c in pos:
            if isinstance(c, list) and len(c) >= 2:
                p1, p2 = c[0], dict(c[1])
                import torch as _t
                p2["guidance"] = _t.tensor([guidance], dtype=_t.float32)
                p2["ref_latents"] = [ref_samples]
                out.append([p1, p2])
            else:
                out.append(c)
        return out

    def _build_neg_cond(self, pos_cond):
        neg = []
        for c in pos_cond:
            if isinstance(c, list) and len(c) >= 2:
                p1, p2 = c[0], {}
                for k, v in c[1].items():
                    if k not in ("guidance", "ref_latents") and hasattr(v, "zeros_like"):
                        p2[k] = v.zeros_like()
                neg.append([p1, p2])
            else:
                neg.append(c)
        return neg

    def _sample_one_scene(
        self, model_patcher, clip_obj, vae_obj, device,
        text, guidance, cfg_scale, ref_samples,
        width, height, steps, seed,
    ):
        """Flux2 sampling para una sola escena via Reference Latent."""
        from comfy.samplers import CFGGuider, KSAMPLER
        import comfy.k_diffusion.sampling as kdsamp

        # 1. Build conditioning
        pos_cond = self._build_flux_cond(clip_obj, text, guidance, ref_samples)
        neg_cond = self._build_neg_cond(pos_cond) if cfg_scale > 1.0 else None

        # 2. Empty latent
        inner = model_patcher.get_inner_model()
        latent_ch = inner.latent_format.latent_channels
        h_l, w_l = height // 8, width // 8
        empty_lat = torch.zeros((1, latent_ch, h_l, w_l), device=device, dtype=torch.float32)

        # 3. Noise
        gen = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn((1, latent_ch, h_l, w_l), generator=gen, device=device)

        # 4. CFGGuider
        mp = model_patcher.clone()
        guider = CFGGuider(mp)
        if cfg_scale > 1.0:
            guider.set_conds(pos_cond, neg_cond)
            guider.set_cfg(cfg_scale)
        else:
            guider.set_conds(pos_cond, None)
            guider.set_cfg(1.0)

        # 5. Sigmas + sampler
        sigmas = inner.model_sampling.get_sigmas(steps).to(device)
        ksampler = KSAMPLER(kdsamp.sample_euler)

        # 6. Sample
        samples_out = guider.outter_sample(
            noise=noise, latent_image=empty_lat, sampler=ksampler,
            sigmas=sigmas, disable_pbar=False, callback=lambda *a: None, seed=seed,
        )

        # 7. VAE decode -> imagen
        decoded_out = vae_obj.decode(samples_out)
        if isinstance(decoded_out, list):
            decoded = decoded_out[0] if "samples" not in decoded_out[0] else decoded_out[0]["samples"]
        else:
            decoded = decoded_out

        img = decoded.to("cpu").float().clamp(0, 1)
        return img.permute(0, 2, 3, 1)


# ═════════ Mapeos ComfyUI ════

NODE_CLASS_MAPPINGS = {
    "Flux2KeyframeGenerator": Flux2KeyframeGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KeyframeGenerator": "🖼️ Flux2 Keyframe Generator",
}
