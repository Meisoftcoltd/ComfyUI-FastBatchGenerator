"""FishSpeech Unified Batch — Procesa lote de textos → audio (MP3) en un solo nodo.

Reemplaza toda la cadena original:
  FishSpeechModelLoader→RefEncoder→TextToSemantic→Decoder→AudioConcat

Entradas desde BatchTextFileReader: text_list + file_names.
Salidas: AUDIO último, LOG rutas MP3, trigger passthrough.
"""

import gc
import os
import pathlib
import re
import sys

import folder_paths
import torch
import torchaudio
from pydub import AudioSegment

_fs_base = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../ComfyUI-fish-speech"
)
for _p in [_fs_base, os.path.join(_fs_base, "fish_speech")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fish_speech.models.text2semantic.inference import (
    init_model as init_llama_model,
    generate_long,
)
from fish_speech.models.dac.inference import load_model as load_dac_model


# ────────────────── Helpers ──────────────────

def _split_by_newlines(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]


def _split_by_sentences(text: str) -> list[str]:
    chunks: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        s = sentence.strip()
        if s:
            chunks.append(s)
    return chunks


def _keep_all(text: str) -> list[str]:
    return [text.strip()] if text.strip() else []


def _concat_with_silence(
    tensors: list[torch.Tensor], sr: int, gap: float = 0.3,
) -> torch.Tensor | None:
    if not tensors:
        return None
    samps = int(sr * gap)
    dev = tensors[0].device
    dt = tensors[0].dtype
    sil = torch.zeros((1, samps), dtype=dt, device=dev)
    out: list[torch.Tensor] = []
    for t in tensors:
        if t.ndim == 1:
            t = t.unsqueeze(0)
        out.append(t.to(dev).to(dt))
        out.append(sil)
    return torch.cat(out, dim=1)


def _write_mp3(tensor: torch.Tensor, sr: int, path: str) -> None:
    cpu = tensor.squeeze().cpu()
    tmp = path.replace(".mp3", "_temp.wav")
    torchaudio.save(tmp, cpu.unsqueeze(0), sr)
    seg = AudioSegment.from_wav(tmp)
    seg.export(path, format="mp3", bitrate="192k")
    if os.path.exists(tmp):
        os.remove(tmp)


# ────────────────── Nodo ──────────────────

class FishSpeechUnifiedBatch:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "text_list": ("STRING", {"forceInput": True, "multiline": True}),
                "file_names": ("STRING", {"forceInput": True, "multiline": False}),
                "checkpoint_path": ("STRING", {"default": "models/fish_speech/s2-pro"}),
                "llama_device": (["cuda", "cpu"], {"default": "cpu"}),
                "decoder_device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "split_mode": ([
                    "Párrafos (Saltos de línea)",
                    "Oraciones (Signos de Puntuación)",
                    "Todo el Texto",
                ], {"default": "Párrafos (Saltos de línea)"}),
                "temperature": ("FLOAT", {"default": 0.75, "min": 0.10, "max": 2.00, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.80, "min": 0.10, "max": 1.00, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.10, "min": 0.50, "max": 2.00, "step": 0.01}),
                "chunk_length": ("INT", {"default": 200, "min": 50, "max": 4096, "step": 10}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 128, "max": 8192}),
                "silence_duration": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.1}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "prompt_text": ("STRING", {"multiline": True, "default": ""}),
                "trigger_in": ("*", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "*")
    RETURN_NAMES = ("audio_output", "batch_log", "trigger_out")
    FUNCTION = "generate"
    CATEGORY = "audio/batch"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def generate(
        self,
        text_list: str, file_names: str, checkpoint_path: str,
        llama_device: str, decoder_device: str, precision: str,
        split_mode: str, temperature: float, top_p: float,
        repetition_penalty: float, chunk_length: int, max_new_tokens: int,
        silence_duration: float,
        reference_audio=None, prompt_text: str = "", trigger_in=None,
    ):
        comfy_out = pathlib.Path(folder_paths.get_output_directory())
        logs: list[str] = []

        def _log(m):
            print(f"[FishSpeechUnifiedBatch] {m}")
            logs.append(str(m))

        # ── 0. Parsear nombres y textos ────────────────
        raw_names = [n.strip() for n in file_names.split(",") if n.strip()]
        if not raw_names:
            _log("⚠️ No se recibieron nombres de archivo.")
            empty = {"waveform": torch.zeros((1, 1, 22050)), "sample_rate": 22050}
            return (empty, "\n".join(logs), trigger_in)

        stems = [pathlib.Path(n).stem for n in raw_names]
        _log(f"📂 {len(stems)} archivos: {', '.join(raw_names)}")

        groups = [g.strip() for g in text_list.split("\n\n") if g.strip()]
        if not groups or len(groups) == 1 and "\n\n" not in text_list:
            groups = [text_list]
        while len(groups) < len(stems):
            groups.append("")

        _log(f"📝 {len(groups)} textos → {len(stems)} archivos destino")

        # ── 1. Splitter ───────────────────────────────
        if "Párrafos" in split_mode:
            splitter = _split_by_newlines
        elif "Oraciones" in split_mode:
            splitter = _split_by_sentences
        else:
            splitter = _keep_all

        # ── 2. Cargar modelos ─────────────────────────
        pt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[precision]

        _log(f"Cargando LLaMA desde {checkpoint_path}")
        llama_model, decode_one_token = init_llama_model(
            checkpoint_path=checkpoint_path, device=llama_device,
            precision=pt, compile=False,
        )

        codec_cands = [
            os.path.join(checkpoint_path, "codec.pth"),
            os.path.join(checkpoint_path, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
        ]
        cp = next((p for p in codec_cands if os.path.exists(p)), None)
        if not cp:
            raise FileNotFoundError(f"No codec en {checkpoint_path}")

        _log(f"Cargando DAC decoder")
        dec_model = load_dac_model(
            config_name="modded_dac_vq", checkpoint_path=cp, device=decoder_device,
        )

        # ── 3. Encodear referencia ────────────────────
        p_tokens = None
        if reference_audio is not None:
            _log("Extrayendo prompt tokens del audio de referencia")
            wf = reference_audio["waveform"]
            sr_in = reference_audio["sample_rate"]
            dev = next(dec_model.parameters()).device
            if wf.shape[1] > 1:
                wf = wf.mean(dim=1, keepdim=True)
            wf = torchaudio.functional.resample(wf, sr_in, dec_model.sample_rate).to(dev)
            alen = torch.tensor([wf.shape[2]], device=dev, dtype=torch.long)
            with torch.no_grad():
                idxs, _ = dec_model.encode(wf, alen)
                if idxs.ndim == 3:
                    idxs = idxs[0]
            p_tokens = idxs.to(dev)

        # ── 4. Loop archivos ──────────────────────────
        last_audio = None
        mp3_list: list[str] = []

        for i, (stem, txt) in enumerate(zip(stems, groups)):
            _log(f"\n{'=' * 40}")
            _log(f"Archivo {i + 1}/{len(stems)}: {stem}")

            if not txt.strip():
                _log("   Texto vacío, saltando")
                continue

            out_dir = comfy_out / stem
            out_dir.mkdir(parents=True, exist_ok=True)
            mp3_path = str(out_dir / f"{stem}.mp3")

            chunks = splitter(txt)
            _log(f"   → {len(chunks)} chunk(s)")

            audios: list[torch.Tensor] = []

            for ci, chunk in enumerate(chunks):
                full_text = f"<|speaker:0|> {chunk}"
                _log(f"     ├─ Chunk {ci+1}: LLaMA → tokens")

                # Subir LLaMA a GPU si es cuda
                torch.cuda.synchronize()
                llama_model.to(llama_device)
                llama_model.config.max_seq_len = 2048
                with torch.device(llama_device):
                    llama_model.setup_caches(
                        max_batch_size=1, max_seq_len=2048,
                        dtype=next(llama_model.parameters()).dtype,
                    )

                gen = generate_long(
                    model=llama_model, device=llama_device,
                    decode_one_token=decode_one_token, text=full_text,
                    num_samples=1, max_new_tokens=max_new_tokens,
                    top_p=top_p, repetition_penalty=repetition_penalty,
                    temperature=temperature, compile=False,
                    iterative_prompt=True, chunk_length=chunk_length,
                    prompt_text=prompt_text if prompt_text else None,
                    prompt_tokens=p_tokens if p_tokens is not None else None,
                )

                codes = []
                for resp in gen:
                    if resp.action == "sample":
                        codes.append(resp.codes)

                if not codes:
                    _log(f"     ⚠️ LLaMA sin tokens para chunk {ci+1}")
                    continue

                sem = torch.cat(codes, dim=1).cpu()
                del codes

                llama_model.to("cpu")
                gc.collect(); torch.cuda.empty_cache()

                # Decodificar DAC
                _log(f"     ├─ Chunk {ci+1}: DAC → audio")
                ddev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                torch.cuda.synchronize()
                dec_model.to(ddev)

                igp = sem.to(ddev)
                if igp.ndim == 2:
                    igp = igp.unsqueeze(0)

                with torch.no_grad():
                    fake = dec_model.from_indices(igp)

                wav_out = fake.cpu()
                del igp, sem, fake

                # Normalizar -1 dB peak
                mx = torch.max(torch.abs(wav_out))
                if mx > 0:
                    tgt = 10 ** (-1.0 / 20)
                    wav_out = wav_out * (tgt / mx)

                sr_out = dec_model.sample_rate

                dec_model.to("cpu")
                gc.collect(); torch.cuda.empty_cache()

                audios.append(wav_out)

            if not audios:
                _log("   ⚠️ Sin audio generado, saltando")
                continue

            # Concatenar chunks con silencio
            final = _concat_with_silence(audios, sr_out, silence_duration)
            del audios

            if final is not None:
                _write_mp3(final, sr_out, mp3_path)
                _log(f"   ✅ MP3 guardado: {mp3_path}")
                last_audio = {"waveform": final, "sample_rate": sr_out}
                mp3_list.append(mp3_path)
            else:
                _log("   ⚠️ Formas de onda vacía")

            del final
            gc.collect()

        # ── 5. Limpieza final ────────────────────────
        _log(f"\n{'=' * 40}")
        _log(f"Lote completado: {len(mp3_list)}/{len(stems)} MP3 generado(s)")
        gc.collect()

        if last_audio is None:
            empty = {"waveform": torch.zeros((1, 1, 22050)), "sample_rate": 22050}
            return (empty, "\n".join(logs), trigger_in)

        return (last_audio, "\n".join(logs), trigger_in)


NODE_CLASS_MAPPINGS = {
    "FishSpeechUnifiedBatch": FishSpeechUnifiedBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FishSpeechUnifiedBatch": "🐟 FishSpeech Unified Batch (MP3)",
}
