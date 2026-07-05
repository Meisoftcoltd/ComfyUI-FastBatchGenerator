"""Audio Storyboard Director - Transcribe audio via Whisper -> Ollama AGENTE 1.

Flujo interno:
  1. Recibe AUDIO + config fps/ventanas
  2. FishSpeechWhisperTranscriber transcribe -> texto con ventanas
  3. Parsea numero de ventanas N
  4. Ollama AGENTE 1 genera exactamente N escenas JSON
  5. Guarda [stem]/[stem].json en output/ (folder_paths)
  6. Retorna (raw_json, scene_count, json_path, trigger_out)
"""

import gc
import json
import os
import pathlib
import re
import sys
from typing import Any, Tuple, Optional

import folder_paths
import torch

_fs_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../ComfyUI-fish-speech")
for _p in [_fs_base, os.path.join(_fs_base, "fish_speech")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fish_speech.nodes import FishSpeechWhisperTranscriber

try:
    import ollama
except ImportError:
    ollama = None


def _sanitize_stem(file_name: str) -> str:
    stem = pathlib.Path(str(file_name)).stem
    for ch in ["<", ">", ":", '"', "/", "\\", "|", "?", "*", " ", "."]:
        stem = stem.replace(ch, "_")
    return str(stem) if stem else "unnamed_project"


def _get_output_dir(file_name: str) -> pathlib.Path:
    base = pathlib.Path(folder_paths.get_output_directory())
    return base / _sanitize_stem(file_name)


def _connect_ollama(url: str, model: str) -> Tuple[Optional[Any], Optional[str]]:
    if ollama is None:
        return None, "ollama package no instalado"
    try:
        client = ollama.Client(host=url)
        client.list()
        return client, None
    except Exception as e:
        return None, f"Ollama unreachable: {e}"


def _query_ollama_agent(
    client: Any, model_name: str, system_prompt: str, user_prompt: str,
    options_dict: dict, retries: int = 2,
) -> Tuple[str, Optional[str]]:
    last_err: Optional[str] = None
    for attempt in range(max(retries, 1)):
        try:
            resp = client.chat(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                options=options_dict, stream=False)
            return resp["message"]["content"].strip(), None
        except Exception as e:
            last_err = str(e)
            print(f"[AudioStoryboardDirector] Retry {attempt+1}/{retries}: {last_err}")
    return "", last_err


def _parse_json(text: str) -> Tuple[Any, Optional[str]]:
    try:
        return json.loads(text.strip()), None
    except json.JSONDecodeError:
        pass
    match = re.search(r"```:\s*(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip()), None
        except json.JSONDecodeError as e2:
            return None, str(e2)
    return None, f"JSON invalido: {text[:200]}"


def _build_transcript_prompt(transcribed_text: str, n_windows: int) -> str:
    hdr = "<BEGIN TRANSCRIPT>" + transcribed_text + "</END TRANSCRIPT>" + "\n\n"
    return hdr + f"WINDOWS COUNT = {n_windows}. Generate exactly {n_windows} scenes."


def _count_windows_in_transcript(text: str) -> int:
    return len(re.findall(r"Ventana\s+(\d+)", text)) or 1


def _make_ollama_options(
    num_ctx: int = 32768, temperature: float = 1.0, top_k: int = 64,
    top_p: float = 0.95, min_p: float = 0.05,
) -> dict:
    return {"num_ctx": num_ctx, "temperature": temperature,
            "top_k": top_k, "top_p": top_p, "min_p": min_p}


DIRECTOR_SYSTEM_PROMPT_DEFAULT = (
    'You are the Lead Storyboard Director for an AI video project.'
    ' Analyze transcript windows and generate scene descriptions.'
    ' MANDATORY RULES: 1. EXACT MAPPING: N scenes where N = WINDOWS COUNT.'
    ' 2. Output ONLY valid JSON with key scenes containing dicts with keys:'
    ' scene_id, start_frame, end_frame, frame_count, flux_prompt, wan_prompt'
    ' 3. Every scene references the SAME protagonist. 4. Plain raw JSON only.')

# ========== CLASS: OllamaOptions ==========
class OllamaOptions:
    """Config reusable de params Ollama."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            'required': {
                'url': ('STRING', {'default': 'http://localhost:11434'}),
                'model': ('STRING', {'default': 'devstral-small-2-256k:latest'}),
                'num_ctx': ('INT', {'default': 32768, 'min': 4096, 'max': 131072}),
                'temperature': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 2.0, 'step': 0.1}),
                'top_k': ('INT', {'default': 64, 'min': 1, 'max': 256}),
                'top_p': ('FLOAT', {'default': 0.95, 'min': 0.0, 'max': 1.0, 'step': 0.05}),
                'min_p': ('FLOAT', {'default': 0.05, 'min': 0.0, 'max': 1.0, 'step': 0.05}),
            }
        }

    RETURN_TYPES = ('OLLAMA_OPT',)
    RETURN_NAMES = ('ollama_config',)
    FUNCTION = 'build'
    CATEGORY = 'Video/Director'

    def build(
        self, url: str, model: str, num_ctx: int,
        temperature: float, top_k: int, top_p: float, min_p: float):
        opts = _make_ollama_options(num_ctx, temperature, top_k, top_p, min_p)
        cfg = {'url': url, 'model': model} | opts
        print(f'[OllamaOptions] Configurada: {cfg}')
        return (cfg,)


class OllamaSystemPrompt:
    """System Prompt configurable desde TXT externo o inline."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            'required': {
                'system_text': ('STRING', {'default': '', 'multiline': True}),
            },
            'optional': {
                'load_from_file': ('STRING', {'forceInput': True}),
            }
        }

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('system_prompt',)
    FUNCTION = 'resolve'
    CATEGORY = 'Video/Director'

    def resolve(self, system_text: str, load_from_file=None):
        if load_from_file and os.path.isfile(load_from_file):
            with open(load_from_file, 'r', encoding='utf-8') as fh:
                txt = fh.read().strip()
                if txt: system_text = txt
        return (system_text,)


class AudioStoryboardDirector:
    """Director de Storyboard - Whisper transcripcion -> Ollama AGENTE 1 -> N escenas JSON."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            'required': {
                'audio': ('AUDIO',),
                'fps': ('FLOAT', {'default': 12.0, 'min': 5.0, 'max': 60.0, 'step': 1.0}),
                'frame_window': ('INT', {'default': 41, 'min': 10, 'max': 200, 'step': 1}),
                'motion_frame': ('INT', {'default': 13, 'min': 0, 'max': 1024}),
                'file_name': ('STRING', {'default': 'project_audio', 'multiline': False}),
                'ollama_url': ('STRING', {'default': 'http://localhost:11434'}),
                'ollama_model': ('STRING', {'default': 'devstral-small-2-256k:latest'}),
            },
            'optional': {
                'ollama_options': ('OLLAMA_OPT', {'forceInput': True}),
                'system_prompt_director': (
                    'STRING', {
                        'default': DIRECTOR_SYSTEM_PROMPT_DEFAULT,
                        'multiline': True, 'forceInput': True,
                    }),
                'trigger_in': ('*', {'forceInput': True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "*")
    RETURN_NAMES = ("agent_json_string", "scene_count", "json_file_path", "trigger_out")
    FUNCTION = "generate_storyboard"
    CATEGORY = "Video/Director"
    OUTPUT_NODE = True

    def generate_storyboard(self, audio, fps, frame_window, motion_frame, file_name,
        ollama_url="http://localhost:11434", ollama_model="devstral-small-2-256k:latest",
        ollama_options=None, system_prompt_director=None, trigger_in=None):

        logs = []
        def _log(msg):
            print(f"[AudioStoryboardDirector] {msg}")
            logs.append(str(msg))

        effective_opts = (ollama_options if ollama_options else {}) | {
            'num_ctx': 32768, 'temperature': 1.0, 'top_k': 64,
            'top_p': 0.95, 'min_p': 0.05,}
        model_name = effective_opts.pop("model", ollama_model)
        server_url = effective_opts.get("url", ollama_url)

        sys_prompt = (
            system_prompt_director
            if system_prompt_director and system_prompt_director.strip()
            else DIRECTOR_SYSTEM_PROMPT_DEFAULT
        )

        _log(f"Proyecto: {file_name}")
        _log(f"fps={fps}, fw={frame_window}, mf={motion_frame}")

        # --- 1. Transcribir con Whisper ---
        _log("Ejecutando transcripcion Whisper...")
        try:
            transcript_text, = FishSpeechWhisperTranscriber.transcribe(
                model_size='medium', language='auto', device='cuda',
                output_format='video_windows', fps=fps,
                frame_window=frame_window, motion_frame=motion_frame, audio=audio)
        except Exception as e:
            _log(f'Whisper fallo: {e}')
            return (json.dumps({"scenes": []}), 0, "", trigger_in)

        # --- 2. Contar ventanas ---
        n_windows = _count_windows_in_transcript(transcript_text)
        _log(f"{n_windows} ventanas detectadas en transcripcion")
        if n_windows <= 0:
            return (json.dumps({"scenes": []}), 0, "", trigger_in)

        # --- Helper para write empty JSON placeholder ---
        def _empty_return(err_msg: str) -> tuple:
            proj_dir = _get_output_dir(file_name)
            proj_dir.mkdir(parents=True, exist_ok=True)
            jp = str(proj_dir / f"{_sanitize_stem(file_name)}.json")
            with open(jp, "w", encoding="utf-8") as fp:
                json.dump({"scenes": []}, fp)
            return (json.dumps({"scenes": []}), 0, jp, trigger_in)

        # --- 3. Conectar Ollama ---
        client, err = _connect_ollama(server_url, model_name)
        if not client:
            _log(f'Ollama sin conexion: {err}')
            return _empty_return(err or "Error desconocido de conexión Ollama")

        _log(f"Conectado a {server_url} / modelo {model_name}")
        # --- 4. Call AGENTE Ollama ---
        user_prompt = _build_transcript_prompt(transcript_text, n_windows)
        _log(f"Consultando AGENTE 1 para {n_windows} escenas...")
        raw_response, err = _query_ollama_agent(
            client=client, model_name=model_name,
            system_prompt=sys_prompt, user_prompt=user_prompt,
            options_dict=effective_opts, retries=2)

        if err:
            _log(f'Fallo Ollama: {err}')
            return _empty_return(err)

        # --- 5. Parsear y validar JSON ---
        parsed_json, parse_err = _parse_json(raw_response)
        if parse_err:
            _log(f'Error parseo JSON: {parse_err}')
            _log(f"Preview respuesta: {raw_response[:300]}...")
            return _empty_return(str(parse_err))

        valid_scenes = parsed_json.get("scenes", [])
        scene_count = len(valid_scenes)
        if scene_count != n_windows and scene_count > 0:
            _log(f"Generadas {scene_count} escenas vs {n_windows} ventanas esperadas")
        elif scene_count == 0:
            _log('AGENTE devolvio 0 escenas, generando JSON vacio')
        _log(f"Storyboard listo: {scene_count} escena(s)")

        # --- 6. Persistir JSON a disco ---
        proj_dir = _get_output_dir(file_name)
        proj_dir.mkdir(parents=True, exist_ok=True)
        json_path = str(proj_dir / f"{_sanitize_stem(file_name)}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as fp:
                json.dump(parsed_json, fp, indent=2, ensure_ascii=False)
            _log(f"JSON guardado en {json_path}")
        except OSError as e:
            _log(f"Error al guardar JSON: {e}")

        # --- 7. Retorno ---
        raw_json_str = json.dumps(parsed_json, indent=2, ensure_ascii=False)
        return (raw_json_str, scene_count, json_path, trigger_in)


# ═════════════ Mapeos ComfyUI ═════════════
NODE_CLASS_MAPPINGS = {
    "AudioStoryboardDirector": AudioStoryboardDirector,
    "OllamaOptions": OllamaOptions,
    "OllamaSystemPrompt": OllamaSystemPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioStoryboardDirector": "🎬 Audio Storyboard Director",
    "OllamaOptions": "⚙️ Ollama Options Config",
    "OllamaSystemPrompt": "📋 Ollama System Prompt",
}
