from .node_logic import FishSpeechUnifiedBatch, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .node_audio_director import (
    AudioStoryboardDirector, OllamaOptions, OllamaSystemPrompt,
    NODE_CLASS_MAPPINGS as DIRMCM,
    NODE_DISPLAY_NAME_MAPPINGS as DIRDMNM,
)

NODE_CLASS_MAPPINGS.update(DIRMCM)
NODE_DISPLAY_NAME_MAPPINGS.update(DIRDMNM)

# --- Flux Keyframes ---
try:
    from .node_flux_keyframes import (
        Flux2KeyframeGenerator,
        NODE_CLASS_MAPPINGS as FLXMCM,
        NODE_DISPLAY_NAME_MAPPINGS as FLXDMNM,
    )
    NODE_CLASS_MAPPINGS.update(FLXMCM)
    NODE_DISPLAY_NAME_MAPPINGS.update(FLXDMNM)
except Exception as e:
    print(f"[ComfyUI-FastBatchGenerator] Error cargando Flux2KeyframeGenerator: {e}")

# --- Meisoft Categorías ---
_CAT = {
    "FishSpeechUnifiedBatch": "Meisoft ⚠️/TTS",
    "AudioStoryboardDirector": "Meisoft ⚠️/TTS/Director",
    "OllamaOptions": "Meisoft ⚠️/TTS/Director",
    "OllamaSystemPrompt": "Meisoft ⚠️/TTS/Director",
    "Flux2KeyframeGenerator": "Meisoft/Video/Keyframes",
}
for _cls in NODE_CLASS_MAPPINGS.values():
    if _cls.__name__ in _CAT:
        _cls.CATEGORY = _CAT[_cls.__name__]
