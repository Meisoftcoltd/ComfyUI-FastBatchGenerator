import importlib.util as _iutil
import sys as _sys
import os as _os

__dir__ = _os.path.dirname(_os.path.realpath(__file__))

# Add our own dir to path so sibling module imports resolve correctly
if __dir__ not in _sys.path:
    _sys.path.insert(0, __dir__)

def _load(name, path):
    spec = _iutil.spec_from_file_location(name, path)
    mod = _iutil.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# ── Load flux helpers first (no deps on sibling node modules) ──
_load("flux_helpers", __dir__ + "/flux_helpers.py")
_load("flux_ollama", __dir__ + "/flux_ollama.py")
_load("flux_models", __dir__ + "/flux_models.py")
_load("flux_clip", __dir__ + "/flux_clip.py")
_load("flux_sampler", __dir__ + "/flux_sampler.py")

# ── Load node modules (they can now import helpers by name) ──
_logic    = _load("node_logic",       __dir__ + "/node_logic.py")
_director = _load("node_audio_director",  __dir__ + "/node_audio_director.py")
_flux_kf  = _load("node_flux_keyframes",  __dir__ + "/node_flux_keyframes.py")

# ── Merge all NODE_CLASS/NAME mappings ──
NODE_CLASS_MAPPINGS = dict(_logic.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(_logic.NODE_DISPLAY_NAME_MAPPINGS)

for mod in [_director, _flux_kf]:
    NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(mod.NODE_DISPLAY_NAME_MAPPINGS)

# ── Override CATEGORY per node ──
_CAT = {
    "FishSpeechUnifiedBatch": "Meisoft ⚠️/TTS",
    "AudioStoryboardDirector": "Meisoft ⚠️/TTS/Director",
    "OllamaOptions": "Meisoft ⚠️/TTS/Director",
    "OllamaSystemPrompt": "Meisoft ⚠️/TTS/Director",
    "Flux2KeyframeGenerator": "Meisoft/Video/Keyframes",
}
for cls in NODE_CLASS_MAPPINGS.values():
    name = cls.__name__
    if name in _CAT:
        cls.CATEGORY = _CAT[name]
