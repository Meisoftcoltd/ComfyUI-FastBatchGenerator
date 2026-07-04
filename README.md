# 🚀 ComfyUI-FastBatchGenerator

**Consolidated FishSpeech TTS Batch Processor — Single Node.**

A standalone custom node for ComfyUI that replaces the entire 6-node TTS pipeline (ModelLoader → RefEncoder → TextToSemantic → DAC Decoder → AudioConcat → Save) with a single, self-contained class. Handles batch text files, manages VRAM efficiently across batches, and outputs individual `.mp3` files organized in subfolders.

---

## 🌟 Key Features

| Feature | Description |
|---|---|
| 🐟 **Full TTS Pipeline** | Loads FishSpeech LLaMA + DAC decoder, generates semantic tokens, decodes to waveform and exports high-quality MP3 — all within one node. |
| 📂 **Smart Organization** | Creates a subfolder per batch inside your ComfyUI output directory using the source filename. |
| 🏷️ **Clean Output Names** | Generates `.mp3` files that inherit the EXACT name of the input file (no random numbers, no timestamps). |
| ⚡ **VRAM-Juggling** | Intelligently moves models between GPU and CPU during inference to prevent OOM without sacrificing speed. |
| 🔊 **Volume Normalization** | FishSpeechDecoder-style normalization per chunk with configurable peak level (`target_peak_db` from -10dB to 0dB). |
| 🎸 **Time Stretch + Pitch Shift** | Pedalboard-powered post-processing: change tempo or shift pitch without sacrificing transients (Spotify's algorithm). |

---

## 🛠️ Requirements & Dependencies

**⚠️ Prerequisite:** This node calls internal FishSpeech functions (`init_llama_model`, `load_dac_model`). You must have the [ComfyUI-fish-speech](https://github.com/Meisoftcoltd/ComfyUI-fish-speech) custom node installed in `custom_nodes/`. The repo is already on your filesystem.

**Python Dependencies:**
Ensure your environment has the following installed (included in `requirements.txt`):
- `torch` & `torchaudio` — tensor manipulation + resampling
- `pydub` — silent-free MP3 encoding via PCM bytes in memory
- `numpy` — numeric array operations
- `pedalboard>=0.3.0` — time stretch / pitch shift post-processing

```bash
pip install -r ComfyUI-FastBatchGenerator/requirements.txt
```

---

## 🚀 Installation

1. Clone this repository into your `ComfyUI/custom_nodes/` folder:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/Meisoftcoltd/ComfyUI-FastBatchGenerator.git
   ```
2. Install the required Python dependencies:
   ```bash
   pip install -r ComfyUI-FastBatchGenerator/requirements.txt
   ```
3. Restart ComfyUI. The node appears under `Audio → Batch → 🔊 FishSpeech Unified Batch (MP3)`.

---

## 📖 Inputs Reference

### Required Inputs (wired from upstream nodes)

| Input | Type | Description |
|---|---|---|
| `text_list` | STRING | Full text to synthesize. Connect from BatchTextFileReader or a Telegram trigger node. |
| `file_names` | STRING | Comma-separated filenames matching each text group. Used for output naming. |

### Configuration Inputs (set directly on the node)

| Input | Type & Range | Default | Description |
|---|---|---|---|
| `checkpoint_path` | STRING | `"models/fish_speech/s2-pro"` | Path to FishSpeech model directory. |
| `llama_device` | ENUM | `cpu` | Run LLaMA on CPU or CUDA. |
| `decoder_device` | ENUM | `cuda` | Run DAC Decoder on CPU or CUDA. |
| `precision` | ENUM | `bfloat16` | Model precision: bfloat16 / float16 / float32. |
| `split_mode` | ENUM | `Párrafos` | Párrafos (newlines) · Oraciones (punctuation) · Todo el Texto |
| `temperature` | FLOAT 0.10–2.00 | `0.75` | Generation temperature for LLaMA. |
| `top_p` | FLOAT 0.10–1.00 | `0.80` | Nucleus sampling threshold. |
| `repetition_penalty` | FLOAT 0.50–2.00 | `1.10` | Repetition penalty factor. |
| `chunk_length` | INT 50–4096 | `200` | Max tokens per generation chunk. |
| `max_new_tokens` | INT 128–8192 | `4096` | Cap for generated sequence length. |
| `silence_duration` | FLOAT 0.0–5.0 | `0.3` | Silence (seconds) inserted between concatenated chunks. |

### Optional Inputs (connected from the right, defaults are backward-compatible ✅)

| Input | Type | Default | Description |
|---|---|---|---|
| `reference_audio` | AUDIO | — | Reference audio for voice cloning; pass waveform + sample_rate. |
| `prompt_text` | STRING | `""` | Extra text prompt appended to all generated segments. |
| `trigger_in` | * | — | Universal trigger passthrough for sequential workflows. |
| **`normalize_audio`** | BOOLEAN | ✅ `True` | Enable FishSpeechDecoder-style peak normalization per chunk. |
| **`target_peak_db`** | FLOAT -10→0 | `-1.0` | Normalization target peak level in decibels (lower = quieter). |
| **`time_stretch_enabled`** | BOOLEAN | ❌ `False` | Toggle Pedalboard post-processing (speed + pitch). Leave off for zero overhead. |
| **`speed_factor`** | FLOAT 0.50→2.00 | `1.0` | Playback rate: <1.0 slows down, >1.0 speeds up. Preserves pitch unless combined with shift. |
| **`pitch_shift_semitones`** | FLOAT -12→12 | `0.0` | Shift pitch ±12 semitones. Negative = deeper voice, positive = higher. Works independently of speed. |

---

## 📤 Outputs

| Port | Type | Description |
|---|---|---|
| `audio_output` | AUDIO | Last generated waveform (3D tensor) + sample rate — ready for PreviewAudio / Save nodes. |
| `batch_log` | STRING | Multi-line log with file count, processing timestamps, and output paths. |
| `trigger_out` | * | Passthrough trigger for chaining sequential batches. |

---

## 📁 Output Structure

```
<ComfyUI_Output>/
└── <Source_Filename>_No_Ext/
    └── <Source_Filename>_No_Ext.mp3
```

Uses `folder_paths.get_output_directory()` internally — fully respects your ComfyUI output path configuration (external drives, network mounts, etc.).

---

## ⚡ Processing Flow per Batch File

```
Text Input → Split (mode) → LLaMA gen tokens 
    → DAC decode → Normalize (if ON) 
        → Time Stretch + Pitch Shift (if ON)
            → Write MP3 → Output to disk + return AUDIO tensor
```

---

## 🐛 Troubleshooting

| Error | Cause & Fix |
|---|---|
| `ImportError: cannot import name 'FishSpeechUnifiedBatch' from 'node_logic'` | Node file was empty or missing the class. Pull latest via `git pull`. |
| `RuntimeError: Tensors must have same number of dimensions: got 3 and 2` | Resolved in v0.1.0. Normalizes DAC output before concat. |
| `ImportError: TorchCodec is required for save_with_torchcodec` | Resolved in v0.1.0. MP3 writing now uses pure `pydub` with raw PCM bytes — no torchaudio.save or FFmpeg required. |
| `IndexError: Dimension out of range (expected [-1, 0], but got 1)` | Resolved in v0.1.0. Output waveform is properly shaped to 3D `(batch, channels, samples)` for ComfyUI's AUDIO type. |

---

## ✅ Version History

| Tag | Date | Changes |
|---|---|---|
| **v0.1.0** | *Current* | Stabilized release: normalize_audio + target_peak_db controls, time_stretch_enabled + speed_factor + pitch_shift_semitones post-processing via pedalboard, fixed torchcodec crash (pure pydub MP3), fixed AUDIO output 3D shape compatibility. |

---

*Developed with ❤️ for high-performance TTS workflows on ComfyUI.*
