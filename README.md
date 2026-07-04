# 🐟 FishSpeech Unified Batch Generator — ComfyUI Custom Node

Consolida toda la cadena de Text-to-Speech de FishSpeech (Carga Modelo → Encoder Referencia → Generación Semántica LLaMA → Decodificación DAC → Concatenación) en **un solo nodo**. Procesa lotes de archivos de texto, gestiona eficientemente la VRAM y guarda los resultados como MP3 individuales organizados por proyecto.

---

## 📥 Instalación

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Meisoftcoltd/ComfyUI-FastBatchGenerator.git
pip install -r ComfyUI-FastBatchGenerator/requirements.txt
# Reiniciar ComfyUI
```

**Requerimientos previos:** Necesitas [ComfyUI-fish-speech](https://github.com/Meisoftcoltd/ComfyUI-fish-speech) instalado en `custom_nodes/` — este nodo llama a sus funciones internas (`init_llama_model`, `load_dac_model`).

---

## 🔧 Inputs del Nodo

### Entradas Requeridas (cableadas desde upstream)

| Input | Tipo | Descripción |
|-------|------|-------------|
| `text_list` | STRING | Texto completo a sintetizar. Conectar desde BatchTextFileReader o nodo Telegram. |
| `file_names` | STRING | Nombres de archivo separados por coma. Usado para nomenclatura de salida. |

### Configuración del Modelo (ajustable en el nodo)

| Input | Tipo & Rango | Default | Descripción |
|-------|-------------|---------|-------------|
| `checkpoint_path` | STRING | `"models/fish_speech/s2-pro"` | Ruta al directorio del modelo FishSpeech. |
| `llama_device` | ENUM: `cuda`, `cpu` | `cpu` | Dispositivo para generar tokens LLaMA. |
| `decoder_device` | ENUM: `cuda`, `cpu` | `cuda` | Dispositivo para el DAC decoder. |
| `precision` | ENUM: `bfloat16`, `float16`, `float32` | `bfloat16` | Precisión del modelo (menor = menos VRAM). |
| `split_mode` | ENUM | `Párrafos` | Criterio de división de texto en chunks. |
| `temperature` | FLOAT 0.10–2.00 | `0.75` | Temperatura de generación LLaMA. |
| `top_p` | FLOAT 0.10–1.00 | `0.80` | Umbral de muestreo nucleus. |
| `repetition_penalty` | FLOAT 0.50–2.00 | `1.10` | Factor contra repeticiones. |
| `chunk_length` | INT 50–4096 | `200` | Tokens máximos por chunk de generación. |
| `max_new_tokens` | INT 128–8192 | `4096` | Límite total de tokens generados. |
| `silence_duration` | FLOAT 0.0–5.0 | `0.3` | Silencio (segundos) entre chunks concatenados. |

### Entradas Opcionales (conectar desde la derecha, defaults backward-compatible ✅)

| Input | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `reference_audio` | AUDIO | — | Audio de referencia para voice cloning. |
| `prompt_text` | STRING | `""` | Texto extra añadido a cada segmento generado. |
| `trigger_in` | * | — | Passthrough para cadenas secuenciales. |

### 🔊 Control de Normalización (desde FishSpeechDecoder)

| Input | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| **`normalize_audio`** | BOOLEAN | ✅ `True` | Activar normalización de pico por chunk. |
| **`target_peak_db`** | FLOAT -10→0 | `-1.0` | Nivel pico objetivo en decibelios (más bajo = más suave). |

### 🎸 Control de Time Stretch / Pitch (desde AudioTimeStretchPedalboard)

| Input | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| **`time_stretch_enabled`** | BOOLEAN | ❌ `False` | Activa post-procesamiento con Pedalboard. Apagado = sin overhead. |
| **`speed_factor`** | FLOAT 0.50→2.00 | `1.0` | Factor de velocidad: <1 ralentiza, >1 acelera. |
| **`pitch_shift_semitones`** | FLOAT -12→12 | `0.0` | Desplazamiento de tono en semitonos. Negativo = más grave, positivo = más agudo. |

---

## 📤 Outputs

| Puerto | Tipo | Descripción |
|--------|------|-------------|
| `audio_output` | AUDIO | Última forma de onda generada (3D tensor) + sample_rate — lista para PreviewAudio / Save. |
| `batch_log` | STRING | Log multilinea con conteos, timestamps y rutas de salida. |
| `trigger_out` | * | Passthrough trigger para encadenar batches secuenciales. |

---

## 📁 Estructura de Salida

```
<ComfyUI_Output>/
└── <Stem_Del_Archivo_Source>/
    └── <Stem_Del_Archivo_Source>.mp3
```

Usa internamente `folder_paths.get_output_directory()` — respeta completamente tu configuración de salida de ComfyUI (unidades externas, mounts de red, etc.).

---

## ⚙️ Flujo de Procesamiento por Archivo del Batch

```
Texto Input → Split (modo configurado) 
    → LLaMA genera tokens 
        → DAC decodifica a waveform 
            → Normalizar volumen (si normalize_audio = ON)
                → Time Stretch + Pitch Shift (si time_stretch_enabled = ON)
                    → Escribe MP3 → Disco + retorna tensor AUDIO 3D para ComfyUI
```

---

## 📋 Versionado

| Tag | Fecha | Cambios |
|-----|-------|---------|
| **v0.1.0** | *Current* | Lanzamiento estable: normalización configurable (`normalize_audio`, `target_peak_db`), post-procesamiento time stretch/pitch via pedalboard, solución torchcodec (MP3 puro con pydub), tensor AUDIO 3D compatible, parser robusto de textos múltiples en `\n` o `\n\n`. |

---

*Desarrollado con ❤️ para flujos TTS de alto rendimiento en ComfyUI.*
