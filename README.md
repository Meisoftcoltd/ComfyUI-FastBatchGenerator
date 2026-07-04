# 🐟 FishSpeech Unified Batch Generator para ComfyUI

¡Hola! Este nodo fue creado por una razón muy concreta: el proceso normal de texto a voz con FishSpeech requiere encadenar 5 o 6 nodos diferentes (cargar modelo, codificar referencia, generar tokens semánticos, decodificar audio, concatenar y guardar). Cada uno consume recursos, añade complejidad al flujo y es fácil que algo se desconecte por el camino.

**Aquí todo eso se redujo a un solo nodo.** Le das texto y te devuelve voz en formato MP3, organizada por carpetas, con opciones para ajustar volumen, velocidad y tono sin necesidad de añadir nodos extra.

---

## 🎯 ¿Qué hace exactamente?

Imagina que tienes varios archivos de texto, uno por personaje o escena. Puedes conectarlos todos aquí desde un lector de archivos, un bot de Telegram, o cualquier nodo upstream. El nodo se encarga de:

1. **Leer el texto** y dividirlo en trozos manejables (párrafos, oraciones o todo junto)
2. **Generar los tokens** usando el modelo LLaMA de FishSpeech
3. **Convertir esos tokens a audio real** con el decodificador DAC
4. **Normalizar el volumen** opcionalmente para que suene consistente
5. **Ajustar velocidad y tono** si lo necesitas, usando la misma tecnología de Spotify (Pedalboard)
6. **Guardar como MP3** en una carpeta limpia que lleva el nombre de tu archivo original

Al final tienes todos tus archivos de audio organizados y listos para conectar con PreviewAudio, Telegram o cualquier otro nodo downstream. La salida AUDIO del último archivo también está disponible directamente si necesitas previsualizarlo sin salir de ComfyUI.

---

## 🔧 ¿Cómo lo instalo?

```bash
cd tu/ComfyUI/custom_nodes/
git clone https://github.com/Meisoftcoltd/ComfyUI-FastBatchGenerator.git
pip install -r ComfyUI-FastBatchGenerator/requirements.txt
```

Reinicia ComfyUI y aparecerá como **🐟 FishSpeech Unified Batch (MP3)** bajo la categoría *Audio → Batch*.

> **Importante:** Necesitas tener instalado [ComfyUI-fish-speech](https://github.com/Meisoftcoltd/ComfyUI-fish-speech) en tu carpeta de `custom_nodes/`. Este nodo no duplica los modelos: simplemente reutiliza las funciones internas de FishSpeech para mantener todo ligero.

---

## 🎛️ Entradas principales

Solo necesitas cablear dos cosas desde upstream y el resto ya trae valores por defecto que funcionan bien:

| Campo | ¿Cómo se conecta? | Descripción |
|-------|-----------------|-------------|
| **text_list** | STRING (cableado) | El texto a convertir en voz. Puede venir de BatchTextFileReader, TelegramSuite o cualquier fuente de texto. |
| **file_names** | STRING (cableado) | Los nombres de archivo separados por coma, uno por cada bloque de texto. Se usa para crear las carpetas y nombrar los MP3 resultantes. |

A partir de ahí tienes un montón de controles que puedes ajustar sin cablear nada: desde el dispositivo donde corre cada modelo (GPU o CPU), la precisión numérica del cálculo (bfloat16, float16, float32), hasta parámetros creativos como temperatura, top-p y repetición. Todo está dentro del mismo nodo con valores por defecto seguros para empezar.

---

## 🎚️ Normalización de volumen

Activada por defecto. El nodo iguala cada fragmento a un nivel pico objetivo antes de guardar el MP3 final — esto evita que unos párrafos suenen más fuertes que otros. Puedes cambiar el nivel objetivo (`target_peak_db`) de -10 dB (suave) a 0 dB (al máximo), o desactivar toda la normalización simplemente marcando `normalize_audio` como **False**.

---

## 🏃‍♂️ Velocidad y tono con Pedalboard

Esta funcionalidad está apagada por defecto para no añadir overhead. Cuando la actives, puede:

- **Cambiar la velocidad** (`speed_factor`): 0.5x para lentitud dramática, 2.0x para locución rápida. Valores entre ambos se interpolan de forma natural y transparente gracias a los algoritmos de time-stretch de Spotify.
- **Cambiar el tono** (`pitch_shift_semitones`): de -12 semitonos (una octava más grave) a +12 (una octava más aguda). Funciona independientemente del cambio de velocidad, así que puedes ralentizar sin cambiar el tono o viceversa.

Si por alguna razón Pedalboard no está disponible en tu instalación, se registra un aviso en el log y continuea salvando el MP3 igual — nunca bloquea la ejecución.

---

## 📂 ¿Por dónde salen los archivos?

Se respeta completamente tu configuración de salida de ComfyUI (incluyendo unidades externas o montajes de red). Para cada nombre de archivo detectado se crea:

```
<tu_directorio_de_output_ComfyUI>/
└── NombreDelArchivo/
    └── NombreDelArchivo.mp3
```

Nada de números aleatorios, timestamps ni sufijos misteriosos. El archivo resultante lleva exactamente el mismo nombre que el original, sin extensión.

---

## ❓ Solución de problemas rápidos

**"No se detectan los textos individualmente"** → El parser acepta tanto saltos simples (`\n`) como dobles (`\n\n`). Si ves un desajuste en el log con los previews numéricos, revisa que cada archivo coincida con su bloque de texto correspondiente.

**"Error con torchcodec / FFmpeg"** → No debería aparecer. La codificación MP3 usa únicamente pydub escribiendo bytes PCM directamente desde memoria, sin librerías externas problemáticas. Si lo ves, asegúrate de estar en la última versión (`git pull` + `pip install -r requirements.txt`).

**"El audio suena muy bajo o distorsionado"** → Ajusta `target_peak_db`. Con el valor predeterminado (-1 dB) debería sonar alto pero sin corte. Si baja a -6 dB o inferior, el nivel general cae mucho y suena plano.

---

## ✉️ ¿Problemas, sugerencias o pull requests?

El repositorio es abierto y acepta contribuciones. Si este nodo te resultó útil para algún flujo de trabajo específico, compártelo. Si encuentra un caso límite roto, repórtalo con confianza.

*Hecho con ❤️ para flujos TTS productivos en ComfyUI.*
