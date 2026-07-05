# 🎬🎯 FastBatch Generator | Pipeline TTS → Storyboard → Keyframes

¡Hola! Este repositorio nace de una necesidad muy concreta del día a día en ComfyUI: los workflows que conectan "texto a voz", "generación de storyboards" y "creación de keyframes visuales" terminan siendo un laberinto de 15 o más nodos interconectados, difíciles de mantener, propensos a desconexiones inesperadas entre pasos, y con una curva de entrada bastante alta si no estás familiarizado con todos los parámetros internos.

**El objetivo aquí es consolidar ese laberinto en una serie de módulos grandes pero intuitivos que cubren todo el proceso:** desde convertir texto en voz (FishSpeechUnifiedBatch), hasta crear un storyboard inteligente a partir del audio transcrito (AudioStoryboardDirector + helpers Ollama), pasando por generar keyframes visuales con Flux2 basados en ese storyboard (Flux2KeyframeGenerator).

---

## 🎯 ¿Qué hace exactamente?

Imagina esto como una mini línea de producción para proyectos audiovisuales:

```
Texto o Audio → Voz sintetizada → Storyboard JSON (IA conversa) → Keyframes Fluj2 (visuales anime/ilustración)
```

### FishSpeechUnifiedBatch | 🐟

Consolida la cadena completa de conversión texto-a-voz en un solo nodo. Antes, necesitabas nodos separados para cargar el modelo, codificar referencias, generar tokens semánticos, decodificar audio y guardar el archivo. Ahora le das texto (por ejemplo desde TelegramSuite o BatchTextFileReader) y listo: te devuelve voz en formato MP3 organizada por carpetas, con controles avanzados pero intuitivos de normalización de volumen, velocidad y tono.

---

### AudioStoryboardDirector | 🎬

El siguiente paso una vez tienes el audio transcrito es transformarlo en una secuencia narrativa visual. El nodo transcribe automáticamente mediante Whisper (con windows de frames configurables) y consulta a tu agente IA local (Ollama) para generar exactamente N escenas en formato JSON listo para producción, con frames start/end, prompts visuales y metadatos de storyboard completos. Además puedes usar nodos accesorios `OllamaOptions` y `OllamaSystemPrompt` para personalizar el comportamiento del agente según el género o estilo deseado sin tocar código.

---

### Flux2KeyframeGenerator | 🖼️

El eslabón final de la línea toma ese JSON del storyboard y una imagen de referencia, cargando los modelos UNet + CLIP + VAE correspondientes de manera segura y eficientes bajo `flux-2-klein-9b.safetensors`. Cada escena del storyboard se convierte en uno o más keyframes visuales (anime/ilustración 2D), aplicando conditioning semántico, guidance con ref_latents, CFGGuider con sampling euler, VAE decoding completo y guardado como PNGs ordenados bajo carpetas.

---

## 📦 Instalación

```bash
cd tu/ComfyUI/custom_nodes/
git clone https://github.com/Meisoftcoltd/ComfyUI-FastBatchGenerator.git
pip install -r ComfyUI-FastBatchGenerator/requirements.txt
```

Reinicia ComfyUI y los nuevos módulos aparecerán bajo las categorías plegables `Meisoft ⚠️/TTS`, `Meisoft ⚠️/TTS/Director` y `Meisoft/Video/Keyframes`.

> **Requisitos previos:** Es necesario contar [ComfyUI-fish-speech](https://github.com/Meisoftcoltd/ComfyUI-fish-speech) instalado en la misma carpeta de *custom_nodes* porque FastBatch reutiliza funciones internas sin duplicar modelos. También puedes configurar un servidor Ollama local para los agentes IA del storyboard, y cargar los pesos de Flux2 (`flux-2-klein-9b`, `qwen_3_8b_fp8mixed` y `flux2-vae`) bajo sus respectivos directorios dentro de ComfyUI como cualquier otro checkpoint UNet o archivo VAE.

---

## 🧩 Nodos disponibles

| Nodo | Categoría | Entrada principal | Salida principal |
|------|-----------|-------------------|------------------|
| FishSpeechUnifiedBatch | Meisoft ⚠️/TTS | text_list, file_names | MP3s organizados por carpeta + signal AUDIO final |
| AudioStoryboardDirector | Meisoft ⚠️/TTS/Director | audio transcrito + configuración Ollama | JSON storyboard completo con N escenas + paths a disco |
| Flux2KeyframeGenerator | Meisoft/Video/Keyframes | JSON del storyboard + imagen referencia | Lote de imágenes PNG bajo carpeta keyframes/ con path visible |

Los nodos accesorios `OllamaOptions` y `OllamaSystemPrompt` se usan principalmente como configuradores genéricos que puedes reutilizar en cualquier flujo donde necesites llamar a un agente conversacional local sin repetir parámetros numéricamente cada vez.

---

## 🔄 Flujo de trabajo típico

1️⃣ Cargas tu texto o recibes audio por Telegram y lo pasas al nodo TTS para generar voz limpia.
2️⃣ Esas voces se convierten en transcripciones segmentadas y se conectan al Director junto con tus opciones de Ollama y el prompt system del storyboard.
3️⃣ El JSON resultante viaja directamente a Flux2KeyframeGenerator junto con una imagen base de personaje para consistencia visual.
4️⃣ Al terminar, tienes un lote completo de keyframes ordenados cronológicamente listos para importar en WanVideo, Pika o cualquier motor interpolador posterior.
5️⃥ (Opcional) Conectar estos outputs a otros custom nodes que manejes como DemucsSeparator, AutoCaptions o Sequential-Batcher según el tipo postproduccion requerida.

---

## ❓ Preguntas frecuentes y solución rápida de problemas

**"Al conectar desde Telegram no separa los archivos correctamente"**  
El parser interno detecta tanto saltos simples como dobles entre líneas. Si revisas el log y ves que un índice corresponde a texto desajustado, comprueba upstream que cada archivo tenga una extensión `.txt` clara; si lo envías como string directo asegúrate de indicar en la entrada *file_names* cuántas secciones hay separadas por comas.

**"Ollama me dice Model not found o unreachable"**  
Los parámetros `ollama_url` y `ollama_model` son configurables desde el propio nodo. El primero apunta por defecto a localhost:11434 y el segundo al tag completo del modelo en tu instancia Ollama local. Puedes verificar la lista disponible ejecutando manualmente `ollama list` o probando conexión con los helpers externos integrados. Si falla durante la escena completa pero quieres continuar, desactiva temporalmente el refinamiento de prompts desde `use_ollama_refinement=False` y continúa igual.

**"Flux2 tarda mucho generando cada keyframe o consumiendo VRAM en pico"**  
El modelo Klein-9B es pesado (~16 GB VRAM mínimo por escena). Puedes reducir la resolución base hasta 1024 o bajando pasos de inferencia a rangos seguros entre 30 y 50. También el cfg_scale puede permanecer en 1.0 si no necesitas contraste visual adicional ya que el propio conditioning guidance maneja parte de esa relación internal del modelo FLUX sin penalizar calidad perceptual notable.

**"Los nombres de los archivos MP3 me salen con caracteres extraños"**  
Todos los stem se sanifican automáticamente para evitar colisiones en la creación de carpetas finales incluyendo sanitizing especial para acentos, espacios y extensiones duplicadas. Si ves sufijos inesperados revisa que tu entrada original no contenga metadatos ocultos generados por otro custom_node antes del proceso batch real.

---

## ☁️ Licencia

Código abierto bajo licencia MIT. Puedes usarlo libremente en proyectos personales o comerciales dentro de tus flujos audiovisuales sin obligación adicional alguna más allá de mantener referencia al autor original como buena práctica colaborativa habitual del ecosistema ComfyUI.

*Hecho con ❤️ para flujos TTS productivos y pipelines visuales avanzados en ComfyUI.*
