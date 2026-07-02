# 🚀 ComfyUI-FastBatchGenerator 🎭✨

**The Ultra-Optimized Emotional Engine for Fish Speech TTS.**

This custom node is designed to transform raw, unformatted text into highly expressive, production-ready audio batches. It specializes in handling complex "musical scores" within text, allowing for precise control over emotions and pauses without breaking the workflow.

---

## 🌟 Key Features

*   🎭 **Emotional Tag Engine:** Supports advanced emotional metadata directly in your text (e.g., `[emphasis]`, `					[soft]`, `[smug]`, `[breathy]`).
*   ⏸️ **Smart Pause Injection:** Recognizes `[long pause]` tags and injects real, audible silence into the audio stream to maintain natural pacing.
*   📂 **Intelligent Project Organization:** Automatically creates a dedicated sub-folder for every batch based on your source filename, keeping your `output/` folder clean and organized.
*   🏷️ **Traceable Output:** Generates `.mp3` files that inherit the exact name of your input file (no random numbers or timestamps!), making it easy for downstream nodes to find and process them.
*   ⚡ **VRAM-Aware Singleton Pattern:** Implements intelligent model loading. It reuses the heavy LLaMA and DAC models in VRAM across batches, drastically reducing loading overhead and preventing memory fragmentation.

---

## 🛠️ Requirements & Dependencies

**⚠️ Prerequisite:** This node is a high-level orchestrator. It **requires** the following custom node to be installed and active in your `custom_nodes/` directory:

1.  **[ComfyUI-fish-speech](https://github.com/path-to-original-repo)** (The core inference engine).
    *   *Note: This node hooks into the `init_model` and `load_model` functions of the FishSpeech implementation.*

**Python Dependencies:**
Ensure your environment has the following installed (included in `requirements.txt`):
*   `torch` & `torodaudio` (for tensor manipulations)
*   `pydub` & `ffmpeg` (for high-quality `.mp3` encoding and silence injection)
*   `numpy` (for waveform processing)

---

## 🚀 Installation

1. Clone this repository into your `ComfyUI/custom_nodes/` folder:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/your-username/ComfyUI-FastBatchGenerator.git
   ```
2. Install the required Python dependencies:
   ```bash
   pip install -r ComfyUI-FastBatchGenerator/requirements.txt
   ```
3. Restart ComfyUI.

---

## 📖 Usage Example

Simply feed your text and a reference audio to the `FishSpeechUnifiedBatch` node.

**Input Text Example:**
```text
[emphasis] This is an intense moment.
[long pause]
[soft] And now, we transition into a whisper.
```

**The Result:**
An `.mp3` file located in `output/fishspeech_batches/[Your_Source_Name]/[Your_Source_Name].mp3`, featuring the exact emotional cadence defined by your tags.

---
*Developed with ❤️ for high-performance TTS workflows.*
