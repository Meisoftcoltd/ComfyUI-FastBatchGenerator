"""Fast Batch Generator — Batch process multiple audio scripts automatically."""
import os, pathlib

def __init__():
    """Constructor para inicializar el nodo. Se llama al cargar la lista de inputs."""
    return {
        "class_type": "ComfyFastBatch",
        "input_types": {
            # Ruta raíz donde se cargan los archivos .txt desde TelegramSuite_BatchTextFileReader
            "text_inputs_path_root": ("STRING", {"default": "input"}),
        },
    }


def generate(text_paths: list[list]) -> pathlib.Path:
    """Bucle interno: para cada archivo del lote, genera un MP3 autónomo.

    Args:
        text_paths (list[list]): Lista de rutas absolutas al texto original.

    Returns:
        pathlib.Path: directorio raíz donde se escribieron los mp3 individuales generados.
    """
    root = pathlib.Path(os.getcwd()) / "input"

    for idx in range(0, len(text_paths)):
        file_name = text_paths[idx]  # ejemplo: "Elara/tts/Tzelem El Secreto del Alma.txt"
        file_path = root / file_name  # ruta absoluta al archivo .txt
        sanitized = "".join(c for c in str(file_path).lstrip("/") if not c.isspace())[:30]  # sanitize

        out_dir = pathlib.PurePosixPath(root, sanitized)  # nuevo dir por input
        out_mp3 = pathlib.PurePosixPath(out_dir, "{}.mp3".format(os.path.splitext(str(file_path.name))[0]))
        out_dir.mkdir(parents=True, exist_ok=True)

        # Aquí se llamaría al FishSpeechDecoder para generar MP3 en output_root/{subfolder}/ (no timestamp).

    # Retorna el directorio de salida consolidado como JSON a ComfyUI.
    return pathlib.Path("." ) / "output" / "fishspeech" / sanitized
