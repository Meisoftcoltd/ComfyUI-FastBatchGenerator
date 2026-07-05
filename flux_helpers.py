"""Flux helpers: rutas, archivos, utils para keyframe generation."""

import pathlib
import folder_paths


def sanitize_stem(file_name: str) -> str:
    stem = pathlib.Path(str(file_name)).stem
    for ch in ["<", ">", ":", '"', "/", "\\", "|", "?", "*", " ", "."]:
        stem = stem.replace(ch, "_")
    return str(stem) if stem else "unnamed_project"


def get_output_dir(file_name: str) -> pathlib.Path:
    base = pathlib.Path(folder_paths.get_output_directory())
    return base / sanitize_stem(file_name)


def resolve_aspect(w_h: str, base_res: int) -> tuple[int, int]:
    """Devuelve (width, height) alineado a múltiplo de 16."""
    ratios: dict[str, tuple[int, int]] = {
        "9:16": (9, 16),
        "16:9": (16, 9),
        "1:1": (1, 1),
        "4:3": (4, 3),
        "3:4": (3, 4),
    }
    w_r, h_r = ratios.get(w_h, (9, 16))
    gcd_val = __import__("math").gcd(w_r, h_r)
    wn = w_r // gcd_val
    hn = h_r // gcd_val
    scale = base_res / max(wn, hn)
    width = int(round(wn * scale))
    height = int(round(hn * scale))
    # Alinear a múltiplo de 16 para Flux latent (patch_size=2)
    width = (width // 16) * 16
    height = (height // 16) * 16
    return width, height


def save_image(image_tensor, path: str) -> None:
    """Guardar imagen ComfyUI [B,H,W,C] float32 en rango [0,1] → PNG."""
    from PIL import Image
    img_np = image_tensor.squeeze().cpu().numpy()
    if img_np.dtype != float:
        img_np = img_np.astype(float) / 255.0
    rgb = img_np[:, :, :3].clip(0, 1)
    im = Image.fromarray((rgb * 255).astype("uint8"), mode="RGB")
    im.save(path)
