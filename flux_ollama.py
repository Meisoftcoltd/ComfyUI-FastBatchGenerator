"""Ollama helpers para Flux keyframe pipeline."""

import re
from typing import Any, Optional, Tuple

try:
    import ollama
except ImportError:
    ollama = None


def connect_ollama(url: str, model: str) -> Tuple[Optional[Any], Optional[str]]:
    if ollama is None:
        return None, "ollama package no instalado"
    try:
        client = ollama.Client(host=url)
        client.list()
        return client, None
    except Exception as e:
        return None, f"Ollama unreachable: {e}"


def query_agent(
    client: Any,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    options_dict: dict,
    retries: int = 2,
) -> Tuple[str, Optional[str]]:
    last_err: Optional[str] = None
    for attempt in range(max(retries, 1)):
        try:
            resp = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options=options_dict,
                stream=False,
            )
            return resp["message"]["content"].strip(), None
        except Exception as e:
            last_err = str(e)
    return "", last_err


AGENTE_2_SYSTEM_PROMPT = """You are an expert Anime Illustrator and Prompt Engineer for FLUX diffusion models.
Your ONLY task is to take a basic scene description and enhance it into a highly detailed, breathtaking, and dynamic anime prompt.

CRITICAL RULES:
1. CHARACTER ANCHOR: You MUST always include the exact phrase "the girl from the image" to maintain character consistency.
2. CHARACTER DETAILS: Always append her description: young woman, wavy brown hair, blue eyes, gold hoop earrings, blue cloak with yellow stars, rusty orange dress, brown boots.
3. FACE ALWAYS VISIBLE: The characters face MUST always be clearly visible. NEVER generate shots from behind, silhouettes, or completely obscured faces. Use frontal or 3/4 profiles.
4. ART STYLE: Enforce this exact style: Anime style, 2d illustration, flat cel shading, clean lines, solid colors, masterpiece, cinematic lighting, highly detailed.
5. DYNAMIC COMPOSITION: The scene must be eye-catching and ready for video motion. Use dramatic camera angles (low angle, extreme wide shot, dutch angle). Give the character a dynamic pose (crouching, leaning, casting a spell, running). NO stiff standing.

INPUT FORMAT:
You will receive a basic scene description.

OUTPUT FORMAT:
Output ONLY the final, raw prompt string in English. Do not include markdown, explanations, or quotes."""
