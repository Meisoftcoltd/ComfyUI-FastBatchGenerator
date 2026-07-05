"""Flux2 clip encode - tokenize text to conditioning tensors."""

import torch


def clip_tokenize_encode(clip_obj, text: str) -> list:
    """Text to conditioning. Replica CLIPTextEncode interno.

    Para FLUX type en ComfyUI v0.27, el API correcto es:
      tokenized = clip_obj.tokenize(text)   -> dict/nested tokens
      cond = clip_obj.encode(tokenized)     -> list[conditions]

    Returns list de conditioning (misma estructura que CLIPTextEncode).
    """
    tokenized = clip_obj.tokenize(text)
    pos_cond = clip_obj.encode(tokenized)
    return pos_cond


def clip_encode_zeros(pos_cond: list) -> list:
    """Negative conditioning (all zeros). Replica ConditioningZeroOut."""
    result = []
    for c in pos_cond:
        if isinstance(c, list) and len(c) >= 2:
            part1 = c[0]
            part2 = {}
            for k, v in c[1].items():
                if k in ("guidance", "ref_latents"):
                    # Non-tensor keys se mantienen (o se ponen a None)
                    pass
                else:
                    if hasattr(v, "zeros_like"):
                        part2[k] = v.zeros_like()
                    else:
                        part2[k] = v
            result.append([part1, part2])
        else:
            result.append(c)
    return result
