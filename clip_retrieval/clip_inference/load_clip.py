"""load clip"""

from functools import lru_cache
import torch
import clip


@lru_cache(maxsize=None)
def load_clip(clip_model="ViT-B/32", use_jit=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device=device, jit=use_jit)
    return model, preprocess
