"""load clip"""

from functools import lru_cache
import torch
import clip
import open_clip

@lru_cache(maxsize=None)
def load_clip(clip_model="ViT-B/32", use_jit=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if clip_model.startswith("openclip:"):
        clip_model = clip_model[len("openclip:"):]
        model, _, preprocess = open_clip.create_model_and_transforms(clip_model,
                                                                     pretrained='laion400m_e32',
                                                                     device=device)
    else:
        model, preprocess = clip.load(clip_model, device=device, jit=use_jit)
    return model, preprocess
