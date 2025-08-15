"""clip retrieval"""

# Fix PyTorch 2.6+ compatibility issue with CLIP model loading
import torch
from all_clip import load_clip as _original_load_clip
import all_clip

from .clip_back import clip_back
from .clip_filter import clip_filter
from .clip_index import clip_index
from .clip_inference.main import main as clip_inference

# from .clip_inference import clip_inference
from .clip_end2end import clip_end2end
from .clip_front import clip_front

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False for CLIP model compatibility with TorchScript archives
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def _patched_load_clip(clip_model="ViT-B/32", use_jit=True, device=None, **kwargs):
    # Automatically set use_jit=False for OpenAI CLIP models to avoid TorchScript issues
    if not clip_model.startswith(("open_clip:", "hf_clip:", "nm:", "ja_clip:")):
        # This is an OpenAI CLIP model (no prefix or openai_clip: prefix)
        use_jit = False
    return _original_load_clip(clip_model=clip_model, use_jit=use_jit, device=device, **kwargs)


# Replace load_clip in all_clip module
all_clip.load_clip = _patched_load_clip
