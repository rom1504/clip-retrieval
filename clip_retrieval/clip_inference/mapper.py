"""mapper module transform images and text to embeddings"""

import torch
from .load_clip import load_clip
from sentence_transformers import SentenceTransformer


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class ClipMapper:
    """transforms images and texts into clip embeddings"""

    def __init__(self, enable_image, enable_text, enable_metadata, use_mclip, clip_model, use_jit, mclip_model):
        self.enable_image = enable_image
        self.enable_text = enable_text
        self.enable_metadata = enable_metadata
        self.use_mclip = use_mclip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = load_clip(clip_model, use_jit)
        self.model_img = model.encode_image
        self.model_txt = model.encode_text
        if use_mclip:
            print("\nLoading MCLIP model for text embedding\n")
            mclip = SentenceTransformer(mclip_model)
            self.model_txt = mclip.encode

    def __call__(self, item):
        with torch.no_grad():
            image_embs = None
            text_embs = None
            image_filename = None
            text = None
            metadata = None
            if self.enable_image:
                image_features = self.model_img(item["image_tensor"].to(self.device))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_embs = image_features.cpu().numpy()
                image_filename = item["image_filename"]
            if self.enable_text:
                if self.use_mclip:
                    text_embs = normalized(self.model_txt(item["text"]))
                else:
                    text_features = self.model_txt(item["text_tokens"].to(self.device))
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_embs = text_features.cpu().numpy()
                text = item["text"]
            if self.enable_metadata:
                metadata = item["metadata"]

            return {
                "image_embs": image_embs,
                "text_embs": text_embs,
                "image_filename": image_filename,
                "text": text,
                "metadata": metadata,
            }
