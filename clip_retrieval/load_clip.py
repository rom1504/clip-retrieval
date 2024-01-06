"""load clip"""
from functools import lru_cache
from torch import autocast, nn
import torch
import clip
from PIL import Image
import time
import numpy as np


class HFClipWrapper(nn.Module):
    """
    Wrap Huggingface ClipModel
    """

    def __init__(self, inner_model, device):
        super().__init__()
        self.inner_model = inner_model
        self.device = torch.device(device=device)
        if self.device.type == "cpu":
            self.dtype = torch.float32
        else:
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def encode_image(self, image):
        if self.device.type == "cpu":
            return self.inner_model.get_image_features(image.squeeze(1))
        with autocast(device_type=self.device.type, dtype=self.dtype):
            return self.inner_model.get_image_features(image.squeeze(1))

    def encode_text(self, text):
        if self.device.type == "cpu":
            return self.inner_model.get_text_features(text)
        with autocast(device_type=self.device.type, dtype=self.dtype):
            return self.inner_model.get_text_features(text)

    def forward(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)


class OpenClipWrapper(nn.Module):
    """
    Wrap OpenClip for managing input types
    """

    def __init__(self, inner_model, device):
        super().__init__()
        self.inner_model = inner_model
        self.device = torch.device(device=device)
        if self.device.type == "cpu":
            self.dtype = torch.float32
        else:
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def encode_image(self, image):
        if self.device.type == "cpu":
            return self.inner_model.encode_image(image)
        with autocast(device_type=self.device.type, dtype=self.dtype):
            return self.inner_model.encode_image(image)

    def encode_text(self, text):
        if self.device.type == "cpu":
            return self.inner_model.encode_text(text)
        with autocast(device_type=self.device.type, dtype=self.dtype):
            return self.inner_model.encode_text(text)

    def forward(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)


def load_hf_clip(clip_model, device="cuda"):
    """load hf clip"""
    from transformers import CLIPProcessor, CLIPModel  # pylint: disable=import-outside-toplevel

    model = CLIPModel.from_pretrained(clip_model)
    preprocess = CLIPProcessor.from_pretrained(clip_model).image_processor
    model = HFClipWrapper(inner_model=model, device=device)
    model.to(device=device)
    return model, lambda x: preprocess(x, return_tensors="pt").pixel_values


def load_open_clip(clip_model, use_jit=True, device="cuda", clip_cache_path=None):
    """load open clip"""

    import open_clip  # pylint: disable=import-outside-toplevel

    torch.backends.cuda.matmul.allow_tf32 = True

    pretrained = dict(open_clip.list_pretrained())
    checkpoint = pretrained[clip_model]
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model, pretrained=checkpoint, device=device, jit=use_jit, cache_dir=clip_cache_path
    )
    model = OpenClipWrapper(inner_model=model, device=device)
    model.to(device=device)
    return model, preprocess


class DeepSparseWrapper(nn.Module):
    """
    Wrap DeepSparse for managing input types
    """

    def __init__(self, model_path):
        super().__init__()

        import deepsparse  # pylint: disable=import-outside-toplevel

        ##### Fix for two-input models
        from deepsparse.clip import CLIPTextPipeline  # pylint: disable=import-outside-toplevel

        def custom_process_inputs(self, inputs):
            if not isinstance(inputs.text, list):
                # Always wrap in a list
                inputs.text = [inputs.text]
            if not isinstance(inputs.text[0], str):
                # If not a string, assume it's already been tokenized
                tokens = np.stack(inputs.text, axis=0, dtype=np.int32)
                return [tokens, np.array(tokens.shape[0] * [tokens.shape[1] - 1])]
            else:
                tokens = [np.array(t).astype(np.int32) for t in self.tokenizer(inputs.text)]
                tokens = np.stack(tokens, axis=0)
                return [tokens, np.array(tokens.shape[0] * [tokens.shape[1] - 1])]

        # This overrides the process_inputs function globally for all CLIPTextPipeline classes
        CLIPTextPipeline.process_inputs = custom_process_inputs
        ####

        self.textual_model_path = model_path + "/textual.onnx"
        self.visual_model_path = model_path + "/visual.onnx"

        self.textual_model = deepsparse.Pipeline.create(task="clip_text", model_path=self.textual_model_path)
        self.visual_model = deepsparse.Pipeline.create(task="clip_visual", model_path=self.visual_model_path)

    def encode_image(self, image):
        image = [np.array(image)]
        embeddings = self.visual_model(images=image).image_embeddings[0]
        return torch.from_numpy(embeddings)

    def encode_text(self, text):
        text = [t.numpy() for t in text]
        embeddings = self.textual_model(text=text).text_embeddings[0]
        return torch.from_numpy(embeddings)

    def forward(self, *args, **kwargs):  # pylint: disable=unused-argument
        return NotImplemented


def load_deepsparse(clip_model):
    """load deepsparse"""

    from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel

    # Download the model from HF
    model_folder = snapshot_download(repo_id=clip_model)
    # Compile the model with DeepSparse
    model = DeepSparseWrapper(model_path=model_folder)

    from deepsparse.clip.constants import CLIP_RGB_MEANS, CLIP_RGB_STDS  # pylint: disable=import-outside-toplevel

    def process_image(image):
        image = model.visual_model._preprocess_transforms(image.convert("RGB"))  # pylint: disable=protected-access
        image_array = np.array(image)
        image_array = image_array.transpose(2, 0, 1).astype("float32")
        image_array /= 255.0
        image_array = (image_array - np.array(CLIP_RGB_MEANS).reshape((3, 1, 1))) / np.array(CLIP_RGB_STDS).reshape(
            (3, 1, 1)
        )
        return torch.from_numpy(np.ascontiguousarray(image_array, dtype=np.float32))

    return model, process_image


@lru_cache(maxsize=None)
def get_tokenizer(clip_model):
    """Load clip"""
    if clip_model.startswith("open_clip:"):
        import open_clip  # pylint: disable=import-outside-toplevel

        clip_model = clip_model[len("open_clip:") :]
        return open_clip.get_tokenizer(clip_model)
    else:
        return lambda t: clip.tokenize(t, truncate=True)


@lru_cache(maxsize=None)
def load_clip_without_warmup(clip_model, use_jit, device, clip_cache_path):
    """Load clip"""
    if clip_model.startswith("open_clip:"):
        clip_model = clip_model[len("open_clip:") :]
        model, preprocess = load_open_clip(clip_model, use_jit, device, clip_cache_path)
    elif clip_model.startswith("hf_clip:"):
        clip_model = clip_model[len("hf_clip:") :]
        model, preprocess = load_hf_clip(clip_model, device)
    elif clip_model.startswith("nm:"):
        clip_model = clip_model[len("nm:") :]
        model, preprocess = load_deepsparse(clip_model)
    else:
        model, preprocess = clip.load(clip_model, device=device, jit=use_jit, download_root=clip_cache_path)
    return model, preprocess


@lru_cache(maxsize=None)
def load_clip(clip_model="ViT-B/32", use_jit=True, warmup_batch_size=1, clip_cache_path=None, device=None):
    """Load clip then warmup"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip_without_warmup(clip_model, use_jit, device, clip_cache_path)

    start = time.time()
    print(f"warming up with batch size {warmup_batch_size} on {device}", flush=True)
    warmup(warmup_batch_size, device, preprocess, model)
    duration = time.time() - start
    print(f"done warming up in {duration}s", flush=True)
    return model, preprocess


def warmup(batch_size, device, preprocess, model):
    fake_img = Image.new("RGB", (224, 224), color="red")
    fake_text = ["fake"] * batch_size
    image_tensor = torch.cat([torch.unsqueeze(preprocess(fake_img), 0)] * batch_size).to(device)
    text_tokens = clip.tokenize(fake_text).to(device)
    for _ in range(2):
        with torch.no_grad():
            model.encode_image(image_tensor)
            model.encode_text(text_tokens)
