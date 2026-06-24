"""marengo module provides TwelveLabs Marengo embeddings as an opt-in alternative to CLIP

Marengo is a video-native multimodal embedding model from TwelveLabs. It embeds text and
images into the same 512-dim space as the video segment embeddings it produces, which makes
it a drop-in alternative to CLIP when the corpus you index is video (or a mix of video and
images) rather than still images only.

This module is only imported when a model name with the ``twelvelabs:`` prefix is requested
(for example ``twelvelabs:marengo3.0``), so the ``twelvelabs`` SDK stays an optional dependency.
"""

import os

import numpy as np

MARENGO_PREFIX = "twelvelabs:"
DEFAULT_MARENGO_MODEL = "marengo3.0"
# Marengo embeddings are 512-dim float vectors, the same width as ViT-B/32 CLIP embeddings.
MARENGO_DIMENSIONS = 512


def is_marengo_model(clip_model):
    """return True if clip_model selects a TwelveLabs Marengo model"""
    return isinstance(clip_model, str) and clip_model.startswith(MARENGO_PREFIX)


def marengo_model_name(clip_model):
    """extract the TwelveLabs model name from a ``twelvelabs:<name>`` string"""
    name = clip_model[len(MARENGO_PREFIX) :] if is_marengo_model(clip_model) else clip_model
    return name or DEFAULT_MARENGO_MODEL


class MarengoModel:
    """thin wrapper around the TwelveLabs embed API exposing encode_text / encode_image

    Vectors are L2-normalized to match the convention used everywhere else in clip-retrieval
    (faiss inner-product search over unit vectors).
    """

    def __init__(self, model_name=DEFAULT_MARENGO_MODEL, api_key=None):
        from twelvelabs import TwelveLabs  # pylint: disable=import-outside-toplevel

        api_key = api_key or os.environ.get("TWELVELABS_API_KEY")
        if not api_key:
            raise ValueError(
                "TwelveLabs API key not found. Set the TWELVELABS_API_KEY environment variable "
                "or pass api_key=. Get a free key at https://twelvelabs.io"
            )
        self.model_name = model_name
        self._client = TwelveLabs(api_key=api_key)

    def _normalize(self, vector):
        vector = np.asarray(vector, dtype="float32")
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def encode_text(self, text):
        """embed a single text string, returning a (1, 512) float32 numpy array"""
        response = self._client.embed.create(model_name=self.model_name, text=text)
        vector = response.text_embedding.segments[0].float_
        return self._normalize(vector)[np.newaxis, :]

    def encode_image(self, image_bytes):
        """embed a single image (raw bytes), returning a (1, 512) float32 numpy array

        Marengo requires images of at least 128x128 pixels.
        """
        response = self._client.embed.create(model_name=self.model_name, image_file=image_bytes)
        vector = response.image_embedding.segments[0].float_
        return self._normalize(vector)[np.newaxis, :]


def load_marengo(clip_model, api_key=None):
    """load a Marengo model from a ``twelvelabs:<name>`` model string"""
    return MarengoModel(model_name=marengo_model_name(clip_model), api_key=api_key)
