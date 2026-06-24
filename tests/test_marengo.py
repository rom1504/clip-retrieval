"""Test the TwelveLabs Marengo embedding backend."""

import os
import numpy as np
import pytest

from clip_retrieval.marengo import (
    MARENGO_DIMENSIONS,
    MarengoModel,
    is_marengo_model,
    load_marengo,
    marengo_model_name,
)


def test_is_marengo_model():
    assert is_marengo_model("twelvelabs:marengo3.0")
    assert is_marengo_model("twelvelabs:")
    assert not is_marengo_model("ViT-B/32")
    assert not is_marengo_model("open_clip:ViT-H-14")
    assert not is_marengo_model(None)


def test_marengo_model_name():
    assert marengo_model_name("twelvelabs:marengo3.0") == "marengo3.0"
    assert marengo_model_name("twelvelabs:") == "marengo3.0"  # falls back to default


class _FakeSegment:
    def __init__(self, vector):
        self.float_ = vector


class _FakeEmbedding:
    def __init__(self, vector):
        self.segments = [_FakeSegment(vector)]


class _FakeResponse:
    def __init__(self, text=None, image=None):
        self.text_embedding = _FakeEmbedding(text) if text is not None else None
        self.image_embedding = _FakeEmbedding(image) if image is not None else None


class _FakeEmbed:
    def create(self, model_name, text=None, image_file=None):  # noqa: ARG002
        raw = [3.0, 4.0] + [0.0] * (MARENGO_DIMENSIONS - 2)
        if text is not None:
            return _FakeResponse(text=raw)
        return _FakeResponse(image=raw)


class _FakeClient:
    def __init__(self):
        self.embed = _FakeEmbed()


def _model_with_fake_client():
    model = MarengoModel.__new__(MarengoModel)
    model.model_name = "marengo3.0"
    model._client = _FakeClient()  # pylint: disable=protected-access
    return model


def test_encode_text_shape_and_normalization():
    model = _model_with_fake_client()
    out = model.encode_text("a cat")
    assert out.shape == (1, MARENGO_DIMENSIONS)
    assert out.dtype == np.float32
    # 3-4-0... normalized to unit length -> 0.6, 0.8
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-5)
    np.testing.assert_allclose(out[0, :2], [0.6, 0.8], atol=1e-5)


def test_encode_image_shape():
    model = _model_with_fake_client()
    out = model.encode_image(b"fake-image-bytes")
    assert out.shape == (1, MARENGO_DIMENSIONS)
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-5)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("TWELVELABS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TwelveLabs API key"):
        MarengoModel(api_key=None)


@pytest.mark.skipif(
    not os.environ.get("TWELVELABS_API_KEY"),
    reason="requires TWELVELABS_API_KEY for a live Marengo embedding call",
)
def test_live_text_embedding():
    """Live smoke test against the TwelveLabs API (skipped when no key is set)."""
    model = load_marengo("twelvelabs:marengo3.0")
    out = model.encode_text("a cat playing the piano")
    assert out.shape == (1, MARENGO_DIMENSIONS)
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-4)
