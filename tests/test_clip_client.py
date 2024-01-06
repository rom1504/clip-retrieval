"""Test the ClipClient class."""
import logging
import pytest

LOGGER = logging.getLogger(__name__)
LOGGER.info("Test ClipClient.query()")
from clip_retrieval.clip_client import ClipClient, Modality

test_url = "https://placekitten.com/400/600"
test_caption = "an image of a cat"
test_image_1 = "tests/test_clip_inference/test_images/123_456.jpg"
test_image_2 = "tests/test_clip_inference/test_images/416_264.jpg"

knn_service_url = "https://knn.laion.ai/knn-service"


# NOTE: This test may fail if the backend is down.
@pytest.mark.skip(reason="temporarily skipping this test while laion knn is down")
def test_query():
    """
    Test the ClipClient.query() method.
    """
    # Create a client
    client = ClipClient(
        url=knn_service_url,
        indice_name="laion5B-L-14",
        use_mclip=False,
        aesthetic_score=9,
        aesthetic_weight=0.5,
        modality=Modality.IMAGE,
        num_images=40,
    )

    # test search via text
    text_search_results = client.query(text=test_caption)
    assert len(text_search_results) > 0, "No results found"
    assert "url" in text_search_results[0], "url not found in search results"
    assert "caption" in text_search_results[0], "caption not found in search results"
    assert "similarity" in text_search_results[0], "similarity not found in search results"
    assert "id" in text_search_results[0], "id not found in search results"
    LOGGER.info(f"{len(text_search_results)} results found")
    LOGGER.info(text_search_results[0])

    # test search via image
    image_search_results = client.query(image=test_image_1)
    assert len(image_search_results) > 0, "No results found"
    assert "url" in image_search_results[0], "url not found in search results"
    assert "caption" in image_search_results[0], "caption not found in search results"
    assert "similarity" in image_search_results[0], "similarity not found in search results"
    assert "id" in image_search_results[0], "id not found in search results"
    LOGGER.info(f"{len(image_search_results)} results found")
    LOGGER.info(image_search_results[0])

    # test search via url of image
    image_url_search_results = client.query(image=test_url)
    assert len(image_url_search_results) > 0, "No results found"
    assert "url" in image_url_search_results[0], "url not found in search results"
    assert "caption" in image_url_search_results[0], "caption not found in search results"
    assert "similarity" in image_url_search_results[0], "similarity not found in search results"
    assert "id" in image_url_search_results[0], "id not found in search results"
    LOGGER.info(f"{len(image_url_search_results)} results found")
    LOGGER.info(image_url_search_results[0])
