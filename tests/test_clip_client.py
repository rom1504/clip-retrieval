"""Test the ClipClient class."""
from pathlib import Path
import tempfile
from clip_retrieval.clip_client import ClipClient, Modality

test_url = "https://placekitten.com/400/600"
test_caption = "an image of a cat"
test_image_1 = "tests/test_clip_inference/test_images/123_456.jpg"
test_image_2 = "tests/test_clip_inference/test_images/416_264.jpg"

knn_service_url = "https://knn5.laion.ai/knn-service"


def test_query():
    """
    Test the ClipClient.query() method.
    """
    # Create a temporary directory to store the results
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a client
        client = ClipClient(
            url=knn_service_url,
            result_dir=Path(tmp_dir).joinpath("results"),
            indice_name="laion5B",
            use_mclip=False,
            aesthetic_score=9,
            aesthetic_weight=0.5,
            modality=Modality.IMAGE,
            num_images=40,
            num_result_ids=3000,
        )

        # test search via text
        text_search_results = client.query(text=test_caption)
        # Note, this test is highly non-deterministic
        assert len(text_search_results) > 0, "No results found"

        # test search via image
        image_search_results = client.query(image=test_image_1)
        # Note, this test is highly non-deterministic
        assert len(image_search_results) > 0, "No results found"

        # test search via url of image
        image_url_search_results = client.query(image=test_url)
        # Note, this test is highly non-deterministic
        assert len(image_url_search_results) > 0, "No results found"
