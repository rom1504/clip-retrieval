from clip_retrieval.clip_inference.writer import NumpyWriter
import numpy as np
import pickle
import tempfile
import os


def test_writer():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = NumpyWriter(
            partition_id=0,
            output_folder=tmpdir,
            enable_text=False,
            enable_image=True,
            enable_metadata=False,
            output_partition_count=1,
        )
        current_folder = os.path.dirname(__file__)
        embedding_files = [i for i in os.listdir(current_folder + "/test_embeddings")]
        expected_shape = 0
        for embedding_file in embedding_files:
            with open(current_folder + "/test_embeddings/{}".format(embedding_file), "rb") as f:
                embedding = pickle.load(f)
                expected_shape += embedding["image_embs"].shape[0]
                writer(embedding)
        writer.flush()

        with open(tmpdir + "/img_emb/img_emb_0.npy", "rb") as f:
            image_embs = np.load(f)
            assert image_embs.shape[0] == expected_shape
