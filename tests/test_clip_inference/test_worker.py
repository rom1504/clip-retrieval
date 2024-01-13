import os
import numpy as np

import tempfile
from clip_retrieval.clip_inference.worker import worker


def test_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    current_folder = os.path.dirname(__file__)
    input_dataset = os.path.join(current_folder, "test_images")

    with tempfile.TemporaryDirectory() as tmpdir:
        worker(
            tasks=[0, 1],
            input_dataset=input_dataset,
            output_folder=tmpdir,
            input_format="files",
            output_partition_count=2,
            cache_path=None,
            batch_size=2,
            num_prepro_workers=6,
            enable_text=False,
            enable_image=True,
            enable_metadata=False,
            wds_image_key="jpg",
            wds_caption_key="txt",
            clip_model="ViT-B/32",
            mclip_model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
            use_mclip=False,
            use_jit=True,
            clip_cache_path=None,
        )

        with open(tmpdir + "/img_emb/img_emb_0.npy", "rb") as f:
            image_embs = np.load(f)
            assert image_embs.shape[0] == 4

        with open(tmpdir + "/img_emb/img_emb_1.npy", "rb") as f:
            image_embs = np.load(f)
            assert image_embs.shape[0] == 3
