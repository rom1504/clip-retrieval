import os
import numpy as np

import tempfile
from clip_retrieval.clip_inference.main import main


def test_main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    current_folder = os.path.dirname(__file__)
    input_dataset = current_folder + "/test_images"

    with tempfile.TemporaryDirectory() as tmpdir:
        from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

        spark = (
            SparkSession.builder.config("spark.driver.memory", "16G")
            .master("local[" + str(2) + "]")
            .appName("spark-stats")
            .getOrCreate()
        )

        main(
            input_dataset,
            output_folder=tmpdir,
            input_format="files",
            cache_path=None,
            batch_size=8,
            num_prepro_workers=8,
            enable_text=False,
            enable_image=True,
            enable_metadata=False,
            write_batch_size=4,
            wds_image_key="jpg",
            wds_caption_key="txt",
            clip_model="ViT-B/32",
            mclip_model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
            use_mclip=False,
            use_jit=True,
            distribution_strategy="pyspark",
            wds_number_file_per_input_file=10000,
            output_partition_count=None,
        )

        with open(tmpdir + "/img_emb/img_emb_0.npy", "rb") as f:
            image_embs = np.load(f)
            assert image_embs.shape[0] == 4
        with open(tmpdir + "/img_emb/img_emb_1.npy", "rb") as f:
            image_embs = np.load(f)
            assert image_embs.shape[0] == 3
