import os
import numpy as np
import tempfile
import pytest

from clip_retrieval.clip_inference.distributor import SequentialDistributor, PysparkDistributor


@pytest.mark.parametrize("distributor_kind", ["sequential", "pyspark"])
def test_distributor(distributor_kind):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    with tempfile.TemporaryDirectory() as tmpdir:
        current_folder = os.path.dirname(__file__)
        input_dataset = os.path.join(current_folder, "test_images")

        worker_args = {
            "input_dataset": input_dataset,
            "output_folder": tmpdir,
            "output_partition_count": 2,
            "num_prepro_workers": 6,
            "batch_size": 2,
            "enable_text": False,
            "enable_image": True,
            "enable_metadata": False,
        }

        tasks = [0, 1]

        if distributor_kind == "sequential":
            distributor = SequentialDistributor(tasks=tasks, worker_args=worker_args)
        elif distributor_kind == "pyspark":
            from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

            spark = (
                SparkSession.builder.config("spark.driver.memory", "16G")
                .master("local[" + str(2) + "]")
                .appName("spark-stats")
                .getOrCreate()
            )

            distributor = PysparkDistributor(tasks=tasks, worker_args=worker_args)

        distributor()

        with open(os.path.join(tmpdir, "img_emb/img_emb_0.npy"), "rb") as f:
            image_embs = np.load(f)
            assert image_embs.shape[0] == 4

        with open(os.path.join(tmpdir, "img_emb/img_emb_1.npy"), "rb") as f:
            image_embs = np.load(f)
            assert image_embs.shape[0] == 3
