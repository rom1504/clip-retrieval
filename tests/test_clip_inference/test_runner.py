from clip_retrieval.clip_inference.logger import LoggerWriter
from clip_retrieval.clip_inference.runner import Runner
from clip_retrieval.clip_inference.reader import FilesReader
from clip_retrieval.clip_inference.mapper import ClipMapper
from clip_retrieval.clip_inference.writer import NumpyWriter
from all_clip import load_clip
import os
import numpy as np
import tempfile


def test_runner():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    output_partition_count = 2
    num_prepro_workers = 8
    batch_size = 8
    current_folder = os.path.dirname(__file__)
    folder = current_folder + "/test_images"

    with tempfile.TemporaryDirectory() as tmpdir:

        def reader_builder(sampler):
            _, preprocess, tokenizer = load_clip(warmup_batch_size=batch_size)
            return FilesReader(
                sampler,
                preprocess,
                tokenizer,
                folder,
                batch_size,
                num_prepro_workers,
                enable_text=False,
                enable_image=True,
                enable_metadata=False,
            )

        def mapper_builder():
            return ClipMapper(
                enable_image=True,
                enable_text=False,
                enable_metadata=False,
                use_mclip=False,
                clip_model="ViT-B/32",
                use_jit=True,
                mclip_model="",
                warmup_batch_size=batch_size,
            )

        def logger_builder(i):
            return LoggerWriter(
                partition_id=i,
                stats_folder=tmpdir + "/stats",
            )

        def writer_builder(i):
            return NumpyWriter(
                partition_id=i,
                output_folder=tmpdir,
                enable_text=False,
                enable_image=True,
                enable_metadata=False,
                output_partition_count=output_partition_count,
            )

        runner = Runner(
            reader_builder=reader_builder,
            mapper_builder=mapper_builder,
            writer_builder=writer_builder,
            logger_builder=logger_builder,
            output_partition_count=output_partition_count,
        )

        runner(0)

        with open(tmpdir + "/img_emb/img_emb_0.npy", "rb") as f:
            image_embs = np.load(f)
            assert image_embs.shape[0] == 4
