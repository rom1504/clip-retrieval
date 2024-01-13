"""
Inference Worker:

A completely independent process that will be started once for each GPU node.
Distributors will call this either through the CLI or directly.

The worker sequentially process the tasks passed to it.
Tasks are lists of partition_id's that this worker will be responsible for.
"""

import fire
from braceexpand import braceexpand

from clip_retrieval.clip_inference.runner import Runner
from clip_retrieval.clip_inference.mapper import ClipMapper
from clip_retrieval.clip_inference.writer import NumpyWriter
from clip_retrieval.clip_inference.logger import LoggerWriter
from clip_retrieval.clip_inference.reader import FilesReader, WebdatasetReader
from all_clip import load_clip


def worker(
    tasks,
    input_dataset,
    output_folder,
    output_partition_count,
    input_format="files",
    cache_path=None,
    batch_size=256,
    num_prepro_workers=4,
    enable_text=True,
    enable_image=True,
    enable_metadata=False,
    wds_image_key="jpg",
    wds_caption_key="txt",
    clip_model="ViT-B/32",
    mclip_model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
    use_mclip=False,
    use_jit=True,
    clip_cache_path=None,
):
    """Start a worker"""
    print("Starting the worker", flush=True)

    # check for brace expansion
    if input_format == "webdataset" and not isinstance(input_dataset, list):
        input_dataset = list(braceexpand(input_dataset))

    print(f"dataset is {len(input_dataset)}", flush=True)

    def reader_builder(sampler):
        _, preprocess, tokenizer = load_clip(
            clip_model=clip_model,
            use_jit=use_jit,
            warmup_batch_size=batch_size,
            clip_cache_path=clip_cache_path,
        )
        if input_format == "files":
            return FilesReader(
                sampler,
                preprocess,
                tokenizer,
                input_dataset,
                batch_size,
                num_prepro_workers,
                enable_text=enable_text,
                enable_image=enable_image,
                enable_metadata=enable_metadata,
            )
        elif input_format == "webdataset":
            return WebdatasetReader(
                sampler,
                preprocess,
                tokenizer,
                input_dataset,
                batch_size,
                num_prepro_workers,
                enable_text=enable_text,
                enable_image=enable_image,
                enable_metadata=enable_metadata,
                wds_image_key=wds_image_key,
                wds_caption_key=wds_caption_key,
                cache_path=cache_path,
            )
        else:
            raise ValueError(f"Unknown input_format: {input_format}")

    def mapper_builder():
        return ClipMapper(
            enable_image=enable_image,
            enable_text=enable_text,
            enable_metadata=enable_metadata,
            use_mclip=use_mclip,
            clip_model=clip_model,
            use_jit=use_jit,
            mclip_model=mclip_model,
            clip_cache_path=clip_cache_path,
            warmup_batch_size=batch_size,
        )

    def writer_builder(i):
        return NumpyWriter(
            partition_id=i,
            output_folder=output_folder,
            enable_text=enable_text,
            enable_image=enable_image,
            enable_metadata=enable_metadata,
            output_partition_count=output_partition_count,
        )

    def logger_builder(i):
        return LoggerWriter(
            partition_id=i,
            stats_folder=output_folder + "/stats",
        )

    runner = Runner(
        reader_builder=reader_builder,
        mapper_builder=mapper_builder,
        writer_builder=writer_builder,
        logger_builder=logger_builder,
        output_partition_count=output_partition_count,
    )

    for task in tasks:
        print(f"Starting work on task {task}", flush=True)
        runner(task)


if __name__ == "__main__":
    fire.Fire(worker)
