"""main module combines distributor, runner, reader, mapper, writer to produce clip embeddings"""

from braceexpand import braceexpand
import fire
from clip_retrieval.clip_inference.load_clip import load_clip
from clip_retrieval.clip_inference.logger import LoggerReader, LoggerWriter
from clip_retrieval.clip_inference.reader import folder_to_keys

from clip_retrieval.clip_inference.mapper import ClipMapper
from clip_retrieval.clip_inference.reader import FilesReader, WebdatasetReader
from clip_retrieval.clip_inference.writer import NumpyWriter
from clip_retrieval.clip_inference.distributor import PysparkDistributor, SequentialDistributor
from clip_retrieval.clip_inference.runner import Runner


def main(
    input_dataset,
    output_folder,
    input_format="files",
    cache_path=None,
    batch_size=256,
    num_prepro_workers=8,
    enable_text=True,
    enable_image=True,
    enable_metadata=False,
    write_batch_size=10 ** 6,
    wds_image_key="jpg",
    wds_caption_key="txt",
    clip_model="ViT-B/32",
    mclip_model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
    use_mclip=False,
    use_jit=True,
    distribution_strategy="sequential",
    wds_number_file_per_input_file=10000,
    output_partition_count=None,
    wandb_project="clip_retrieval",
    enable_wandb=False,
):
    if input_format == "webdataset":
        input_dataset = list(braceexpand(input_dataset))
    if output_partition_count is None:
        if input_format == "files":
            keys, text_files, image_files, metadata_files = folder_to_keys(
                input_dataset, enable_text=enable_text, enable_image=enable_image, enable_metadata=enable_metadata
            )
            if text_files is None or len(text_files) == 0:
                enable_text = False
            if image_files is None or len(image_files) == 0:
                enable_image = False
            if metadata_files is None or len(metadata_files) == 0:
                enable_metadata = False
            keys, text_files, image_files, metadata_files = folder_to_keys(
                input_dataset, enable_text=enable_text, enable_image=enable_image, enable_metadata=enable_metadata
            )
            sample_count = len(keys)
        elif input_format == "webdataset":
            sample_count = len(input_dataset) * wds_number_file_per_input_file

        if sample_count == 0:
            print("no sample found")
            return
        else:
            print("The number of samples has been estimated to be {}".format(sample_count))

        output_partition_count = int(sample_count / write_batch_size) + 1

    def reader_builder(sampler):
        _, preprocess = load_clip(clip_model=clip_model, use_jit=use_jit)
        if input_format == "files":
            return FilesReader(
                sampler,
                preprocess,
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
        return LoggerWriter(partition_id=i, stats_folder=output_folder + "/stats",)

    runner = Runner(
        reader_builder=reader_builder,
        mapper_builder=mapper_builder,
        writer_builder=writer_builder,
        logger_builder=logger_builder,
        output_partition_count=output_partition_count,
    )

    logger_reader = LoggerReader(
        stats_folder=output_folder + "/stats", wandb_project=wandb_project, enable_wandb=enable_wandb
    )
    logger_reader.start()

    if distribution_strategy == "sequential":
        distributor = SequentialDistributor(runner, output_partition_count)
    elif distribution_strategy == "pyspark":
        distributor = PysparkDistributor(runner, output_partition_count)
    distributor()

    logger_reader.end()


if __name__ == "__main__":
    fire.Fire(main)
