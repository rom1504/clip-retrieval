"""main module combines distributor, runner, reader, mapper, writer to produce clip embeddings"""

import fire
import math
from braceexpand import braceexpand

from clip_retrieval.clip_inference.logger import LoggerReader
from clip_retrieval.clip_inference.reader import folder_to_keys
from clip_retrieval.clip_inference.slurm_distributor import SlurmDistributor
from clip_retrieval.clip_inference.distributor import PysparkDistributor, SequentialDistributor


def calculate_partition_count(
    input_format,
    input_dataset,
    enable_image,
    enable_text,
    enable_metadata,
    write_batch_size,
    wds_number_file_per_input_file,
):
    """
    Calculate the partition count needed to store the resulting embeddings.

    Return:
        - the output partition count and the updated toggles for image, text and metadata.
    """

    sample_count = 0

    if input_format == "files":
        keys, text_files, image_files, metadata_files = folder_to_keys(
            input_dataset,
            enable_text=enable_text,
            enable_image=enable_image,
            enable_metadata=enable_metadata,
        )
        if text_files is None or len(text_files) == 0:
            enable_text = False
        if image_files is None or len(image_files) == 0:
            enable_image = False
        if metadata_files is None or len(metadata_files) == 0:
            enable_metadata = False
        if not enable_text and not enable_image and not enable_metadata:
            raise ValueError("no sample found")
        keys, text_files, image_files, metadata_files = folder_to_keys(
            input_dataset,
            enable_text=enable_text,
            enable_image=enable_image,
            enable_metadata=enable_metadata,
        )
        sample_count = len(keys)
    elif input_format == "webdataset":
        sample_count = len(input_dataset) * wds_number_file_per_input_file
    else:
        raise ValueError(f"Unsupported input_format {input_format}")

    if sample_count == 0:
        raise ValueError("no sample found")

    print(f"The number of samples has been estimated to be {sample_count}")

    output_partition_count = math.ceil(sample_count / write_batch_size)

    return output_partition_count, enable_text, enable_image, enable_metadata


# pylint: disable=unused-argument
def main(
    input_dataset,
    output_folder,
    input_format="files",
    cache_path=None,
    batch_size=256,
    num_prepro_workers=4,
    enable_text=True,
    enable_image=True,
    enable_metadata=False,
    write_batch_size=10**6,
    wds_image_key="jpg",
    wds_caption_key="txt",
    clip_model="ViT-B/32",
    mclip_model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
    use_mclip=False,
    use_jit=False,
    distribution_strategy="sequential",
    wds_number_file_per_input_file=10000,
    output_partition_count=None,
    wandb_project="clip_retrieval",
    enable_wandb=False,
    clip_cache_path=None,
    slurm_job_name=None,
    slurm_partition=None,
    slurm_nodes=None,
    slurm_job_comment=None,
    slurm_nodelist=None,
    slurm_exclude=None,
    slurm_job_timeout=None,
    slurm_cache_path=None,
    slurm_verbose_wait=False,
):
    # package arguments to pass on to the distributor
    local_args = dict(locals())

    expanded_dataset = list(braceexpand(input_dataset)) if input_format == "webdataset" else input_dataset

    # compute this now for the distributors to use
    if output_partition_count is None:
        output_partition_count, enable_text, enable_image, enable_metadata = calculate_partition_count(
            input_format=input_format,
            input_dataset=expanded_dataset,
            enable_image=enable_image,
            enable_text=enable_text,
            enable_metadata=enable_metadata,
            write_batch_size=write_batch_size,
            wds_number_file_per_input_file=wds_number_file_per_input_file,
        )

        # update the local args to match the computed values
        local_args["output_partition_count"] = output_partition_count
        local_args["enable_text"] = enable_text
        local_args["enable_image"] = enable_image
        local_args["enable_metadata"] = enable_metadata

    local_args.pop("wds_number_file_per_input_file")
    local_args.pop("write_batch_size")
    local_args.pop("distribution_strategy")
    local_args.pop("wandb_project")
    local_args.pop("enable_wandb")

    tasks = list(range(output_partition_count))
    worker_args = {k: v for k, v in local_args.items() if not k.startswith("slurm_")}

    if distribution_strategy == "sequential":
        distributor = SequentialDistributor(tasks=tasks, worker_args=worker_args)
    elif distribution_strategy == "pyspark":
        distributor = PysparkDistributor(tasks=tasks, worker_args=worker_args)
    elif distribution_strategy == "slurm":
        slurm_args = {k.lstrip("slurm_"): v for k, v in local_args.items() if k.startswith("slurm_")}
        distributor = SlurmDistributor(tasks=tasks, worker_args=worker_args, slurm_args=slurm_args)
    else:
        print(
            f"The {distribution_strategy} strategy is not implemented. Please choose from: [sequential, pyspark, slurm]"
        )

    logger_reader = LoggerReader(
        stats_folder=output_folder + "/stats",
        wandb_project=wandb_project,
        enable_wandb=enable_wandb,
    )

    logger_reader.start()

    distributor()

    logger_reader.end()


if __name__ == "__main__":
    fire.Fire(main)
