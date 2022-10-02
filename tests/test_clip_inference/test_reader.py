import pytest
from clip_retrieval.clip_inference.reader import FilesReader, WebdatasetReader
from clip_retrieval.clip_inference.runner import Sampler
import os

from clip_retrieval.clip_inference.load_clip import load_clip


@pytest.mark.parametrize("file_format", ["files", "webdataset"])
def test_reader(file_format):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    current_folder = os.path.dirname(__file__)
    if file_format == "files":
        input_dataset = current_folder + "/test_images"
    else:
        tar_folder = current_folder + "/test_tars"
        input_dataset = [tar_folder + "/image1.tar", tar_folder + "/image2.tar"]
    batch_size = 2
    num_prepro_workers = 2
    _, preprocess = load_clip(warmup_batch_size=batch_size)

    output_partition_count = 2
    actual_values = []
    for output_partition_id in range(output_partition_count):
        sampler = Sampler(output_partition_id=output_partition_id, output_partition_count=output_partition_count)
        if file_format == "files":
            reader = FilesReader(
                sampler,
                preprocess,
                input_dataset,
                batch_size,
                num_prepro_workers,
                enable_text=False,
                enable_image=True,
                enable_metadata=False,
            )
        elif file_format == "webdataset":
            reader = WebdatasetReader(
                sampler,
                preprocess,
                input_dataset,
                batch_size,
                num_prepro_workers,
                enable_text=False,
                enable_image=True,
                enable_metadata=False,
            )
        vals = [i["image_tensor"].shape[0] for i in reader]
        actual_values.append(vals)

    assert actual_values == [[2, 2], [2, 1]]
