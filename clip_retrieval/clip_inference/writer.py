"""writer module saves embeddings"""

import fsspec
from io import BytesIO
import json
import math


class OutputSink:
    """This output sink can save image, text embeddings as npy and metadata as parquet"""

    def __init__(self, output_folder, enable_text, enable_image, enable_metadata, partition_id, output_partition_count):
        self.enable_text = enable_text
        self.enable_image = enable_image
        self.enable_metadata = enable_metadata
        self.fs, output_folder = fsspec.core.url_to_fs(output_folder)
        self.output_folder = output_folder
        self.img_emb_folder = output_folder + "/img_emb"
        self.text_emb_folder = output_folder + "/text_emb"
        self.metadata_folder = output_folder + "/metadata"
        self.batch_num = partition_id
        self.oom_partition_count = int(math.log10(output_partition_count)) + 1

        if enable_image:
            self.fs.makedirs(self.img_emb_folder, exist_ok=True)

        if enable_text:
            self.fs.makedirs(self.text_emb_folder, exist_ok=True)

        self.fs.makedirs(self.metadata_folder, exist_ok=True)

        self.batch_count = 0
        self.__init_batch()

    def __init_batch(self):
        self.image_embeddings = []
        self.text_embeddings = []
        self.image_names = []
        self.captions = []
        self.metadata = []
        self.batch_count = 0

    def add(self, sample):
        """
        add to buffers the image embeddings, text embeddings, and meta
        """

        self.batch_count += sample["image_embs"].shape[0] if self.enable_image else sample["text_embs"].shape[0]
        if self.enable_image:
            self.image_embeddings.append(sample["image_embs"])
            self.image_names.extend(sample["image_filename"])
        if self.enable_text:
            self.captions.extend(sample["text"])
            self.text_embeddings.append(sample["text_embs"])
        if self.enable_metadata:
            self.metadata.extend(sample["metadata"])

    def __write_batch(self):
        """
        write a batch of embeddings and meta to npy and parquet
        """
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pandas as pd  # pylint: disable=import-outside-toplevel

        data_lists = []
        data_columns = []
        batch_num_str = str(self.batch_num).zfill(self.oom_partition_count)
        if self.enable_image:
            img_emb_mat = np.concatenate(self.image_embeddings)
            output_path_img = self.img_emb_folder + "/img_emb_" + batch_num_str

            with self.fs.open(output_path_img + ".npy", "wb") as f:
                npb = BytesIO()
                np.save(npb, img_emb_mat)
                f.write(npb.getbuffer())

            data_lists.append(self.image_names)
            data_columns.append("image_path")

        if self.enable_text:
            text_emb_mat = np.concatenate(self.text_embeddings)
            output_path_text = self.text_emb_folder + "/text_emb_" + batch_num_str

            with self.fs.open(output_path_text + ".npy", "wb") as f:
                npb = BytesIO()
                np.save(npb, text_emb_mat)
                f.write(npb.getbuffer())

            data_lists.append(self.captions)
            data_columns.append("caption")

        if self.enable_metadata:
            data_lists.append(self.metadata)
            data_columns.append("metadata")

        df = pd.DataFrame(data=list(zip(*data_lists)), columns=data_columns)
        if self.enable_metadata:
            parsed_metadata = pd.json_normalize(df["metadata"].apply(json.loads))
            without_existing_columns = parsed_metadata.drop(
                columns=set(["caption", "metadata", "image_path"]) & set(parsed_metadata.keys())
            )
            df = df.join(without_existing_columns).drop(columns=["metadata"])

        output_path_metadata = self.metadata_folder + "/metadata_" + batch_num_str + ".parquet"
        with self.fs.open(output_path_metadata, "wb") as f:
            df.to_parquet(f)

    def flush(self):
        if self.batch_count == 0:
            return
        self.__write_batch()
        self.__init_batch()


class NumpyWriter:
    """the numpy writer writes embeddings to folders img_emb, text_emb, and metadata"""

    def __init__(self, partition_id, output_folder, enable_text, enable_image, enable_metadata, output_partition_count):
        self.sink = OutputSink(
            output_folder, enable_text, enable_image, enable_metadata, partition_id, output_partition_count
        )

    def __call__(self, batch):
        self.sink.add(batch)

    def flush(self):
        self.sink.flush()
