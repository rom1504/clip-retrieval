"""Clip inference is a tool to go from image/text collection to clip embeddings"""

#!pip install clip-anytorch fire
import fire
from PIL import Image
import json
import fsspec
from io import BytesIO
from pathlib import Path
import tqdm
import io


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def get_image_dataset():
    """retrieve image dataset module without importing torch at the top level"""

    from torch.utils.data import Dataset  # pylint: disable=import-outside-toplevel

    class ImageDataset(Dataset):
        """ImageDataset is a pytorch Dataset exposing image and text tensors from a folder of image and text"""

        def __init__(self, preprocess, folder, enable_text=True, enable_image=True, enable_metadata=False):
            super().__init__()
            import clip  # pylint: disable=import-outside-toplevel

            path = Path(folder)
            self.enable_text = enable_text
            self.enable_image = enable_image
            self.enable_metadata = enable_metadata

            if self.enable_text:
                text_files = [*path.glob("**/*.txt")]
                text_files = {text_file.stem: text_file for text_file in text_files}
                if len(text_files) == 0:
                    self.enable_text = False
            if self.enable_image:
                image_files = [
                    *path.glob("**/*.png"),
                    *path.glob("**/*.jpg"),
                    *path.glob("**/*.jpeg"),
                    *path.glob("**/*.bmp"),
                ]
                image_files = {image_file.stem: image_file for image_file in image_files}
                if len(image_files) == 0:
                    self.enable_image = False
            if self.enable_metadata:
                metadata_files = [*path.glob("**/*.json")]
                metadata_files = {metadata_file.stem: metadata_file for metadata_file in metadata_files}
                if len(metadata_files) == 0:
                    self.enable_metadata = False

            keys = None
            join = lambda new_set: new_set & keys if keys is not None else new_set
            if self.enable_text:
                keys = join(text_files.keys())
            elif self.enable_image:
                keys = join(image_files.keys())
            elif self.enable_metadata:
                keys = join(metadata_files.keys())

            self.keys = list(keys)
            if self.enable_text:
                self.tokenizer = lambda text: clip.tokenize([text], truncate=True)[0]
                self.text_files = {k: v for k, v in text_files.items() if k in keys}
            if self.enable_image:
                self.image_files = {k: v for k, v in image_files.items() if k in keys}
                self.image_transform = preprocess
            if self.enable_metadata:
                self.metadata_files = {k: v for k, v in metadata_files.items() if k in keys}

        def __len__(self):
            return len(self.keys)

        def __getitem__(self, ind):
            key = self.keys[ind]
            output = {}

            if self.enable_image:
                image_file = self.image_files[key]
                image_tensor = self.image_transform(Image.open(image_file))
                output["image_filename"] = str(image_file)
                output["image_tensor"] = image_tensor

            if self.enable_text:
                text_file = self.text_files[key]
                caption = text_file.read_text()
                tokenized_text = self.tokenizer(caption)
                output["text_tokens"] = tokenized_text
                output["text"] = caption

            if self.enable_metadata:
                metadata_file = self.metadata_files[key]
                metadata = metadata_file.read_text()
                output["metadata"] = metadata

            return output

    return ImageDataset


def create_webdataset(
    urls,
    image_transform,
    enable_text=True,
    enable_image=True,
    image_key="jpg",
    caption_key="txt",
    enable_metadata=False,
    cache_path=None,
):
    """Create a WebDataset reader, it can read a webdataset of image, text and json"""
    import clip  # pylint: disable=import-outside-toplevel
    import webdataset as wds  # pylint: disable=import-outside-toplevel

    dataset = wds.WebDataset(urls, cache_dir=cache_path, cache_size=10 ** 10, handler=wds.handlers.warn_and_continue)
    tokenizer = lambda text: clip.tokenize([text], truncate=True)[0]

    def filter_dataset(item):
        if enable_text and caption_key not in item:
            return False
        if enable_image and image_key not in item:
            return False
        if enable_metadata and "json" not in item:
            return False
        return True

    filtered_dataset = dataset.select(filter_dataset)

    def preprocess_dataset(item):
        output = {}
        if enable_image:
            image_data = item[image_key]
            image = Image.open(io.BytesIO(image_data))
            image_tensor = image_transform(image)
            output["image_filename"] = item["__key__"]
            output["image_tensor"] = image_tensor

        if enable_text:
            text = item[caption_key]
            caption = text.decode("utf-8")
            tokenized_text = tokenizer(caption)
            output["text_tokens"] = tokenized_text
            output["text"] = caption

        if enable_metadata:
            metadata_file = item["json"]
            metadata = metadata_file.decode("utf-8")
            output["metadata"] = metadata
        return output

    transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
    return transformed_dataset


class OutputSink:
    """This output sink can save image, text embeddings as npy and metadata as parquet"""

    def __init__(self, output_folder, enable_text, enable_image, enable_metadata, write_batch_size):
        self.enable_text = enable_text
        self.enable_image = enable_image
        self.enable_metadata = enable_metadata
        self.fs, output_folder = fsspec.core.url_to_fs(output_folder)
        self.output_folder = output_folder
        self.img_emb_folder = output_folder + "/img_emb"
        self.text_emb_folder = output_folder + "/text_emb"
        self.metadata_folder = output_folder + "/metadata"

        if not self.fs.exists(self.output_folder):
            self.fs.mkdir(self.output_folder)
            batch_init_num = -1
        else:
            existing_top_level_files = self.fs.walk(self.metadata_folder).__next__()[2]
            if len(existing_top_level_files) == 0:
                batch_init_num = -1
            else:
                batch_init_num = max(
                    [int(x.split("/")[-1].split(".")[0].split("_")[1]) for x in existing_top_level_files]
                )
        if enable_image and not self.fs.exists(self.img_emb_folder):
            self.fs.mkdir(self.img_emb_folder)

        if enable_text and not self.fs.exists(self.text_emb_folder):
            self.fs.mkdir(self.text_emb_folder)

        if not self.fs.exists(self.metadata_folder):
            self.fs.mkdir(self.metadata_folder)

        self.write_batch_size = write_batch_size
        self.batch_count = 0
        self.batch_num = batch_init_num
        self.__init_batch()

    def __init_batch(self):
        self.image_embeddings = []
        self.text_embeddings = []
        self.image_names = []
        self.captions = []
        self.metadata = []
        self.batch_count = 0
        self.batch_num += 1

    def add(self, image_embs, text_embs, image_filenames, captions, metadata):
        """
        add to buffers the image embeddings, text embeddings, and meta
        """
        self.batch_count += image_embs.shape[0] if self.enable_image else text_embs.shape[0]
        if self.enable_image:
            self.image_embeddings.append(image_embs)
            self.image_names.extend(image_filenames)
        if self.enable_text:
            self.captions.extend(captions)
            self.text_embeddings.append(text_embs)
        if self.enable_metadata:
            self.metadata.extend(metadata)
        if self.batch_count > self.write_batch_size:
            self.flush()

    def __write_batch(self):
        """
        write a batch of embeddings and meta to npy and parquet
        """
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pandas as pd  # pylint: disable=import-outside-toplevel

        data_lists = []
        data_columns = []
        if self.enable_image:
            img_emb_mat = np.concatenate(self.image_embeddings)
            output_path_img = self.img_emb_folder + "/img_emb_" + str(self.batch_num)

            with self.fs.open(output_path_img + ".npy", "wb") as f:
                npb = BytesIO()
                np.save(npb, img_emb_mat)
                f.write(npb.getbuffer())

            data_lists.append(self.image_names)
            data_columns.append("image_path")

        if self.enable_text:
            text_emb_mat = np.concatenate(self.text_embeddings)
            output_path_text = self.text_emb_folder + "/text_emb_" + str(self.batch_num)

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

        output_path_metadata = self.metadata_folder + "/metadata_" + str(self.batch_num) + ".parquet"
        with self.fs.open(output_path_metadata, "wb") as f:
            df.to_parquet(f)

    def flush(self):
        if self.batch_count == 0:
            return
        self.__write_batch()
        self.__init_batch()


def clip_inference(
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
    subset_size=None,
    wds_image_key="jpg",
    wds_caption_key="txt",
    clip_model="ViT-B/32",
    mclip_model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
    use_mclip=False,
):
    """clip inference goes from a image text dataset to clip embeddings"""

    import clip  # pylint: disable=import-outside-toplevel
    from sentence_transformers import SentenceTransformer  # pylint: disable=import-outside-toplevel
    from torch.utils.data import DataLoader  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device=device, jit=False)
    model_img = model.encode_image
    model_txt = model.encode_text
    if use_mclip:
        print("\nLoading MCLIP model for text embedding\n")
        mclip = SentenceTransformer(mclip_model)
        model_txt = mclip.encode
    if input_format == "files":
        dataset = get_image_dataset()(preprocess, input_dataset, enable_text=enable_text, enable_image=enable_image)
        enable_text = dataset.enable_text
        enable_image = dataset.enable_image
        enable_metadata = dataset.enable_metadata
    elif input_format == "webdataset":
        dataset = create_webdataset(
            input_dataset,
            preprocess,
            enable_text,
            enable_image,
            image_key=wds_image_key,
            caption_key=wds_caption_key,
            enable_metadata=enable_metadata,
            cache_path=cache_path,
        )
    else:
        raise Exception(f"No such input format {input_format}")

    data = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_prepro_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    output_sink = OutputSink(output_folder, enable_text, enable_image, enable_metadata, write_batch_size)

    c = 0
    bar = tqdm.tqdm()
    for item in data:
        with torch.no_grad():
            image_embs = None
            text_embs = None
            image_filename = None
            text = None
            metadata = None
            if enable_image:
                image_features = model_img(item["image_tensor"].to(device))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_embs = image_features.cpu().numpy()
                image_filename = item["image_filename"]
            if enable_text:
                if use_mclip:
                    text_embs = normalized(model_txt(item["text"]))
                else:
                    text_features = model_txt(item["text_tokens"].to(device))
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_embs = text_features.cpu().numpy()
                text = item["text"]
            if enable_metadata:
                metadata = item["metadata"]
            output_sink.add(image_embs, text_embs, image_filename, text, metadata)
        bar.update(batch_size)
        c += batch_size
        if subset_size is not None and c >= subset_size:
            break
    output_sink.flush()


if __name__ == "__main__":
    fire.Fire(clip_inference)
