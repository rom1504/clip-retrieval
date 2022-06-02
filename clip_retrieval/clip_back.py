"""Clip back: host a knn service using clip as an encoder"""


from typing import Callable, Dict, Any, List
from flask import Flask, request, make_response
from flask_restful import Resource, Api
from flask_cors import CORS
import faiss
from collections import defaultdict
from multiprocessing.pool import ThreadPool
import json
from io import BytesIO
from PIL import Image
import base64
import os
import fire
from pathlib import Path
import pandas as pd
import urllib
import tempfile
import io
import numpy as np
from functools import lru_cache
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import pyarrow as pa
import fsspec

import h5py
from tqdm import tqdm
from prometheus_client import Histogram, REGISTRY, make_wsgi_app
import math
import logging

from clip_retrieval.ivf_metadata_ordering import (
    Hdf5Sink,
    external_sort_parquet,
    get_old_to_new_mapping,
    re_order_parquet,
)
from dataclasses import dataclass


LOGGER = logging.getLogger(__name__)


for coll in list(REGISTRY._collector_to_names.keys()):  # pylint: disable=protected-access
    REGISTRY.unregister(coll)

FULL_KNN_REQUEST_TIME = Histogram("full_knn_request_time", "Time spent processing knn request")
DOWNLOAD_TIME = Histogram("download_time", "Time spent downloading an url")
TEXT_CLIP_INFERENCE_TIME = Histogram("text_clip_inference_time", "Time spent doing a text clip inference")
IMAGE_CLIP_INFERENCE_TIME = Histogram("image_clip_inference_time", "Time spent doing a image clip inference")
METADATA_GET_TIME = Histogram("metadata_get_time", "Time spent retrieving metadata")
KNN_INDEX_TIME = Histogram("knn_index_time", "Time spent doing a knn on the index")
DEDUP_TIME = Histogram("dedup_time", "Time spent deduping")
SAFETY_TIME = Histogram("safety_time", "Time spent doing a safety inference")
IMAGE_PREPRO_TIME = Histogram("image_prepro_time", "Time spent doing the image preprocessing")
TEXT_PREPRO_TIME = Histogram("text_prepro_time", "Time spent doing the text preprocessing")


def metric_to_average(metric):
    metric_data = metric.collect()[0]
    metric_name = metric_data.name
    metric_description = metric_data.documentation
    samples = metric_data.samples
    metric_sum = [sample.value for sample in samples if sample.name == metric_name + "_sum"][0]
    metric_count = [sample.value for sample in samples if sample.name == metric_name + "_count"][0]
    if metric_count == 0:
        return metric_name, metric_description, 0, 0.0
    return metric_name, metric_description, metric_count, 1.0 * metric_sum / metric_count


class Health(Resource):
    def get(self):
        return "ok"


class MetricsSummary(Resource):
    """
    metrics endpoint for prometheus
    """

    def get(self):
        """define the metric endpoint get"""
        _, _, full_knn_count, full_knn_avg = metric_to_average(FULL_KNN_REQUEST_TIME)
        if full_knn_count == 0:
            s = "No request yet, go do some"
        else:
            sub_metrics = sorted(
                [
                    (name, description, metric_count, avg, avg / full_knn_avg)
                    for (name, description, metric_count, avg) in [
                        metric_to_average(metric)
                        for metric in [
                            DOWNLOAD_TIME,
                            TEXT_CLIP_INFERENCE_TIME,
                            IMAGE_CLIP_INFERENCE_TIME,
                            METADATA_GET_TIME,
                            KNN_INDEX_TIME,
                            DEDUP_TIME,
                            SAFETY_TIME,
                            IMAGE_PREPRO_TIME,
                            TEXT_PREPRO_TIME,
                        ]
                    ]
                ],
                key=lambda e: -e[3],
            )

            sub_metrics_strings = [
                (name, description, int(metric_count), f"{avg:0.4f}s", f"{proportion*100:0.1f}%")
                for name, description, metric_count, avg, proportion in sub_metrics
            ]

            s = ""
            s += (
                f"Among {full_knn_count} calls to the knn end point with an average latency of {full_knn_avg:0.4f}s "
                + "per request, the step costs are (in order): \n\n"
            )
            df = pd.DataFrame(
                data=sub_metrics_strings, columns=("name", "description", "calls", "average", "proportion")
            )
            s += df.to_string()

        response = make_response(s, 200)
        response.mimetype = "text/plain"
        return response


class IndicesList(Resource):
    def __init__(self, **kwargs):
        super().__init__()
        self.indices = kwargs["indices"]

    def get(self):
        return self.indices


@DOWNLOAD_TIME.time()
def download_image(url):
    urllib_request = urllib.request.Request(
        url,
        data=None,
        headers={"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"},
    )
    with urllib.request.urlopen(urllib_request, timeout=10) as r:
        img_stream = io.BytesIO(r.read())
    return img_stream


class MetadataService(Resource):
    """The metadata service provides metadata given indices"""

    def __init__(self, **kwargs):
        super().__init__()
        self.clip_resources = kwargs["clip_resources"]

    def post(self):
        json_data = request.get_json(force=True)
        ids = json_data["ids"]
        if len(ids) == 0:
            return []
        indice_name = json_data["indice_name"]
        metadata_provider = self.clip_resources[indice_name].metadata_provider
        metas = metadata_provider.get(ids, self.clip_resources[indice_name].columns_to_return)
        metas_with_ids = [{"id": item_id, "metadata": meta_to_dict(meta)} for item_id, meta in zip(ids, metas)]
        return metas_with_ids


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class KnnService(Resource):
    """the knn service provides nearest neighbors given text or image"""

    def __init__(self, **kwargs):
        super().__init__()
        self.clip_resources = kwargs["clip_resources"]

    def compute_query(
        self, clip_resource, text_input, image_input, image_url_input, use_mclip, aesthetic_score, aesthetic_weight
    ):
        """compute the query embedding"""
        import torch  # pylint: disable=import-outside-toplevel
        import clip  # pylint: disable=import-outside-toplevel

        if text_input is not None:
            if use_mclip:
                with TEXT_CLIP_INFERENCE_TIME.time():
                    query = normalized(clip_resource.model_txt_mclip(text_input))
            else:
                with TEXT_PREPRO_TIME.time():
                    text = clip.tokenize([text_input], truncate=True).to(clip_resource.device)
                with TEXT_CLIP_INFERENCE_TIME.time():
                    with torch.no_grad():
                        text_features = clip_resource.model.encode_text(text)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    query = text_features.cpu().detach().numpy().astype("float32")
        if image_input is not None or image_url_input is not None:
            if image_input is not None:
                binary_data = base64.b64decode(image_input)
                img_data = BytesIO(binary_data)
            elif image_url_input is not None:
                img_data = download_image(image_url_input)
            with IMAGE_PREPRO_TIME.time():
                img = Image.open(img_data)
                prepro = clip_resource.preprocess(img).unsqueeze(0).to(clip_resource.device)
            with IMAGE_CLIP_INFERENCE_TIME.time():
                with torch.no_grad():
                    image_features = clip_resource.model.encode_image(prepro)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                query = image_features.cpu().detach().numpy().astype("float32")

        if clip_resource.aesthetic_embeddings is not None and aesthetic_score is not None:
            aesthetic_embedding = clip_resource.aesthetic_embeddings[aesthetic_score]
            query = query + aesthetic_embedding * aesthetic_weight
            query = query / np.linalg.norm(query)

        return query

    def hash_based_dedup(self, embeddings):
        """deduplicate embeddings based on their hash"""
        seen_hashes = set()
        to_remove = []
        for i, embedding in enumerate(embeddings):
            h = hash(np.round(embedding, 2).tobytes())
            if h in seen_hashes:
                to_remove.append(i)
                continue
            seen_hashes.add(h)

        return to_remove

    def connected_components(self, neighbors):
        """find connected components in the graph"""
        seen = set()

        def component(node):
            r = []
            nodes = set([node])
            while nodes:
                node = nodes.pop()
                seen.add(node)
                nodes |= set(neighbors[node]) - seen
                r.append(node)
            return r

        u = []
        for node in neighbors:
            if node not in seen:
                u.append(component(node))
        return u

    def get_non_uniques(self, embeddings, threshold=0.94):
        """find non-unique embeddings"""
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)  # pylint: disable=no-value-for-parameter
        l, _, I = index.range_search(embeddings, threshold)  # pylint: disable=no-value-for-parameter,invalid-name

        same_mapping = defaultdict(list)

        # https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes#range-search
        for i in range(embeddings.shape[0]):
            for j in I[l[i] : l[i + 1]]:
                same_mapping[int(i)].append(int(j))

        groups = self.connected_components(same_mapping)
        non_uniques = set()
        for g in groups:
            for e in g[1:]:
                non_uniques.add(e)

        return list(non_uniques)

    def connected_components_dedup(self, embeddings):
        non_uniques = self.get_non_uniques(embeddings)
        return non_uniques

    def get_unsafe_items(self, safety_model, embeddings, threshold=0.5):
        """find unsafe embeddings"""
        nsfw_values = safety_model.predict(embeddings, batch_size=embeddings.shape[0])
        x = np.array([e[0] for e in nsfw_values])
        return np.where(x > threshold)[0]

    def get_violent_items(self, safety_prompts, embeddings):
        safety_predictions = np.einsum("ij,kj->ik", embeddings, safety_prompts)
        safety_results = np.argmax(safety_predictions, axis=1)
        return np.where(safety_results == 1)[0]

    def post_filter(
        self, safety_model, embeddings, deduplicate, use_safety_model, use_violence_detector, violence_detector
    ):
        """post filter results : dedup, safety, violence"""
        to_remove = set()
        if deduplicate:
            with DEDUP_TIME.time():
                to_remove = set(self.connected_components_dedup(embeddings))

        if use_violence_detector and violence_detector is not None:
            to_remove |= set(self.get_violent_items(violence_detector, embeddings))
        if use_safety_model and safety_model is not None:
            with SAFETY_TIME.time():
                to_remove |= set(self.get_unsafe_items(safety_model, embeddings))

        return to_remove

    def knn_search(
        self, query, modality, num_result_ids, clip_resource, deduplicate, use_safety_model, use_violence_detector
    ):
        """compute the knn search"""

        image_index = clip_resource.image_index
        text_index = clip_resource.text_index
        if clip_resource.metadata_is_ordered_by_ivf:
            ivf_old_to_new_mapping = clip_resource.ivf_old_to_new_mapping

        index = image_index if modality == "image" else text_index

        with KNN_INDEX_TIME.time():
            if clip_resource.metadata_is_ordered_by_ivf:
                previous_nprobe = faiss.extract_index_ivf(index).nprobe
                if num_result_ids >= 100000:
                    nprobe = math.ceil(num_result_ids / 3000)
                    params = faiss.ParameterSpace()
                    params.set_index_parameters(index, f"nprobe={nprobe},efSearch={nprobe*2},ht={2048}")
            distances, indices, embeddings = index.search_and_reconstruct(query, num_result_ids)
            if clip_resource.metadata_is_ordered_by_ivf:
                results = np.take(ivf_old_to_new_mapping, indices[0])
            else:
                results = indices[0]
            if clip_resource.metadata_is_ordered_by_ivf:
                params = faiss.ParameterSpace()
                params.set_index_parameters(index, f"nprobe={previous_nprobe},efSearch={previous_nprobe*2},ht={2048}")
        nb_results = np.where(results == -1)[0]

        if len(nb_results) > 0:
            nb_results = nb_results[0]
        else:
            nb_results = len(results)
        result_indices = results[:nb_results]
        result_distances = distances[0][:nb_results]
        result_embeddings = embeddings[0][:nb_results]
        result_embeddings = normalized(result_embeddings)
        local_indices_to_remove = self.post_filter(
            clip_resource.safety_model,
            result_embeddings,
            deduplicate,
            use_safety_model,
            use_violence_detector,
            clip_resource.violence_detector,
        )
        indices_to_remove = set()
        for local_index in local_indices_to_remove:
            indices_to_remove.add(result_indices[local_index])
        indices = []
        distances = []
        for ind, distance in zip(result_indices, result_distances):
            if ind not in indices_to_remove:
                indices_to_remove.add(ind)
                indices.append(ind)
                distances.append(distance)

        return distances, indices

    def map_to_metadata(self, indices, distances, num_images, metadata_provider, columns_to_return):
        """map the indices to the metadata"""

        results = []
        with METADATA_GET_TIME.time():
            metas = metadata_provider.get(indices[:num_images], columns_to_return)
        for key, (d, i) in enumerate(zip(distances, indices)):
            output = {}
            meta = None if key + 1 > len(metas) else metas[key]
            if meta is not None and "image_path" in meta:
                path = meta["image_path"]
                if os.path.exists(path):
                    img = Image.open(path)
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    output["image"] = img_str
            if meta is not None:
                output.update(meta_to_dict(meta))
            output["id"] = i.item()
            output["similarity"] = d.item()
            results.append(output)

        return results

    def query(
        self,
        text_input=None,
        image_input=None,
        image_url_input=None,
        modality="image",
        num_images=100,
        num_result_ids=100,
        indice_name=None,
        use_mclip=False,
        deduplicate=True,
        use_safety_model=False,
        use_violence_detector=False,
        aesthetic_score=None,
        aesthetic_weight=None,
    ):
        """implement the querying functionality of the knn service: from text and image to nearest neighbors"""

        if text_input is None and image_input is None and image_url_input is None:
            raise ValueError("must fill one of text, image and image url input")
        if indice_name is None:
            indice_name = next(iter(self.clip_resources.keys()))

        clip_resource = self.clip_resources[indice_name]

        query = self.compute_query(
            clip_resource=clip_resource,
            text_input=text_input,
            image_input=image_input,
            image_url_input=image_url_input,
            use_mclip=use_mclip,
            aesthetic_score=aesthetic_score,
            aesthetic_weight=aesthetic_weight,
        )
        distances, indices = self.knn_search(
            query,
            modality=modality,
            num_result_ids=num_result_ids,
            clip_resource=clip_resource,
            deduplicate=deduplicate,
            use_safety_model=use_safety_model,
            use_violence_detector=use_violence_detector,
        )
        if len(distances) == 0:
            return []
        results = self.map_to_metadata(
            indices, distances, num_images, clip_resource.metadata_provider, clip_resource.columns_to_return
        )

        return results

    @FULL_KNN_REQUEST_TIME.time()
    def post(self):
        """implement the post method for knn service, parse the request and calls the query method"""
        json_data = request.get_json(force=True)
        text_input = json_data.get("text", None)
        image_input = json_data.get("image", None)
        image_url_input = json_data.get("image_url", None)
        modality = json_data["modality"]
        num_images = json_data["num_images"]
        num_result_ids = json_data.get("num_result_ids", num_images)
        indice_name = json_data["indice_name"]
        use_mclip = json_data.get("use_mclip", False)
        deduplicate = json_data.get("deduplicate", False)
        use_safety_model = json_data.get("use_safety_model", False)
        use_violence_detector = json_data.get("use_violence_detector", False)
        aesthetic_score = json_data.get("aesthetic_score", "")
        aesthetic_score = int(aesthetic_score) if aesthetic_score != "" else None
        aesthetic_weight = json_data.get("aesthetic_weight", "")
        aesthetic_weight = float(aesthetic_weight) if aesthetic_weight != "" else None
        return self.query(
            text_input,
            image_input,
            image_url_input,
            modality,
            num_images,
            num_result_ids,
            indice_name,
            use_mclip,
            deduplicate,
            use_safety_model,
            use_violence_detector,
            aesthetic_score,
            aesthetic_weight,
        )


def meta_to_dict(meta):
    output = {}
    for k, v in meta.items():
        if isinstance(v, bytes):
            v = v.decode()
        elif type(v).__module__ == np.__name__:
            v = v.item()
        output[k] = v
    return output


class ParquetMetadataProvider:
    """The parquet metadata provider provides metadata from contiguous ids using parquet"""

    def __init__(self, parquet_folder):
        data_dir = Path(parquet_folder)
        self.metadata_df = pd.concat(
            pd.read_parquet(parquet_file) for parquet_file in sorted(data_dir.glob("*.parquet"))
        )

    def get(self, ids, cols=None):
        if cols is None:
            cols = self.metadata_df.columns.tolist()
        else:
            cols = list(set(self.metadata_df.columns.tolist()) & set(cols))

        return [self.metadata_df[i : (i + 1)][cols].to_dict(orient="records")[0] for i in ids]


def parquet_to_hdf5(parquet_folder, output_hdf5_file, columns_to_return):
    """this convert a collection of parquet file to an hdf5 file"""
    f = h5py.File(output_hdf5_file, "w")
    data_dir = Path(parquet_folder)
    ds = f.create_group("dataset")
    for parquet_files in tqdm(sorted(data_dir.glob("*.parquet"))):
        df = pd.read_parquet(parquet_files)
        for k in df.keys():
            if k not in columns_to_return:
                continue
            col = df[k]
            if col.dtype in ("float64", "float32"):
                col = col.fillna(0.0)
            if col.dtype in ("int64", "int32"):
                col = col.fillna(0)
            if col.dtype == "object":
                col = col.fillna("")
                col = col.str.replace("\x00", "", regex=False)
            z = col.to_numpy()
            if k not in ds:
                ds.create_dataset(k, data=z, maxshape=(None,), compression="gzip")
            else:
                prevlen = len(ds[k])
                ds[k].resize((prevlen + len(z),))
                ds[k][prevlen:] = z

    del ds
    f.close()


class Hdf5MetadataProvider:
    """The hdf5 metadata provider provides metadata from contiguous ids using hdf5"""

    def __init__(self, hdf5_file):
        f = h5py.File(hdf5_file, "r")
        self.ds = f["dataset"]

    def get(self, ids, cols=None):
        """implement the get method from the hdf5 metadata provide, get metadata from ids"""
        items = [{} for _ in range(len(ids))]
        if cols is None:
            cols = self.ds.keys()
        else:
            cols = list(self.ds.keys() & set(cols))
        for k in cols:
            for i, e in enumerate(ids):
                items[i][k] = self.ds[k][e]
        return items


def load_index(path, enable_faiss_memory_mapping):
    if enable_faiss_memory_mapping:
        if os.path.isdir(path):
            return faiss.read_index(path + "/populated.index", faiss.IO_FLAG_ONDISK_SAME_DIR)
        else:
            return faiss.read_index(path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    else:
        return faiss.read_index(path)


class ArrowMetadataProvider:
    """The arrow metadata provider provides metadata from contiguous ids using arrow"""

    def __init__(self, arrow_folder):
        arrow_files = [str(a) for a in sorted(Path(arrow_folder).glob("**/*")) if a.is_file()]
        self.table = pa.concat_tables(
            [pa.ipc.RecordBatchFileReader(pa.memory_map(arrow_file, "r")).read_all() for arrow_file in arrow_files]
        )

    def get(self, ids, cols=None):
        """implement the get method from the arrow metadata provide, get metadata from ids"""
        if cols is None:
            cols = self.table.schema.names
        else:
            cols = list(set(self.table.schema.names) & set(cols))
        t = pa.concat_tables([self.table[i : i + 1] for i in ids])
        return t.select(cols).to_pandas().to_dict("records")


def load_metadata_provider(
    indice_folder, enable_hdf5, reorder_metadata_by_ivf_index, image_index, columns_to_return, use_arrow
):
    """load the metadata provider"""
    parquet_folder = indice_folder + "/metadata"
    ivf_old_to_new_mapping = None
    if use_arrow:
        mmap_folder = parquet_folder
        metadata_provider = ArrowMetadataProvider(mmap_folder)
    elif enable_hdf5:
        hdf5_path = None
        if reorder_metadata_by_ivf_index:
            hdf5_path = indice_folder + "/metadata_reordered.hdf5"
            ivf_old_to_new_mapping_path = indice_folder + "/ivf_old_to_new_mapping.npy"
            if not os.path.exists(ivf_old_to_new_mapping_path):
                ivf_old_to_new_mapping = get_old_to_new_mapping(image_index)
                ivf_old_to_new_mapping_write = np.memmap(
                    ivf_old_to_new_mapping_path, dtype="int64", mode="write", shape=ivf_old_to_new_mapping.shape
                )
                ivf_old_to_new_mapping_write[:] = ivf_old_to_new_mapping
                del ivf_old_to_new_mapping_write
                del ivf_old_to_new_mapping
            ivf_old_to_new_mapping = np.memmap(ivf_old_to_new_mapping_path, dtype="int64", mode="r")
            if not os.path.exists(hdf5_path):
                with tempfile.TemporaryDirectory() as tmpdir:
                    re_order_parquet(image_index, parquet_folder, str(tmpdir), columns_to_return)
                    external_sort_parquet(Hdf5Sink(hdf5_path, columns_to_return), str(tmpdir))
        else:
            hdf5_path = indice_folder + "/metadata.hdf5"
            if not os.path.exists(hdf5_path):
                parquet_to_hdf5(parquet_folder, hdf5_path, columns_to_return)
        metadata_provider = Hdf5MetadataProvider(hdf5_path)
    else:
        metadata_provider = ParquetMetadataProvider(parquet_folder)

    return metadata_provider, ivf_old_to_new_mapping


def get_cache_folder(clip_model):
    """get cache folder for given clip model"""
    from os.path import expanduser  # pylint: disable=import-outside-toplevel

    home = expanduser("~")

    cache_folder = home + "/.cache/clip_retrieval/" + clip_model.replace("/", "_")

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder, exist_ok=True)

    return cache_folder


# needs to do this at load time
@lru_cache(maxsize=None)
def get_aesthetic_embedding(model_type):
    """get aesthetic embedding"""
    if model_type == "ViT-B/32":
        model_type = "vit_b_32"
    elif model_type == "ViT-L/14":
        model_type = "vit_l_14"

    fs, _ = fsspec.core.url_to_fs(
        f"https://github.com/LAION-AI/aesthetic-predictor/blob/main/{model_type}_embeddings/rating0.npy?raw=true"
    )
    embs = {}
    with ThreadPool(10) as pool:

        def get(k):
            with fs.open(
                f"https://github.com/LAION-AI/aesthetic-predictor/blob/main/{model_type}_embeddings/rating{k}.npy?raw=true",
                "rb",
            ) as f:
                embs[k] = np.load(f)

        for _ in pool.imap_unordered(get, range(10)):
            pass
    return embs


@lru_cache(maxsize=None)
def load_violence_detector(clip_model):
    """load violence detector for this clip model"""
    from urllib.request import urlretrieve  #  pylint: disable=import-outside-toplevel

    cache_folder = get_cache_folder(clip_model)
    root_url = "https://github.com/LAION-AI/CLIP-based-NSFW-Detector/raw/main"

    if clip_model == "ViT-L/14":
        name = "violence_detection_vit_l_14.npy"
    elif clip_model == "ViT-B/32":
        name = "violence_detection_vit_b_32.npy"
    else:
        raise ValueError("Unknown clip model")

    url_model = root_url + "/" + name
    prompt_file = cache_folder + "/" + name

    if not os.path.exists(prompt_file):
        urlretrieve(url_model, prompt_file)

    prompts = np.load(prompt_file)
    return prompts


@lru_cache(maxsize=None)
def load_safety_model(clip_model):
    """load the safety model"""
    import autokeras as ak  # pylint: disable=import-outside-toplevel
    from tensorflow.keras.models import load_model  # pylint: disable=import-outside-toplevel

    cache_folder = get_cache_folder(clip_model)

    if clip_model == "ViT-L/14":
        model_dir = cache_folder + "/clip_autokeras_binary_nsfw"
        dim = 768
    elif clip_model == "ViT-B/32":
        model_dir = cache_folder + "/clip_autokeras_nsfw_b32"
        dim = 512
    else:
        raise ValueError("Unknown clip model")
    if not os.path.exists(model_dir):
        os.makedirs(cache_folder, exist_ok=True)

        from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

        path_to_zip_file = cache_folder + "/clip_autokeras_binary_nsfw.zip"
        if clip_model == "ViT-L/14":
            url_model = "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip"
        elif clip_model == "ViT-B/32":
            url_model = (
                "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_nsfw_b32.zip"
            )
        else:
            raise ValueError("Unknown model {}".format(clip_model))  # pylint: disable=consider-using-f-string
        urlretrieve(url_model, path_to_zip_file)
        import zipfile  # pylint: disable=import-outside-toplevel

        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(cache_folder)

    loaded_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)
    loaded_model.predict(np.random.rand(10**3, dim).astype("float32"), batch_size=10**3)

    return loaded_model


@dataclass
class ClipResource:
    """the resource for clip : model, index, options"""

    device: str
    model: Any
    preprocess: Callable
    model_txt_mclip: Any
    safety_model: Any
    violence_detector: Any
    metadata_provider: Any
    image_index: Any
    text_index: Any
    ivf_old_to_new_mapping: Any
    columns_to_return: List[str]
    metadata_is_ordered_by_ivf: bool
    aesthetic_embeddings: Any


@dataclass
class ClipOptions:
    """the options for clip"""

    indice_folder: str
    clip_model: str
    enable_hdf5: bool
    enable_faiss_memory_mapping: bool
    columns_to_return: List[str]
    reorder_metadata_by_ivf_index: bool
    enable_mclip_option: bool
    use_jit: bool
    use_arrow: bool
    provide_safety_model: bool
    provide_violence_detector: bool
    provide_aesthetic_embeddings: bool


def dict_to_clip_options(d, clip_options):
    return ClipOptions(
        indice_folder=d["indice_folder"] if "indice_folder" in d else clip_options.indice_folder,
        clip_model=d["clip_model"] if "clip_model" in d else clip_options.clip_model,
        enable_hdf5=d["enable_hdf5"] if "enable_hdf5" in d else clip_options.enable_hdf5,
        enable_faiss_memory_mapping=d["enable_faiss_memory_mapping"]
        if "enable_faiss_memory_mapping" in d
        else clip_options.enable_faiss_memory_mapping,
        columns_to_return=d["columns_to_return"] if "columns_to_return" in d else clip_options.columns_to_return,
        reorder_metadata_by_ivf_index=d["reorder_metadata_by_ivf_index"]
        if "reorder_metadata_by_ivf_index" in d
        else clip_options.reorder_metadata_by_ivf_index,
        enable_mclip_option=d["enable_mclip_option"]
        if "enable_mclip_option" in d
        else clip_options.enable_mclip_option,
        use_jit=d["use_jit"] if "use_jit" in d else clip_options.use_jit,
        use_arrow=d["use_arrow"] if "use_arrow" in d else clip_options.use_arrow,
        provide_safety_model=d["provide_safety_model"]
        if "provide_safety_model" in d
        else clip_options.provide_safety_model,
        provide_violence_detector=d["provide_violence_detector"]
        if "provide_violence_detector" in d
        else clip_options.provide_violence_detector,
        provide_aesthetic_embeddings=d["provide_aesthetic_embeddings"]
        if "provide_aesthetic_embeddings" in d
        else clip_options.provide_aesthetic_embeddings,
    )


@lru_cache(maxsize=None)
def load_clip(clip_model="ViT-B/32", use_jit=True, device="cpu"):
    import clip  # pylint: disable=import-outside-toplevel

    model, preprocess = clip.load(clip_model, device=device, jit=use_jit)
    return model, preprocess


@lru_cache(maxsize=None)
def load_mclip(clip_model):
    """load the mclip model"""
    from multilingual_clip import pt_multilingual_clip  # pylint: disable=import-outside-toplevel
    import transformers  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    if clip_model == "ViT-L/14":
        model_name = "M-CLIP/XLM-Roberta-Large-Vit-L-14"
    elif clip_model == "ViT-B/32":
        model_name = "M-CLIP/XLM-Roberta-Large-Vit-B-32"

    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    def encode_texts(text):
        with torch.no_grad():
            return model.forward([text], tokenizer)[0].detach().cpu().numpy()

    model_txt_mclip = encode_texts
    return model_txt_mclip


def load_clip_index(clip_options):
    """load the clip index"""
    import torch  # pylint: disable=import-outside-toplevel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip(clip_options.clip_model, use_jit=clip_options.use_jit, device=device)

    if clip_options.enable_mclip_option:
        model_txt_mclip = load_mclip(clip_options.clip_model)
    else:
        model_txt_mclip = None

    safety_model = load_safety_model(clip_options.clip_model) if clip_options.provide_safety_model else None
    violence_detector = (
        load_violence_detector(clip_options.clip_model) if clip_options.provide_violence_detector else None
    )
    aesthetic_embeddings = (
        get_aesthetic_embedding(clip_options.clip_model) if clip_options.provide_aesthetic_embeddings else None
    )

    image_present = os.path.exists(clip_options.indice_folder + "/image.index")
    text_present = os.path.exists(clip_options.indice_folder + "/text.index")

    LOGGER.info("loading indices...")
    image_index = (
        load_index(clip_options.indice_folder + "/image.index", clip_options.enable_faiss_memory_mapping)
        if image_present
        else None
    )
    text_index = (
        load_index(clip_options.indice_folder + "/text.index", clip_options.enable_faiss_memory_mapping)
        if text_present
        else None
    )

    LOGGER.info("loading metadata...")

    metadata_provider, ivf_old_to_new_mapping = load_metadata_provider(
        clip_options.indice_folder,
        clip_options.enable_hdf5,
        clip_options.reorder_metadata_by_ivf_index,
        image_index,
        clip_options.columns_to_return,
        clip_options.use_arrow,
    )

    return ClipResource(
        device=device,
        model=model,
        preprocess=preprocess,
        model_txt_mclip=model_txt_mclip,
        safety_model=safety_model,
        violence_detector=violence_detector,
        metadata_provider=metadata_provider,
        image_index=image_index,
        text_index=text_index,
        ivf_old_to_new_mapping=ivf_old_to_new_mapping if clip_options.reorder_metadata_by_ivf_index else None,
        columns_to_return=clip_options.columns_to_return,
        metadata_is_ordered_by_ivf=clip_options.reorder_metadata_by_ivf_index,
        aesthetic_embeddings=aesthetic_embeddings,
    )


def load_clip_indices(
    indices_paths,
    clip_options,
) -> Dict[str, ClipResource]:
    """This load clips indices from disk"""
    LOGGER.info("loading clip...")

    with open(indices_paths, "r", encoding="utf-8") as f:
        indices = json.load(f)

    clip_resources = {}

    for name, indice_value in indices.items():
        # if indice_folder is a string
        if isinstance(indice_value, str):
            clip_options = dict_to_clip_options({"indice_folder": indice_value}, clip_options)
        elif isinstance(indice_value, dict):
            clip_options = dict_to_clip_options(indice_value, clip_options)
        else:
            raise ValueError("Unknown type for indice_folder")
        clip_resources[name] = load_clip_index(clip_options)

    return clip_resources


# reorder_metadata_by_ivf_index allows faster data retrieval of knn results by re-ordering the metadata by the ivf clusters
def clip_back(
    indices_paths="indices_paths.json",
    port=1234,
    enable_hdf5=False,
    enable_faiss_memory_mapping=False,
    columns_to_return=None,
    reorder_metadata_by_ivf_index=False,
    default_backend=None,
    url_column="url",
    enable_mclip_option=True,
    clip_model="ViT-B/32",
    use_jit=True,
    use_arrow=False,
    provide_safety_model=False,
    provide_violence_detector=False,
    provide_aesthetic_embeddings=True,
):
    """main entry point of clip back, start the endpoints"""
    print("starting boot of clip back")
    if columns_to_return is None:
        columns_to_return = ["url", "image_path", "caption", "NSFW"]
    clip_resources = load_clip_indices(
        indices_paths=indices_paths,
        clip_options=ClipOptions(
            indice_folder="",
            clip_model=clip_model,
            enable_hdf5=enable_hdf5,
            enable_faiss_memory_mapping=enable_faiss_memory_mapping,
            columns_to_return=columns_to_return,
            reorder_metadata_by_ivf_index=reorder_metadata_by_ivf_index,
            enable_mclip_option=enable_mclip_option,
            use_jit=use_jit,
            use_arrow=use_arrow,
            provide_safety_model=provide_safety_model,
            provide_violence_detector=provide_violence_detector,
            provide_aesthetic_embeddings=provide_aesthetic_embeddings,
        ),
    )
    print("indices loaded")

    app = Flask(__name__)
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/metrics": make_wsgi_app()})
    from .clip_front import add_static_endpoints  # pylint: disable=import-outside-toplevel

    add_static_endpoints(app, default_backend, None, url_column)

    api = Api(app)
    api.add_resource(MetricsSummary, "/metrics-summary")
    api.add_resource(IndicesList, "/indices-list", resource_class_kwargs={"indices": list(clip_resources.keys())})
    api.add_resource(
        MetadataService,
        "/metadata",
        resource_class_kwargs={
            "clip_resources": clip_resources,
        },
    )
    api.add_resource(
        KnnService,
        "/knn-service",
        resource_class_kwargs={
            "clip_resources": clip_resources,
        },
    )
    CORS(app)
    LOGGER.info("starting!")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    fire.Fire(clip_back)
