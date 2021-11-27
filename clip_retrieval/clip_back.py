"""Clip back: host a knn service using clip as an encoder"""


from flask import Flask, request, make_response
from flask_restful import Resource, Api
from flask_cors import CORS
import faiss
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
from werkzeug.middleware.dispatcher import DispatcherMiddleware

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

LOGGER = logging.getLogger(__name__)


for coll in list(REGISTRY._collector_to_names.keys()):  # pylint: disable=protected-access
    REGISTRY.unregister(coll)

FULL_KNN_REQUEST_TIME = Histogram("full_knn_request_time", "Time spent processing knn request")
DOWNLOAD_TIME = Histogram("download_time", "Time spent downloading an url")
TEXT_CLIP_INFERENCE_TIME = Histogram("text_clip_inference_time", "Time spent doing a text clip inference")
IMAGE_CLIP_INFERENCE_TIME = Histogram("image_clip_inference_time", "Time spent doing a image clip inference")
METADATA_GET_TIME = Histogram("metadata_get_time", "Time spent retrieving metadata")
KNN_INDEX_TIME = Histogram("knn_index_time", "Time spent doing a knn on the index")
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
        return list(self.indices.keys())


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
        self.indices_loaded = kwargs["indices_loaded"]
        self.columns_to_return = kwargs["columns_to_return"]

    def post(self):
        json_data = request.get_json(force=True)
        ids = json_data["ids"]
        indice_name = json_data["indice_name"]
        metadata_provider = self.indices_loaded[indice_name]["metadata_provider"]
        metas = metadata_provider.get(ids, self.columns_to_return)
        metas_with_ids = [{"id": item_id, "metadata": meta_to_dict(meta)} for item_id, meta in zip(ids, metas)]
        return metas_with_ids


class KnnService(Resource):
    """the knn service provides nearest neighbors given text or image"""

    def __init__(self, **kwargs):
        super().__init__()
        self.indices_loaded = kwargs["indices_loaded"]
        self.device = kwargs["device"]
        self.model = kwargs["model"]
        self.preprocess = kwargs["preprocess"]
        self.columns_to_return = kwargs["columns_to_return"]
        self.metadata_is_ordered_by_ivf = kwargs["metadata_is_ordered_by_ivf"]

    def query(
        self,
        text_input=None,
        image_input=None,
        image_url_input=None,
        modality="image",
        num_images=100,
        num_result_ids=100,
        indice_name=None,
    ):
        """implement the querying functionality of the knn service: from text and image to nearest neighbors"""
        import clip  # pylint: disable=import-outside-toplevel

        if text_input is None and image_input is None and image_url_input is None:
            raise ValueError("must fill one of text, image and image url input")
        if indice_name is None:
            indice_name = next(iter(self.indices_loaded.keys()))
        image_index = self.indices_loaded[indice_name]["image_index"]
        text_index = self.indices_loaded[indice_name]["text_index"]
        metadata_provider = self.indices_loaded[indice_name]["metadata_provider"]
        if self.metadata_is_ordered_by_ivf:
            ivf_old_to_new_mapping = self.indices_loaded[indice_name]["ivf_old_to_new_mapping"]

        if text_input is not None:
            with TEXT_PREPRO_TIME.time():
                text = clip.tokenize([text_input]).to(self.device)
            with TEXT_CLIP_INFERENCE_TIME.time():
                text_features = self.model.encode_text(text)
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
                prepro = self.preprocess(img).unsqueeze(0).to(self.device)
            with IMAGE_CLIP_INFERENCE_TIME.time():
                image_features = self.model.encode_image(prepro)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                query = image_features.cpu().detach().numpy().astype("float32")

        index = image_index if modality == "image" else text_index

        with KNN_INDEX_TIME.time():
            if self.metadata_is_ordered_by_ivf:
                previous_nprobe = faiss.extract_index_ivf(index).nprobe
                if num_result_ids >= 100000:
                    nprobe = math.ceil(num_result_ids / 3000)
                    params = faiss.ParameterSpace()
                    params.set_index_parameters(index, f"nprobe={nprobe},efSearch={nprobe*2},ht={2048}")
            if self.metadata_is_ordered_by_ivf:
                distances, indices = index.search(query, num_result_ids)
                results = np.take(ivf_old_to_new_mapping, indices[0])
            else:
                distances, indices = index.search(query, num_result_ids)
                results = indices[0]
            if self.metadata_is_ordered_by_ivf:
                params = faiss.ParameterSpace()
                params.set_index_parameters(index, f"nprobe={previous_nprobe},efSearch={previous_nprobe*2},ht={2048}")
        nb_results = np.where(results == -1)[0]
        if len(nb_results) > 0:
            nb_results = nb_results[0]
        else:
            nb_results = len(results)
        result_indices = results[:nb_results]
        result_distances = distances[0][:nb_results]
        existing_indices = set()
        indices = []
        distances = []
        for ind, distance in zip(result_indices, result_distances):
            if ind not in existing_indices:
                existing_indices.add(ind)
                indices.append(ind)
                distances.append(distance)
        del existing_indices
        del result_indices
        del result_distances
        results = []
        with METADATA_GET_TIME.time():
            metas = metadata_provider.get(indices[:num_images], self.columns_to_return)
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
        return self.query(text_input, image_input, image_url_input, modality, num_images, num_result_ids, indice_name)


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
            if col.dtype == "float64" or col.dtype == "float32":
                col = col.fillna(0.0)
            if col.dtype == "int64" or col.dtype == "int32":
                col = col.fillna(0)
            if col.dtype == "object":
                col = col.fillna("")
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
            sorted_ids = sorted([(k, i) for i, k in list(enumerate(ids))])
            for_hdf5 = [k for k, _ in sorted_ids]
            for_np = [i for _, i in sorted_ids]
            if len(for_hdf5) <= 10000:
                batch_size = 100
            else:
                batch_size = 1000
            g = [
                self.ds[k][for_hdf5[i * batch_size : (i + 1) * batch_size]]
                for i in range(math.ceil(len(for_hdf5) / batch_size))
            ]
            g = np.concatenate(g)
            gg = g[for_np]
            for i, e in enumerate(gg):
                items[i][k] = e
        return items


def load_clip_indices(
    indices_paths, enable_hdf5, enable_faiss_memory_mapping, columns_to_return, reorder_metadata_by_ivf_index
):
    """This load clips indices from disk"""
    LOGGER.info("loading clip...")
    import clip  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    indices = json.load(open(indices_paths))

    indices_loaded = {}

    for name, indice_folder in indices.items():
        image_present = os.path.exists(indice_folder + "/image.index")
        text_present = os.path.exists(indice_folder + "/text.index")

        LOGGER.info("loading indices...")
        if image_present:
            if enable_faiss_memory_mapping:
                image_index = faiss.read_index(
                    indice_folder + "/image.index", faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
                )
            else:
                image_index = faiss.read_index(indice_folder + "/image.index")
        else:
            image_index = None
        if text_present:
            if enable_faiss_memory_mapping:
                text_index = faiss.read_index(
                    indice_folder + "/text.index", faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
                )
            else:
                text_index = faiss.read_index(indice_folder + "/text.index")
        else:
            text_index = None

        parquet_folder = indice_folder + "/metadata"
        LOGGER.info("loading metadata...")
        if enable_hdf5:
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

        indices_loaded[name] = {
            "metadata_provider": metadata_provider,
            "image_index": image_index,
            "text_index": text_index,
        }
        if reorder_metadata_by_ivf_index:
            indices_loaded[name]["ivf_old_to_new_mapping"] = ivf_old_to_new_mapping

    return indices_loaded, indices, device, model, preprocess


# reorder_metadata_by_ivf_index allows faster data retrieval of knn results by re-ordering the metadata by the ivf clusters


def clip_back(
    indices_paths="indices_paths.json",
    port=1234,
    enable_hdf5=False,
    enable_faiss_memory_mapping=False,
    columns_to_return=None,
    reorder_metadata_by_ivf_index=False,
    default_backend=None,
):
    """main entry point of clip back, start the endpoints"""
    LOGGER.info("starting boot of clip back")
    if columns_to_return is None:
        columns_to_return = ["url", "image_path", "caption", "NSFW"]
    indices_loaded, indices, device, model, preprocess = load_clip_indices(
        indices_paths, enable_hdf5, enable_faiss_memory_mapping, columns_to_return, reorder_metadata_by_ivf_index
    )
    LOGGER.info("indices loaded")

    app = Flask(__name__)
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/metrics": make_wsgi_app()})
    from .clip_front import add_static_endpoints  # pylint: disable=import-outside-toplevel

    add_static_endpoints(app, default_backend)

    api = Api(app)
    api.add_resource(MetricsSummary, "/metrics-summary")
    api.add_resource(IndicesList, "/indices-list", resource_class_kwargs={"indices": indices})
    api.add_resource(
        MetadataService,
        "/metadata",
        resource_class_kwargs={"indices_loaded": indices_loaded, "columns_to_return": columns_to_return},
    )
    api.add_resource(
        KnnService,
        "/knn-service",
        resource_class_kwargs={
            "indices_loaded": indices_loaded,
            "device": device,
            "model": model,
            "preprocess": preprocess,
            "columns_to_return": columns_to_return,
            "metadata_is_ordered_by_ivf": reorder_metadata_by_ivf_index,
        },
    )
    CORS(app)
    LOGGER.info("starting!")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    fire.Fire(clip_back)
