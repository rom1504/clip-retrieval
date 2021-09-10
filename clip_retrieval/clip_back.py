from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
import clip
import faiss
import torch
import json
from PIL import Image
from io import BytesIO
from PIL import Image
import base64
import os
import fire
from pathlib import Path
import pandas as pd
import urllib
import io
import numpy as np

import h5py
from tqdm import tqdm


class Health(Resource):
    def get(self):
        return "ok"

class IndicesList(Resource):
    def __init__(self, **kwargs):
        super().__init__()
        self.indices = kwargs['indices']


    def get(self):
        return list(self.indices.keys())

def download_image(url):
    request = urllib.request.Request(
        url,
        data=None,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
        },
    )
    with urllib.request.urlopen(request, timeout=10) as r:
        img_stream = io.BytesIO(r.read())
    return img_stream

class KnnService(Resource):
    def __init__(self, **kwargs):
        super().__init__()
        self.indices_loaded = kwargs['indices_loaded']
        self.device = kwargs['device']
        self.model = kwargs['model']
        self.preprocess = kwargs['preprocess']


    def post(self):
        json_data = request.get_json(force=True)
        text_input = json_data["text"] if "text" in json_data else None
        image_input = json_data["image"] if "image" in json_data else None
        image_url_input = json_data["image_url"] if "image_url" in json_data else None
        modality = json_data["modality"]
        num_images = json_data["num_images"]
        indice_name = json_data["indice_name"]
        image_index = self.indices_loaded[indice_name]["image_index"]
        text_index = self.indices_loaded[indice_name]["text_index"]
        metadata_provider = self.indices_loaded[indice_name]["metadata_provider"]

        if text_input is not None:
            text = clip.tokenize([text_input]).to(self.device)
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            query = text_features.cpu().detach().numpy().astype("float32")
        if image_input is not None or image_url_input is not None:
            if image_input is not None:
                binary_data = base64.b64decode(image_input)
                img_data = BytesIO(binary_data)
            elif image_url_input is not None:
                img_data = download_image(image_url_input)
            img = Image.open(img_data)
            prepro = self.preprocess(img).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(prepro)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            query = image_features.cpu().detach().numpy().astype("float32")
        
        index = image_index if modality == "image" else text_index

        D, I = index.search(query, num_images)
        results = []
        metas = metadata_provider.get(I[0], ["url", "image_path", "caption"])
        for key, (d, i) in enumerate(zip(D[0], I[0])):
            output = {}
            meta = metas[key]
            if "image_path" in meta:
                path = meta["image_path"]
                if os.path.exists(path):
                    img = Image.open(path)
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8") 
                    output["image"] = img_str
            for k, v in meta.items():
                if isinstance(v, bytes):
                    v = v.decode()
                elif type(v).__module__ == np.__name__:
                    v = v.item()
                output[k] = v
            output["similarity"] = d.item()
            results.append(output)
        return results

class ParquetMetadataProvider:
    def __init__(self, parquet_folder):
        data_dir = Path(parquet_folder)
        self.metadata_df = pd.concat(
            pd.read_parquet(parquet_file)
            for parquet_file in sorted(data_dir.glob('*.parquet'))
        )

    def get(self, ids, cols=None):
        if cols is None:
            cols = self.metadata_df.columns.tolist()
        else:
            cols = list(set(self.metadata_df.columns.tolist()) & set(cols))

        return [self.metadata_df[i:(i+1)][cols].to_dict(orient='records')[0] for i in ids]


def parquet_to_hdf5(parquet_folder, output_hdf5_file):
    f = h5py.File(output_hdf5_file, 'w')
    data_dir = Path(parquet_folder)
    ds = f.create_group('dataset') 
    for parquet_files in tqdm(sorted(data_dir.glob('*.parquet'))):
        df = pd.read_parquet(parquet_files)
        for k in df.keys():
            if False and not (k == "url"):
                continue
            if False and not (k == "url" or k == 'caption'):
                continue
            col = df[k]
            if col.dtype == 'float64' or col.dtype=='float32':
                col=col.fillna(0.0)
            if col.dtype == 'int64' or col.dtype=='int32':
                col=col.fillna(0)
            if col.dtype == 'object':
                col=col.fillna('')
            z = col.to_numpy()
            if k not in ds:
                ds.create_dataset(k, data=z, maxshape=(None,))
            else:
                prevlen = len(ds[k])
                ds[k].resize((prevlen+len(z),))
                ds[k][prevlen:] = z
    
    del ds
    f.close()

class Hdf5MetadataProvider:
    def __init__(self, hdf5_file):
        f = h5py.File(hdf5_file, 'r')
        self.ds = f['dataset']
    def get(self, ids, cols=None):
        items = [{} for _ in range(len(ids))]
        if cols is None:
            cols = self.ds.keys()
        else:
            cols = list(self.ds.keys() & set(cols))
        for k in cols:
            # broken
            sorted_ids = sorted([(k, i) for i, k in list(enumerate(ids))])
            for_hdf5 = [k for k,_ in sorted_ids]
            for_np = [i for _,i in sorted_ids]
            g = self.ds[k][for_hdf5]
            gg = g[for_np]
            for i, e in enumerate(gg):
                items[i][k] = e
        return items

def clip_back(indices_paths="indices_paths.json", port=1234, enable_hdf5=False, enable_faiss_memory_mapping=False):
    print('loading clip...')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    indices = json.load(open(indices_paths))

    indices_loaded = {}

    for name, indice_folder in indices.items():
        image_present = os.path.exists(indice_folder+"/image.index")
        text_present = os.path.exists(indice_folder+"/text.index")
        hdf5_path = indice_folder+"/metadata.hdf5"
        parquet_folder = indice_folder+"/metadata"
        print('loading metadata...')
        if enable_hdf5:
            if not os.path.exists(hdf5_path):
                parquet_to_hdf5(parquet_folder, hdf5_path)
            metadata_provider = Hdf5MetadataProvider(hdf5_path)
        else:
            metadata_provider = ParquetMetadataProvider(parquet_folder)

        print('loading indices...')
        if image_present:
            if enable_faiss_memory_mapping:
                image_index = faiss.read_index(indice_folder+"/image.index", faiss.IO_FLAG_MMAP|faiss.IO_FLAG_READ_ONLY)
            else:
                image_index = faiss.read_index(indice_folder+"/image.index")
        else:
            image_index = None
        if text_present:
            if enable_faiss_memory_mapping:
                text_index = faiss.read_index(indice_folder+"/text.index", faiss.IO_FLAG_MMAP|faiss.IO_FLAG_READ_ONLY)
            else:
                text_index = faiss.read_index(indice_folder+"/text.index")
        else:
            text_index = None
        indices_loaded[name]={
            'metadata_provider': metadata_provider,
            'image_index': image_index,
            'text_index': text_index
        }

    app = Flask(__name__)
    api = Api(app)
    api.add_resource(IndicesList, '/indices-list', resource_class_kwargs={'indices': indices})
    api.add_resource(KnnService, '/knn-service', resource_class_kwargs={'indices_loaded': indices_loaded, 'device': device, \
    'model': model, 'preprocess': preprocess})
    api.add_resource(Health, '/')
    CORS(app)
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == '__main__':
  fire.Fire(clip_back)
