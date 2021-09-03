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


class Health(Resource):
    def get(self):
        return "ok"

class IndicesList(Resource):
    def __init__(self, **kwargs):
        super().__init__()
        self.indices = kwargs['indices']


    def get(self):
        return list(self.indices.keys())

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
        modality = json_data["modality"]
        num_images = json_data["num_images"]
        indice_name = json_data["indice_name"]
        image_index = self.indices_loaded[indice_name]["image_index"]
        text_index = self.indices_loaded[indice_name]["text_index"]
        image_list = self.indices_loaded[indice_name]["image_list"]
        description_list = self.indices_loaded[indice_name]["description_list"]
        url_list = self.indices_loaded[indice_name]["url_list"]

        if text_input is not None:
            text = clip.tokenize([text_input]).to(self.device)
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            query = text_features.cpu().detach().numpy().astype("float32")
        if image_input is not None:
            binary_data = base64.b64decode(image_input)
            img_data = BytesIO(binary_data)
            img = Image.open(img_data)
            prepro = self.preprocess(img).unsqueeze(0)
            image_features = self.model.encode_image(prepro)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            query = image_features.cpu().detach().numpy().astype("float32")
        
        index = image_index if modality == "image" else text_index

        D, I = index.search(query, num_images)
        results = []
        for d, i in zip(D[0], I[0]):
            output = {}
            if image_list is not None:
                path = image_list[i]
                if os.path.exists(path):
                    img = Image.open(path)
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8") 
                    output["image"] = img_str
            if description_list is not None:
                description = description_list[i]
                output["text"] = description
            if url_list is not None:
                output["url"] = url_list[i]
            results.append(output)
        return results


def clip_back(indices_paths="indices_paths.json", port=1234):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    indices = json.load(open(indices_paths))

    indices_loaded = {}

    for name, indice_folder in indices.items():
        image_present = os.path.exists(indice_folder+"/image.index")
        text_present = os.path.exists(indice_folder+"/text.index")
        data_dir = Path(indice_folder+"/metadata")
        metadata_df = pd.concat(
            pd.read_parquet(parquet_file)
            for parquet_file in data_dir.glob('*.parquet')
        )

        url_list = None
        if "url" in metadata_df:
            url_list = metadata_df["url"].tolist()
        if image_present:
            image_list = metadata_df["image_path"].tolist()
            image_index = faiss.read_index(indice_folder+"/image.index")
        else:
            image_list = None
            image_index = None
        if text_present:
            description_list = metadata_df["caption"].tolist()
            text_index = faiss.read_index(indice_folder+"/text.index")
        else:
            description_list = None
            text_index = None
        indices_loaded[name]={
            'image_list': image_list,
            'url_list': url_list,
            'description_list': description_list,
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
