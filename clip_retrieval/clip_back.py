from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
import clip
import faiss
import torch
import json
import sys
from PIL import Image
from io import BytesIO
from PIL import Image
import base64
import os
import fire


class Health(Resource):
    def get(self):
        return "ok"

class IndicesList(Resource):
    def get(self, **kwargs):
        return list(kwargs['indices'].keys())

class KnnService(Resource):
    def post(self, **kwargs):
        indices_loaded = kwargs['indices_loaded']
        device = kwargs['device']
        model = kwargs['model']
        preprocess = kwargs['preprocess']
        json_data = request.get_json(force=True)
        text_input = json_data["text"] if "text" in json_data else None
        image_input = json_data["image"] if "image" in json_data else None
        modality = json_data["modality"]
        num_images = json_data["num_images"]
        indice_name = json_data["indice_name"]
        image_index = indices_loaded[indice_name]["image_index"]
        text_index = indices_loaded[indice_name]["text_index"]
        image_list = indices_loaded[indice_name]["image_list"]
        description_list = indices_loaded[indice_name]["description_list"]

        if text_input is not None:
            text = clip.tokenize([text_input]).to(device)
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            query = text_features.cpu().detach().numpy().astype("float32")
        if image_input is not None:
            binary_data = base64.b64decode(image_input)
            img_data = BytesIO(binary_data)
            img = Image.open(img_data)
            prepro = preprocess(img).unsqueeze(0)
            image_features = model.encode_image(prepro)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            query = image_features.cpu().detach().numpy().astype("float32")
        
        index = image_index if modality == "image" else text_index

        D, I = index.search(query, num_images)
        results = []
        for d, i in zip(D[0], I[0]):
            path = image_list[i]
            description = description_list[i]
            img = Image.open(path)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8") 
            results.append({"image": img_str, "text": description})
        return results


def clip_back(indices_paths="indices_paths.json", port=1234):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    indices = json.load(open(indices_paths))

    indices_loaded = {}

    for name, indice_folder in indices.items():
        image_present = os.path.exists(indice_folder+"/image_list")
        text_present = os.path.exists(indice_folder+"/description_list")
        if image_present:
            with open(indice_folder+"/image_list") as f:
                image_list = [x for x in f.read().split("\n") if x !=""]
            image_index = faiss.read_index(indice_folder+"/image.index")
        else:
            image_list = None
            image_index = None
        if text_present:
            with open(indice_folder+"/description_list") as f:
                description_list = [x for x in f.read().split("\n") if x !=""]
            text_index = faiss.read_index(indice_folder+"/text.index")
        else:
            description_list = None
            text_index = None
        indices_loaded[name]={
            'image_list': image_list,
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
