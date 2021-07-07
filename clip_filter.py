import clip
import faiss
import torch
import json
import os
import shutil
import fire

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

indices = json.load(open("indices_paths.json"))

indices_loaded = {}

for name, indice_folder in indices.items():
    image_present = os.path.exists(indice_folder+"/image_list")
    with open(indice_folder+"/image_list") as f:
        image_list = f.read().split("\n")
    image_index = faiss.read_index(indice_folder+"/image.index")
    indices_loaded[name]={
        'image_list': image_list,
        'image_index': image_index,
    }

def main(query, output_folder, indice_name, num_results=100, threshold=None):
    text_input = query
    indice_name = indice_name
    image_index = indices_loaded[indice_name]["image_index"]
    image_list = indices_loaded[indice_name]["image_list"]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    text = clip.tokenize([text_input]).to(device)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    query = text_features.cpu().detach().numpy().astype("float32")
    
    index = image_index

    if threshold is not None:
        _, D, I = index.range_search(query, threshold)
        print(f"Found {I.shape} items with query '{text_input}' and threshold {threshold}")
    else:
        D, I = index.search(query, num_results)
        print(f"Found {num_results} items with query '{text_input}'")
        I = I[0]
        D = D[0]
    
    min_D = min(D)
    max_D = max(D)
    print(f"The minimum distance is {min_D:.2f} and the maximum is {max_D:.2f}")
    print("You may want to use these numbers to increase your --num_images parameter. Or use the --threshold parameter.")
    print(f"Copying the images in {output_folder}")

    for _, i in zip(D, I):
        path = image_list[i]
        shutil.copy(path, output_folder)


if __name__ == '__main__':
  fire.Fire(main)
