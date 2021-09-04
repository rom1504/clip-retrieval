import clip
import faiss
import torch
import json
import os
import shutil
import fire
from pathlib import Path
import pandas as pd


def clip_filter(query, output_folder, indice_folder, num_results=100, threshold=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device, jit=False)

    data_dir = Path(indice_folder+"/metadata")
    df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in sorted(data_dir.glob('*.parquet'))
    )

    url_list = None
    if "url" in df:
        url_list = df["url"].tolist()

    image_list = df["image_path"].tolist()
    image_index = faiss.read_index(indice_folder+"/image.index")
    indices_loaded={
        'image_list': image_list,
        'image_index': image_index,
    }

    text_input = query
    image_index = indices_loaded["image_index"]
    image_list = indices_loaded["image_list"]
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
    print("You may want to use these numbers to increase your --num_results parameter. Or use the --threshold parameter.")

    print(f"Copying the images in {output_folder}")

    for _, i in zip(D, I):
        path = image_list[i]
        if os.path.exists(path):
            shutil.copy(path, output_folder)
        if url_list is not None:
            print(url_list[i])


if __name__ == '__main__':
  fire.Fire(clip_filter)
