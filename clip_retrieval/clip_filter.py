import clip
import faiss
import torch
import json
import os
import shutil
import fire


def clip_filter(query, output_folder, indice_folder, num_results=100, threshold=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device, jit=False)

    with open(indice_folder+"/image_list") as f:
        image_list = [x for x in f.read().split("\n") if x !=""]
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
        shutil.copy(path, output_folder)


if __name__ == '__main__':
  fire.Fire(clip_filter)
