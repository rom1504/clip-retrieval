import fire
import faiss
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors, NeighborhoodComponentsAnalysis as NCA
from mlpack import knn
from pathlib import Path


def clip_benchmark(img_embeds: Path, img_index: Path,
                   text_embeds: Path, text_index: Path):

    img_index = faiss.read_index(img_index)
    text_index = faiss.read_index(text_index)
    print(img_index)
    n = 500000
    print(np.linalg.norm(np.load(img_embeds), axis=-1))

    img_embeds = np.load(img_embeds)[:n].astype(np.float32)
    gt = np.arange(len(img_embeds))[:, None]



    text_embeds = np.load(text_embeds)[:n].astype(np.float32)
    print(text_embeds.dtype)
    unique_img_embeds, unique_imgs, unique_img_idx = np.unique(img_embeds, return_index=True, return_inverse=True, axis=0)
    unique_text_embeds, unique_texts, unique_text_idx = np.unique(text_embeds, return_index=True, return_inverse=True, axis=0)
    print("num unique img embeds", len(unique_img_embeds))
    img_gt = gt[:len(unique_img_embeds)]
    text_gt = gt[:len(unique_text_embeds)]

    print("searching faiss")
    img_dists, img_idxs = img_index.search(text_embeds[unique_imgs], 5)
    text_dists, text_idxs = text_index.search(img_embeds[unique_texts], 5)

    print((unique_img_idx[img_idxs] == img_gt).any(axis=-1).sum() / len(img_idxs))
    print((unique_text_idx[text_idxs] == text_gt).any(axis=-1).sum() / len(text_idxs))


    print("trying covertree")
    img_res = knn(tree_type='cover', query=text_embeds[unique_imgs], reference=unique_img_embeds, k=5, epsilon=None)
    text_res = knn(tree_type='cover', query=img_embeds[unique_texts], reference=unique_text_embeds, k=5, epsilon=None)
    img_idxs = img_res["neighbors"]
    text_idxs = text_res["neighbors"]

    print((img_idxs == img_gt).any(axis=-1).sum() / len(img_idxs))
    print((text_idxs == text_gt).any(axis=-1).sum() / len(text_idxs))




    print("fitting index")
    img_ball_index = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(unique_img_embeds)
    text_ball_index = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(unique_text_embeds)

    print("searching ball")

    img_dists, img_idxs = img_ball_index.kneighbors(text_embeds[unique_imgs], 5)
    text_dists, text_idxs = text_ball_index.kneighbors(img_embeds[unique_texts], 5)

    print((img_idxs == img_gt).any(axis=-1).sum() / len(img_idxs))
    print((text_idxs == text_gt).any(axis=-1).sum() / len(text_idxs))
    print(img_embeds.shape)

if __name__ == "__main__":
    fire.Fire(clip_benchmark)
