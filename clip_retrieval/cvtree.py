import fire
import faiss
import numpy as np
import covertree

from pathlib import Path

def clip_benchmark(img_embeds: Path, text_embeds: Path):

    n = 100000000
    img_embeds = np.load(img_embeds).astype(np.float32)[:n]
    text_embeds = np.load(text_embeds).astype(np.float32)[:n]
    print(np.unique(img_embeds, axis=0).shape)
    img_tree = covertree.CoverTree.from_matrix(img_embeds.astype(np.float32))
    text_tree = covertree.CoverTree.from_matrix(text_embeds.astype(np.float32))

    img_idxs = img_tree.NearestNeighbor(text_embeds)
    text_idxs = text_tree.NearestNeighbor(img_embeds)
    gt = np.arange(len(img_embeds))[:, None]

    # img_dists, img_idxs = img_index.search(text_embeds, 5)
    # text_dists, text_idxs = text_index.search(img_embeds, 5)
    print("knn(img) == knn(text)", (img_idxs == text_idxs).sum())
    print(img_idxs.shape)
    print(text_idxs)
    print((img_idxs == gt).any(axis=-1).sum() / len(img_embeds))
    print((text_idxs == gt).any(axis=-1).sum() / len(img_embeds))
    print(img_embeds.shape)
if __name__ == "__main__":
    fire.Fire(clip_benchmark)
