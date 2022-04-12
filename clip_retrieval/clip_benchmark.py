import fire
import faiss
import numpy as np

from pathlib import Path

def clip_benchmark(img_embeds: Path, img_index: Path,
                   text_embeds: Path, text_index: Path):

    img_index = faiss.read_index(img_index)
    text_index = faiss.read_index(text_index)

    n = 100000000
    img_embeds = np.load(img_embeds).astype(np.float32)[:n]
    text_embeds = np.load(text_embeds).astype(np.float32)[:n]
    gt = np.arange(len(img_embeds))[:, None]

    img_dists, img_idxs = img_index.search(text_embeds, 5)
    text_dists, text_idxs = text_index.search(img_embeds, 5)
    print("knn(img) == knn(text)", (img_idxs == text_idxs).sum())
    print((img_idxs == gt).any(axis=-1).sum() / len(img_embeds))
    print((text_idxs == gt).any(axis=-1).sum() / len(img_embeds))
    print(img_embeds.shape)
if __name__ == "__main__":
    fire.Fire(clip_benchmark)
