from pathlib import Path

import numpy as np
from tqdm import tqdm
from fire import Fire
from faiss import IndexFlatIP
from webdataset import WebDataset, WebLoader
from sentence_transformers import SentenceTransformer


def get_sentence_embs(path: Path, dataset: Path):
    if Path(path).exists():
        return np.load(path)
    else:
        ds = WebDataset("/home/a/mscoco/{00000..00059}.tar").to_tuple("txt")
        loader = WebLoader(ds, batch_size=2048, collate_fn=list)
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        sentence_embs = np.concatenate([model.encode([b[0] for b in batch]) for batch in tqdm(loader)], axis=0)
        np.save(path, sentence_embs)
        return sentence_embs


def clip_benchmark(img_embeds_file: Path,
                   text_embeds_file: Path,
                   sentence_embs: Path,
                   n=30000,
                   dataset="/home/a/mscoco{00000..00059}.tar"):


    sentence_embs = get_sentence_embs(sentence_embs, dataset)

    img_embeds = np.load(img_embeds_file)[:n].astype(np.float32)
    gt = np.arange(len(img_embeds))[:, None]
    text_embeds = np.load(text_embeds_file)[:n].astype(np.float32)

    unique_img_embeds, unique_imgs, unique_img_idx = np.unique(img_embeds, return_index=True, return_inverse=True, axis=0)
    unique_text_embeds, unique_texts, unique_text_idx = np.unique(text_embeds, return_index=True, return_inverse=True, axis=0)
    print("num unique img embeds", len(unique_img_embeds))
    print("num unique text embeds", len(unique_text_embeds))
    img_gt = gt[:len(unique_img_embeds)]
    text_gt = gt[:len(unique_text_embeds)]


    img_brute_index = IndexFlatIP(unique_img_embeds.shape[1])
    img_brute_index.add(unique_img_embeds)

    text_brute_index = IndexFlatIP(unique_text_embeds.shape[1])
    text_brute_index.add(unique_text_embeds)
    print("searching brute force")

    img_dists, img_idxs = img_brute_index.search(text_embeds[unique_imgs], 5)
    text_dists, text_idxs = text_brute_index.search(img_embeds[unique_texts], 5)

    print("text->img retrieval @5", (img_idxs == img_gt).any(axis=-1).sum() / len(img_idxs))
    print("img->text retrieval @5", (text_idxs == text_gt).any(axis=-1).sum() / len(text_idxs))

    text_sim = (sentence_embs[unique_imgs[img_idxs]] * sentence_embs[unique_imgs][:, None, :]).sum(axis=-1)
    print("text->img similarity @5:", text_sim.max(1).mean(), text_sim.max(1).std())
    print("text->img similarity @1:", text_sim[:, 0].mean(), text_sim[:, 0].std())

    text_sim = (sentence_embs[unique_texts[text_idxs]] * sentence_embs[unique_texts][:, None, :]).sum(axis=-1)
    print("img->text similarity @5:", text_sim.max(1).mean(), text_sim.max(1).std())
    print("img->img similarity @1:", text_sim[:, 0].mean(), text_sim[:, 0].std())



if __name__ == "__main__":
    Fire(clip_benchmark)
