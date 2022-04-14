import fire
import faiss
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors, NeighborhoodComponentsAnalysis as NCA
from mlpack import knn
from pathlib import Path
from covertree import CoverTree
from webdataset import WebDataset, WebLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
def get_knn(x):
    model, q, k = x
    return knn(query=q, input_model=model, k=k)['neighbors']

def intersect_2d(x, y):
    _, dim = x.shape
    print(x.shape, y.shape)
    # view [_, dim] array as a 1d array with a dim-wide struct
    dtype = dict(names=[f'f{i}' for i in range(dim)],
                 formats=dim * [x.dtype])

    inter, idxs, _ = np.intersect1d(x.view(dtype), y.view(dtype),
                                return_indices=True)

    return idxs

def par_knn_cv(embeds, query, k):
    query = query.astype(np.float64)
    embeds = embeds.astype(np.float64)
    tree = CoverTree.from_matrix(embeds)
    res = tree.kNearestNeighbours(query, k)
    print(res.dtype, res.shape)
    return intersect_2d(embeds, res.reshape(k * query.shape[0], query.shape[-1]))

def par_knn_mlpack(embeds, query, k):
    from concurrent.futures import ProcessPoolExecutor
    print("fitting covertree")
    d = knn(tree_type='cover', reference=embeds, k=k)
    print("searching covertree")

    with ProcessPoolExecutor(max_workers=8) as executor:
        processed_chunks = executor.map(get_knn,
                                        [(d["output_model"], query[i:i+1024], k) for i in range(0, len(query), 1024)])
        processed_chunks = [*processed_chunks]
    return np.concatenate(processed_chunks, axis=0)


def par_knn_faiss(index, query, k):
    dists, idxs = index.search(query.astype(np.float32))[1]
    return idxs

def get_sentence_embs(path: Path):
    if Path(path).exists():
        return np.load(path)
    else:
        ds = WebDataset("/home/a/mscoco/{00000..00059}.tar").to_tuple("txt")
        loader = WebLoader(ds, batch_size=2048, collate_fn=list)
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        sentence_embs = np.concatenate([model.encode([b[0] for b in batch]) for batch in tqdm(loader)], axis=0)
        np.save(path, sentence_embs)
        return sentence_embs

def clip_benchmark(img_embeds_file: Path, img_index: Path,
                   text_embeds_file: Path, text_index: Path,
                   sentence_embs: Path,
                   n=30000):



    sentence_embs = get_sentence_embs(sentence_embs)

    img_embeds = np.load(img_embeds_file)[:n].astype(np.float32)
    gt = np.arange(len(img_embeds))[:, None]
    text_embeds = np.load(text_embeds_file)[:n].astype(np.float32)

    unique_img_embeds, unique_imgs, unique_img_idx = np.unique(img_embeds, return_index=True, return_inverse=True, axis=0)
    unique_text_embeds, unique_texts, unique_text_idx = np.unique(text_embeds, return_index=True, return_inverse=True, axis=0)
    print("num unique img embeds", len(unique_img_embeds))
    print("num unique text embeds", len(unique_text_embeds))
    img_gt = gt[:len(unique_img_embeds)]
    text_gt = gt[:len(unique_text_embeds)]

    # TODO: port over feature ids tracking from old cvtree to here
    # print("trying covertree")
    # img_idxs = par_knn_cv(query=text_embeds[unique_imgs], embeds=unique_img_embeds, k=5)
    # text_idxs = par_knn_cv(query=img_embeds[unique_texts], embeds=unique_text_embeds, k=5)
    # print(img_idxs.shape, text_idxs.shape)
    # print((img_idxs == img_gt).any(axis=-1).sum() / len(img_idxs))
    # print((text_idxs == text_gt).any(axis=-1).sum() / len(text_idxs))


    print("trying covertree")
    img_idxs = par_knn_mlpack(query=text_embeds[unique_imgs], embeds=unique_img_embeds, k=5)
    text_idxs = par_knn_mlpack(query=img_embeds[unique_texts], embeds=unique_text_embeds, k=5)
    print((img_idxs == img_gt).any(axis=-1).sum() / len(img_idxs))
    print((text_idxs == text_gt).any(axis=-1).sum() / len(text_idxs))

    text_sim = (sentence_embs[unique_texts[text_idxs[:, 0]]] * sentence_embs[unique_texts]).sum(axis=-1)
    print("text similarity:", text_sim.mean(), text_sim.std())


    img_index = faiss.read_index(img_index)
    text_index = faiss.read_index(text_index)
    print(img_index)

    unique_img_embeds, unique_imgs, unique_img_idx = np.unique(img_embeds, return_index=True, return_inverse=True, axis=0)
    unique_text_embeds, unique_texts, unique_text_idx = np.unique(text_embeds, return_index=True, return_inverse=True, axis=0)

    img_gt = gt[:len(unique_img_embeds)]
    text_gt = gt[:len(unique_text_embeds)]
    img_embeds = np.load(img_embeds_file).astype(np.float32)
    gt = np.arange(len(img_embeds))[:, None]
    text_embeds = np.load(text_embeds_file).astype(np.float32)

    print("searching faiss")
    img_dists, img_idxs = img_index.search(text_embeds[unique_imgs], 5)
    text_dists, text_idxs = text_index.search(img_embeds[unique_texts], 5)

    print((unique_img_idx[img_idxs] == img_gt).any(axis=-1).sum() / len(img_idxs))
    print((unique_text_idx[text_idxs] == text_gt).any(axis=-1).sum() / len(text_idxs))

    text_sim = (sentence_embs[text_idxs] * sentence_embs[unique_texts]).sum()
    print("text similarity:", text_sim.mean(), text_sim.std())


    print("fitting ball tree")
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
