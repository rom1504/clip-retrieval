"""clip filter is a tool to use a knn index and a image/text collection to extract interesting subsets"""


import fire


def clip_filter(query, output_folder, indice_folder, num_results=100, threshold=None):
    """Entry point of clip filter"""

    import faiss  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    import os  # pylint: disable=import-outside-toplevel
    import shutil  # pylint: disable=import-outside-toplevel
    from pathlib import Path  # pylint: disable=import-outside-toplevel
    import pandas as pd  # pylint: disable=import-outside-toplevel
    import clip  # pylint: disable=import-outside-toplevel
    from PIL import Image  # pylint: disable=import-outside-toplevel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    data_dir = Path(indice_folder + "/metadata")
    df = pd.concat(pd.read_parquet(parquet_file) for parquet_file in sorted(data_dir.glob("*.parquet")))

    url_list = None
    if "url" in df:
        url_list = df["url"].tolist()

    image_list = df["image_path"].tolist()
    image_index = faiss.read_index(indice_folder + "/image.index")
    indices_loaded = {
        "image_list": image_list,
        "image_index": image_index,
    }

    image_index = indices_loaded["image_index"]
    image_list = indices_loaded["image_list"]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if query.endswith((".png", ".jpg", ".jpeg", ".bmp")) and os.path.isfile(query):
        im = Image.open(query)
        query_features = model.encode_image(preprocess(im).unsqueeze(0).to(device))
    else:
        text = clip.tokenize([query]).to(device)
        query_features = model.encode_text(text)
    query_features /= query_features.norm(dim=-1, keepdim=True)
    query_features = query_features.cpu().detach().numpy().astype("float32")

    index = image_index

    if threshold is not None:
        _, d, i = index.range_search(query_features, threshold)
        print(f"Found {i.shape} items with query '{query}' and threshold {threshold}")
    else:
        d, i = index.search(query_features, num_results)
        print(f"Found {num_results} items with query '{query}'")
        i = i[0]
        d = d[0]

    min_d = min(d)
    max_d = max(d)
    print(f"The minimum distance is {min_d:.2f} and the maximum is {max_d:.2f}")
    print(
        "You may want to use these numbers to increase your --num_results parameter. Or use the --threshold parameter."
    )

    print(f"Copying the images in {output_folder}")

    for _, ei in zip(d, i):
        path = image_list[ei]
        if os.path.exists(path):
            shutil.copy(path, output_folder)
        if url_list is not None:
            print(url_list[ei])


if __name__ == "__main__":
    fire.Fire(clip_filter)
