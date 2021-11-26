"""clip end2end combines img2dataset, inference, index, back and front to produce a retrieval system in one command"""

import fire


def clip_end2end(url_list, output_folder, run_back=True):
    """main entry point of clip end2end"""

    import os  # pylint: disable=import-outside-toplevel
    from img2dataset import download  # pylint: disable=import-outside-toplevel
    from clip_retrieval import clip_inference  # pylint: disable=import-outside-toplevel
    from clip_retrieval import clip_index  # pylint: disable=import-outside-toplevel
    from clip_retrieval import clip_back  # pylint: disable=import-outside-toplevel
    import fsspec  # pylint: disable=import-outside-toplevel

    fs, output_folder_in_fs = fsspec.core.url_to_fs(output_folder)
    print(output_folder_in_fs)
    if not fs.exists(output_folder_in_fs):
        fs.mkdir(output_folder_in_fs)
    image_folder_name = os.path.join(output_folder, "images")
    embeddings_folder = os.path.join(output_folder, "embeddings")
    index_folder = os.path.join(output_folder, "index")
    # img2dataset
    download(
        url_list,
        image_size=256,
        output_folder=image_folder_name,
        thread_count=128,
        processes_count=4,
        input_format="parquet",
        output_format="webdataset",
        url_col="URL",
        caption_col="TEXT",
    )
    # Clip inference
    input_files = [image_folder_name + "/" + p for p in next(fs.walk(image_folder_name))[2] if p.endswith(".tar")]
    clip_inference(
        input_dataset=input_files,
        output_folder=embeddings_folder,
        input_format="webdataset",
        enable_metadata=True,
        write_batch_size=100000,
        batch_size=512,
        cache_path=None,
    )
    # Clip index
    os.mkdir(index_folder)
    clip_index(embeddings_folder, index_folder=index_folder)

    # Clip back
    indice_path = os.path.join(output_folder, "indices_paths.json")
    with fsspec.open(indice_path, "w") as f:
        f.write('{"example_index": "' + index_folder + '"}')
    if run_back:
        clip_back(port=1234, indices_paths=indice_path)


if __name__ == "__main__":
    fire.Fire(clip_end2end)
