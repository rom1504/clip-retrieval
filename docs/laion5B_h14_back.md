# How to setup clip-back with an H/14 index of Laion5B

1. Create a python virtual environment & install huggingface-cli & clip-retrieval
    - `pip install huggingface-cli clip-retrieval`
2. Install git-lfs on your system
    - https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage
3. run `git-lfs install`
4. Navigate to your large storage
    - `cd /somehwere/with/lots/of/space`
5. Clone the repository 
    - `git clone https://huggingface.co/datasets/laion/laion5b-h14-index`
6. cd into repository
    - `cd laion5b-h14-index`
7. Download the index files with git-lfs
    - `git-lfs fetch -I index-parts` <- *this will take quite a while*
    - `git-lfs checkout index-parts/*` <- *this will take quite a while*
8. Combine the index parts using the following command
    - `clip_retrieval index_combiner --input_folder "the/indices" --output_folder "output"`

9. Now repeat steps 5-7 with the following metadata repos
    - https://huggingface.co/datasets/laion/laion2b-multi-vit-h-14-embeddings
    - https://huggingface.co/datasets/laion/laion2b-en-vit-h-14-embeddings
    - https://huggingface.co/datasets/laion/laion1b-nolang-vit-h-14-embeddings
    - > NOTE: you will simply be replacing the target folder for each of the git-lfs commands
    - `Example: cd <meta-repo> && git-lfs fetch -I metadata/`

10. Now run the metadata combiner for each of the metadata folders
    - `clip-retrieval parquet_to_arrow --parquet_folder "/" --output_arrow_folder "/media/nvme/large_index/metadata/2B-en"\ --columns_to_return='["url", "caption"]'`

11. Create a parent directory to hold all of the index information
    - `mkdir Laion5B_H14 && mkdir Laion5B_H14/metadata && mkdir Laion5B_H14/image.index`
12. Move all of the metadata `arrow files` to the metadata subfolder of our new parent folder
    - NOTE: in order to maintain the proper ordering, it is recommended to use the following file names
    - 0_en.arrow
    - 1_multi.arrow
    - 2_nolang.arrow
13. Move the files generated from the index combination step into the `image.index` subfolder
15. Create an indices.json file with the following (edit as necessary, more info on parameters in the [Main README](https://github.com/rom1504/clip-retrieval#clip-back))
16. # TODO: safety model
```
{
        "laion5B-H-14": {
                "indice_folder": "laion5B_H14",
                "provide_safety_model": true,
                "enable_faiss_memory_mapping": true,
                "use_arrow": true,
                "enable_hdf5": false,
                "reorder_metadata_by_ivf_index": false,
                "columns_to_return": ["url", "caption"],
                "clip_model": "open_clip:ViT-H-14",
                "enable_mclip_option": false
        }
}
```

