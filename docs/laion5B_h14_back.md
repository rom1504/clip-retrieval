# How to setup clip-back with an H/14 index of Laion5B

1. Create a python virtual environment & install huggingface_hub & clip-retrieval
   - `pip install huggingface_hub clip-retrieval`
2. Install `aria2` on your system
   https://github.com/aria2/aria2
3. Navigate to your large storage
   - `cd /somehwere/with/lots/of/space`
4. Download the index parts from the hugging-face repository
   - `mkdir index-parts && cd index-parts`
   - `for i in {00..79}; do aria2c -x 16 https://huggingface.co/datasets/laion/laion5b-h14-index/resolve/main/index-parts/$i.index -o $i.index; done`
   - `cd ..`
5. Combine the index parts using the following command
   - `clip-retrieval index_combiner --input_folder "index-parts" --output_folder "combined-indices"`
6. Now download the metadata parts from the following metadata repos

   - ***multi embeddings***
        - `mkdir multi-embeddings && cd multi-embeddings`
        - `for i in {0000..2268}; do aria2c -x 16 https://huggingface.co/datasets/laion/laion2b-multi-vit-h-14-embeddings/resolve/main/metadata/metadata_$i.parquet -o metadata_$i.parquet; done`
        - `cd ..`
   - ***english embeddings***
        - `mkdir en-embeddings && cd en-embeddings`
        - `for i in {0000..2313}; do aria2c -x 16 https://huggingface.co/datasets/laion/laion2b-en-vit-h-14-embeddings/resolve/main/metadata/metadata_$i.parquet -o metadata_$i.parquet; done`
        - `cd ..`
   - ***nolang embeddings***
        - `mkdir nolang-embeddings && nolang en-embeddings`
        - `for i in {0000..1273}; do aria2c -x 16 https://huggingface.co/datasets/laion/laion1b-nolang-vit-h-14-embeddings/resolve/main/metadata/metadata_$i.parquet -o metadata_$i.parquet; done`
        - `cd ..`

7. Now run the metadata combiner for each of the metadata folders  (Warning: ensure all metadata parquet files are present before combining them, or the combined arrow file may be misaligned with the index)

   - ***multi embeddings***
        - `clip-retrieval parquet_to_arrow --parquet_folder="multi-embeddings" --output_arrow_folder="multi-combined" --columns_to_return='["url", "caption"]'`
   - ***english embeddings***
        - `clip-retrieval parquet_to_arrow --parquet_folder="en-embeddings" --output_arrow_folder="en-combined" --columns_to_return='["url", "caption"]'`
   - ***nolang embeddings***
        - `clip-retrieval parquet_to_arrow --parquet_folder="nolang-embeddings" --output_arrow_folder="nolang-combined" --columns_to_return='["url", "caption"]'`

8. Create a parent directory to hold all of the index information
   - `mkdir Laion5B_H14 && mkdir Laion5B_H14/metadata && mkdir Laion5B_H14/image.index`
9. Move all of the metadata `arrow files` to the metadata subfolder of our new parent folder
   > **NOTE: in order to maintain the proper ordering, it is important to use the following file names**
   - `mv en-combined/0.arrow Laion5B_H14/metadata/0_en.arrow`
   - `mv multi-combined/0.arrow Laion5B_H14/metadata/1_multi.arrow`
   - `mv nolang-combined/0.arrow Laion5B_H14/metadata/2_nolang.arrow`
10. Move the files generated from the index combination step into the `image.index` subfolder
    - `mv combined-indices/* Laion5B_H14/image.index/`
11. Create an indices.json file with the following (edit as necessary, more info on parameters in the [Main README](https://github.com/rom1504/clip-retrieval#clip-back))

```
{
        "laion5B-H-14": {
                "indice_folder": "Laion5B_H14",
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
