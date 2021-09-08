# clip-retrieval
[![pypi](https://img.shields.io/pypi/v/clip-retrieval.svg)](https://pypi.python.org/pypi/clip-retrieval)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rom1504/clip-retrieval/blob/master/notebook/clip-retrieval-getting-started.ipynb)
[![Try it on gitpod](https://img.shields.io/badge/try-on%20gitpod-brightgreen.svg)](https://gitpod.io/#https://github.com/rom1504/clip-retrieval)

Easily compute clip embeddings and build a clip retrieval system with them. 100M text+image embeddings can be processed in 20h using a 3080.

* clip inference allows you to quickly (1500 sample/s on a 3080) compute image and text embeddings
* clip index builds efficient indices out of the embeddings
* clip filter allows you to filter out the data using the clip index
* clip back hosts the indices with a simple flask service
* clip front is a simple ui querying the back. Check it out at [clip-retrieval ui](https://rom1504.github.io/clip-retrieval/)

End to end this make it possible to build a simple semantic search system.
Interested to learn about semantic search in general ? You can read my [medium post](https://rom1504.medium.com/semantic-search-with-embeddings-index-anything-8fb18556443c) on the topic.

[<img src="./clip-front-pic.png" alt="viewer" width="500">](https://rom1504.github.io/clip-retrieval/)

## Who is using clip retrieval ?

* [cah-prepro](https://github.com/rom1504/cah-prepro) preprocess the 400M image+text crawling at home dataset. clip-retrieval is used to compute 400M clip embeddings and the indices
* [autofaiss](https://github.com/criteo/autofaiss) uses clip-retrieval to display an example of use (see the multimodal notebook example there)
* [afiaka87 openai demo](https://gist.github.com/afiaka87/f662486fc45199fa4394f3456c8246d7#file-dalle_blog_semantic_search-ipynb) shows how to look among the 1M example released by openai for their DALL-E demo
* [antarctic-captions by dzryk](https://github.com/dzryk/antarctic-captions) uses autofaiss and clip inference as a way to generate anchors for the image to text task with great success

## Install

pip install clip-retrieval

## clip inference

Get some images in an `example_folder`, for example by doing:
```
pip install img2dataset
echo 'https://placekitten.com/200/305' >> myimglist.txt
echo 'https://placekitten.com/200/304' >> myimglist.txt
echo 'https://placekitten.com/200/303' >> myimglist.txt
img2dataset --url_list=myimglist.txt --output_folder=image_folder --thread_count=64 --image_size=256
```
You can also put text files with the same names as the images in that folder, to get the text embeddings.

Then run `clip-retrieval inference --input_dataset image_folder --output_folder embeddings_folder`

Output folder will contain:
* img_emb/
    * img_emb_0.npy containing the image embeddings as numpy
* text_emb/
    * text_emb_0.npy containing the text embeddings as numpy
* metadata/
    * metadata_0.parquet containing the image paths, captions and metadata

This scales to million of samples. At 1400 sample/s of a 3080, 10M samples can be processed in 2h.

### API

clip_inference turn a set of text+image into clip embeddings

* **input_dataset** Path to input dataset. Folder if input_format is files. Bash brace pattern such as "{000..150}.tar" (see https://pypi.org/project/braceexpand/) if webdataset (*required*)
* **output_folder** Folder where the clip embeddings will be saved, as well as metadata (*required*)
* **input_format** files or webdataset (default *files*)
* **cache_path** cache path for webdataset (default *None*)
* **batch_size** Number of items to do the inference on at once (default *256*)
* **num_prepro_workers** Number of processes to do the preprocessing (default *8*)
* **enable_text** Enable text processing (default *True*)
* **enable_image** Enable image processing (default *True*)
* **enable_metadata** Enable metadata processing (default *False*)
* **write_batch_size** Write batch size (default *10**6*)
* **subset_size** Only process a subset of this size (default *None*)
* **wds_image_key** Key to use for images in webdataset. (default *jpg*)
* **wds_caption_key** Key to use for captions in webdataset. (default *txt*)

## Clip index

Clip index takes as input the output of clip inference and makes an index out of it using [autofaiss](https://github.com/criteo/autofaiss)

`clip-retrieval index --input_folder embeddings_folder --output_folder index_folder`

An additional `--max_index_memory_usage "4G"` option allow configuring the amount of ram the index will consume. More ram, better knn recall.
`--current_memory_available 16G` allows controlling how much ram is used during the creation process.

The output is a folder containing:
* image.index containing a brute force faiss index for images
* text.index containing a brute force faiss index for texts
* metadata folder containing the parquet metadata

Thanks to autofaiss and faiss, this scales to hundred of million of samples in a few hours.

## Clip filter

Once the embeddings are computed, you may want to filter out the data by a specific query.
For that you can run `clip-retrieval filter --query "cat" --output_folder "cat/" --indice_folder "indice_folder"`
It will copy the 100 best images for this query in the output folder.
Using the `--num_results` or `--threshold` may be helpful to refine the filter

Thanks to fast knn index, this can run in real time (<10ms) for large K values (100000), and in minutes for very large K values.

## Clip back

Then run (output_folder is the output of clip index)
```bash
echo '{"example_index": "output_folder"}' > indices_paths.json
clip-retrieval back --port 1234 --indices-paths indices_paths.json
```

A `--enable_hdf5 True` option can be passed to enable hdf5 caching for the metadata.
HDF5 caching makes it possible to use the metadata with almost no memory usage.

hdf5 caching is a good idea to use if:
* you do not have enough ram to load the metadata in memory
* your disk is fast (ie you have a ssd)

At this point you have a simple flask server running on port 1234 and that can answer these queries:

* `/indices-list` -> return a list of indices
* `/knn-service` that takes as input:
```js
{
    "text": "a text query",
    "image": "a base64 image",
    "modality": "image", // image or text index to use
    "num_images": 4, // number of output images
    "indice_name": "example_index"
}
```
and returns:
```js
[
    {
        "image": "base 64 of an image",
        "text": "some result text"
    },
    {
        "image": "base 64 of an image",
        "text": "some result text"
    }
]
```

This achieve low latency status (10ms). Throughput is about 100 query/s. For high throughput, using a grpc server is required.

## clip-front

Clip front is a simple UI that connects to clip back and display the results.
To run it, go to [front](front) and run `npm install` then `npm start`.
You can also directly use it at [clip-retrieval ui](https://rom1504.github.io/clip-retrieval/)

## For development

Either locally, or in [gitpod](https://gitpod.io/#https://github.com/rom1504/img2dataset) (do `export PIP_USER=false` there)

Setup a virtualenv:

```
python3 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -e .
```

to run tests:
```
pip install -r requirements-test.txt
```
then 
```
python -m pytest -v tests -s
```
