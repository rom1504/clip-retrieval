# clip-retrieval
[![pypi](https://img.shields.io/pypi/v/clip-retrieval.svg)](https://pypi.python.org/pypi/clip-retrieval)
[![NPM version](https://badge.fury.io/js/clip-retrieval-front.svg)](http://badge.fury.io/js/clip-retrieval-front)
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

[<img src="https://github.com/rom1504/clip-retrieval/raw/main/doc_assets/clip-front-pic.png" alt="clip front" width="500">](https://rom1504.github.io/clip-retrieval/)

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

* `--max_index_memory_usage "4G"` option allow configuring the amount of ram the index will consume. More ram, better knn recall.
* `--current_memory_available 16G` allows controlling how much ram is used during the creation process.
* `--copy_metadata True` makes it possible to choose whether to copy metadata or not at the end of the process.
* `--nb_cores 8` allows controlling the number of threads 

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

Clip back is a simple knn service backend. If using both hdf5 and faiss memory mapping, it uses only the memory used by clip which is 4GB.

Run (output_folder is the output of clip index)
```bash
echo '{"example_index": "output_folder"}' > indices_paths.json
clip-retrieval back --port 1234 --indices-paths indices_paths.json
```

`--columns_to_return='["url", "image_path", "caption", "NSFW"]` allows you to specify which columns should be fetched from the metadata and returned by the backend. It's useful to specify less in case of hdf5 caching to speed up the queries.

A `--enable_faiss_memory_mapping=True` option can be passed to use an index with memory mapping.
That decreases the memory usage to zero.

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
    "image_url": "http://some-url.com/a.jpg",
    "modality": "image", // image or text index to use
    "num_images": 4, // number of output images
    "indice_name": "example_index",
    "num_result_ids": 4 // optional, if specified fetch this number of results in total but only num_images with metadata
}
```
text, image and image_url are mutually exclusive
and returns:
```js
[
    {
        "image": "base 64 of an image",
        "text": "some result text",
        "id": 543
    },
    {
        "image": "base 64 of an image",
        "text": "some result text",
        "id": 782
    }
]
```
Each object may also contain an url field if the metadata provides it.

The id is the position of the item in the index. It may be used to query metadata with the /metadata endpoint:
```js
{
    "indice_name": "example_index",
    "ids": [543, 782]
}
```
which returns:
```js
{
    "image": "base 64 of an image",
    "text": "some result text"
    // any other key available in the metadata and specified in columns_to_return cli option
}
```

`num_result_ids` argument of `/knn-service` and `/metadata` can be used together to do large knn queries and then fetch the metadata only when needed. It makes sense to do that as knn search can be very efficient thanks to strong [locality of reference](https://en.wikipedia.org/wiki/Locality_of_reference) of knn IVF index making it fast to do knn with a large K, whereas the current on-disk implementation of metadata (hdf5) does not have that property and hence cannot handle retrieving a large amount of random items fast.
In particular this can be used to implement infinite scroll in a front end.

### Clip back: Benchmark and monitoring

This backends has a 50ms latency if using memory mapped indices and metadata. Throughput is about 20 query/s. For high throughput, using a grpc server is required as well as a GPU for fast clip inference, turning off memory mapping options can also speed up requests, at the cost of high ram usage.

This backends also exposes a prometheus `/metrics` endpoint as well as an human readable summary at `/metrics-summary`.
This can (optionally) be used to setup a [grafana dashboard](doc_assets/grafana_dashboard.json) for monitoring:

[<img src="https://github.com/rom1504/clip-retrieval/raw/main/doc_assets/clip-back-grafana.png" alt="grafana" width="1200">](https://github.com/rom1504/clip-retrieval/raw/main/doc_assets/clip-back-grafana.png)

It can be seen on this dashboard that the slowest part of any call is fetching the image by its url in case of image url search, taking up to 300ms.
For text queries or image queries, the latency is about 50ms.
Here is an example of output in the metrics summary:
```
Among 20.0 calls to the knn end point with an average latency of 0.1889s per request, the step costs are (in order): 
                        name                               description  calls  average proportion
0              download_time             Time spent downloading an url      6  0.3215s     170.2%
1          metadata_get_time            Time spent retrieving metadata     20  0.0415s      21.9%
2             knn_index_time       Time spent doing a knn on the index     20  0.0267s      14.1%
3  image_clip_inference_time   Time spent doing a image clip inference      6  0.0206s      10.9%
4   text_clip_inference_time    Time spent doing a text clip inference     14  0.0186s       9.8%
5          image_prepro_time  Time spent doing the image preprocessing      6  0.0097s       5.2%
6           text_prepro_time   Time spent doing the text preprocessing     14  0.0020s       1.0%
```

## clip-front

Clip front is a simple UI that connects to clip back and display the results.
You can use it at [clip-retrieval ui](https://rom1504.github.io/clip-retrieval/)

Or you can run it yourself with:
```
npm install -g clip-retrieval-front
clip-retrieval-front 3005
```

### Development

For development it, go to [front](front) and run `npm install` then `npm start`.

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
