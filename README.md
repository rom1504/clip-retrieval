# clip-retrieval
[![pypi](https://img.shields.io/pypi/v/clip-retrieval.svg)](https://pypi.python.org/pypi/clip-retrieval)
[![NPM version](https://badge.fury.io/js/clip-retrieval-front.svg)](http://badge.fury.io/js/clip-retrieval-front)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rom1504/clip-retrieval/blob/master/notebook/clip-retrieval-getting-started.ipynb)
[![Try it on gitpod](https://img.shields.io/badge/try-on%20gitpod-brightgreen.svg)](https://gitpod.io/#https://github.com/rom1504/clip-retrieval)
[![Chat on discord](https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white)](https://discord.gg/eq3cAMZtCC)

Easily compute clip embeddings and build a clip retrieval system with them. 100M text+image embeddings can be processed in 20h using a 3080.

* clip inference allows you to quickly (1500 sample/s on a 3080) compute image and text embeddings
* clip index builds efficient indices out of the embeddings
* clip filter allows you to filter out the data using the clip index
* clip back hosts the indices with a simple flask service
* clip front is a simple ui querying the back. Check it out at [clip-retrieval ui](https://rom1504.github.io/clip-retrieval/)
* clip end2end runs img2dataset, inference, index then back and front to make all of this easier to begin with

End to end this make it possible to build a simple semantic search system.
Interested to learn about semantic search in general ? You can read my [medium post](https://rom1504.medium.com/semantic-search-with-embeddings-index-anything-8fb18556443c) on the topic.

Also see [laion5B](https://laion.ai/laion-5b-a-new-era-of-open-large-scale-multi-modal-datasets/) and [semantic search at billions scale](https://rom1504.medium.com/semantic-search-at-billions-scale-95f21695689a) to read more on how to make this scale to billion of samples.

[<img src="https://github.com/rom1504/clip-retrieval/raw/main/doc_assets/clip-front-pic.png" alt="clip front" width="500">](https://rom1504.github.io/clip-retrieval/)

## Who is using clip retrieval ?

* [cah-prepro](https://github.com/rom1504/cah-prepro) preprocess the 400M image+text crawling at home dataset. clip-retrieval is used to compute 400M clip embeddings and the indices
* [autofaiss](https://github.com/criteo/autofaiss) uses clip-retrieval to display an example of use (see the multimodal notebook example there)
* [afiaka87 openai demo](https://gist.github.com/afiaka87/f662486fc45199fa4394f3456c8246d7#file-dalle_blog_semantic_search-ipynb) shows how to look among the 1M example released by openai for their DALL-E demo
* [antarctic-captions by dzryk](https://github.com/dzryk/antarctic-captions) uses autofaiss and clip inference as a way to generate anchors for the image to text task with great success

## Install

pip install clip-retrieval

## clip end2end

First pick a dataset of image urls and captions ([examples](https://github.com/rom1504/img2dataset/tree/main/examples)) then run:

You may want to run `export CUDA_VISIBLE_DEVICES=` to avoid using your GPU if it doesn't have enough VRAM.

```
wget https://github.com/rom1504/img2dataset/raw/main/tests/test_1000.parquet
clip-retrieval end2end test_1000.parquet /tmp/my_output
```

Then go to [http://localhost:1234](http://localhost:1234) and enjoy searching among your pictures

Use `--run_back False` if you don't want to run the backend


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
* **wds_image_key** Key to use for images in webdataset. (default *jpg*)
* **wds_caption_key** Key to use for captions in webdataset. (default *txt*)
* **clip_model** CLIP model to load (default *ViT-B/32*). Specify it as `"open_clip:ViT-B-32-quickgelu"` to use the [open_clip](https://github.com/mlfoundations/open_clip).
* **mclip_model** MCLIP model to load (default *sentence-transformers/clip-ViT-B-32-multilingual-v1*)
* **use_mclip** If False it performs the inference using CLIP; MCLIP otherwise (default *False*)
* **use_jit** uses jit for the clip model (default *True*)
* **distribution_strategy** choose how to distribute the job, see distribution section for details (default *sequential*)
* **wds_number_file_per_input_file** estimation of the number of sample per tar if using wds and not specifying output_partition_count (default *10000*)
* **output_partition_count** number of output partitions (default *None*)
* **wandb_project** wandb project to use (default *clip_retrieval*)
* **enable_wandb** whether to use wandb (default *False*)


### Loading/writing files on hdfs

- To load a webdataset from a hdfs folder, set --input_dataset "pipe:hdfs dfs -cat path_on_hdfs" in the request without the "hdfs://" prefix.
- To write the output on hdfs, set --output_hdfs_folder to the path on hdfs prefixed by "hdfs://"

Example of hdfs query using webdataset format:
`clip_inference --input_dataset "pipe:hdfs dfs -cat /myfolder/webdataset/{00000..00010}.tar" --output_folder "hdfs://myfolder/embeddings" --input_format webdataset

### Loading/writing files on s3

`clip_inference --input_dataset "pipe:aws s3 cp --quiet s3://myfolder/webdataset/{00000..00010}.tar" --output_folder "s3://myfolder/embeddings" --input_format webdataset

### Distributed inference

To run this on multiple nodes (and multiple gpus), see tutorial at [docs/distributed_clip_inference.md](docs/distributed_clip_inference.md)

## Clip index

Clip index takes as input the output of clip inference and makes an index out of it using [autofaiss](https://github.com/criteo/autofaiss)

`clip-retrieval index --input_folder embeddings_folder --output_folder index_folder`

* `--max_index_memory_usage "16G"` option allow configuring the amount of ram the index will consume. More ram, better knn recall.
* `--current_memory_available 24G` allows controlling how much ram is used during the creation process.
* `--copy_metadata True` makes it possible to choose whether to copy metadata or not at the end of the process.
* `--nb_cores 8` allows controlling the number of threads 

The output is a folder containing:
* image.index containing a faiss index for images
* text.index containing a faiss index for texts
* metadata folder containing the parquet metadata

Thanks to autofaiss and faiss, this scales to hundred of million of samples in a few hours.

You may want to carefully pick how much memory to use for your index in order to maximize the knn recall.
[autofaiss index selection colab](https://colab.research.google.com/github/criteo/autofaiss/blob/master/docs/notebooks/autofaiss_index_selection_demo.ipynb) can help along with `autofaiss score_index` command to check the recall of your index. In general indices using more memory get a better recall and hence are closer to a naive (slow) knn

## Clip filter

Once the embeddings are computed, you may want to filter out the data by a specific query.
For that you can run `clip-retrieval filter --query "cat" --output_folder "cat/" --indice_folder "indice_folder"`
It will copy the 100 best images for this query in the output folder.
Using the `--num_results` or `--threshold` may be helpful to refine the filter

Thanks to fast knn index, this can run in real time (<10ms) for large K values (100000), and in minutes for very large K values.

This scripts works for small datasets. For larger ones, please check [notebook/simple_filter.ipynb].

## Clip back

Clip back is a simple knn service backend. If using both hdf5 and faiss memory mapping, it uses only the memory used by clip which is 4GB.

Run (output_folder is the output of clip index)
```bash
echo '{"example_index": "output_folder"}' > indices_paths.json
clip-retrieval back --port 1234 --indices-paths indices_paths.json
```


Options:
* `--use_jit True` uses jit for the clip model
* `--clip_model "ViT-B/32"` allows choosing the clip model to use
* `--enable_mclip_option True` loads the mclip model, making it possible to search in any language.
* `--columns_to_return='["url", "image_path", "caption", "NSFW"]` allows you to specify which columns should be fetched from the metadata and returned by the backend. It's useful to specify less in case of hdf5 caching to speed up the queries.
* `--enable_faiss_memory_mapping=True` option can be passed to use an index with memory mapping.
That decreases the memory usage to zero.
* `--enable_hdf5 True` option can be passed to enable hdf5 caching for the metadata.
HDF5 caching makes it possible to use the metadata with almost no memory usage.
* `--use_arrow True` allows using arrow instead of hdf5. Should be used along with [clip_back_prepro](clip_back_prepro) for very large datasets (billions)
* `--reorder_metadata_by_ivf_index True` option takes advantage of the data locality property of results of a knn ivf indices: it orders the metadata collection in order of the IVF clusters. That makes it possible to have much faster metadata retrieval as the reads are then accessing a few mostly sequential parts of the metadata instead of many non sequential parts. In practice that means being able to retrieve 1M items in 1s whereas only 1000 items can be retrieved in 1s without this method. This will order the metadata using the first image index.
* `--provide_safety_model True` will automatically download and load a [safety model](https://github.com/LAION-AI/CLIP-based-NSFW-Detector). You need to `pip install autokeras` optional dependency for this to work.
* `--provide_violence_detector True` will load a [violence detector](https://github.com/ml-research/OffImgDetectionCLIP), [paper](https://arxiv.org/abs/2202.06675.pdf)
* `--provide_aesthetic_embeddings True` will load the [aesthetic embeddings](https://github.com/LAION-AI/aesthetic-predictor) and allow users to make the query move towards a nicer point of the clip space

These options can also be provided in the config file to have different options for each index. Example:
```json
{
        "laion5B": {
                "indice_folder": "/mnt/laion5B/prepared_data",
                "provide_safety_model": true,
                "enable_faiss_memory_mapping": true,
                "use_arrow": true,
                "enable_hdf5": false,
                "reorder_metadata_by_ivf_index": false,
                "columns_to_return": ["url", "caption"],
                "clip_model": "ViT-L/14",
                "enable_mclip_option": false
        },
        "laion_400m": {
                "indice_folder": "/mnt/laion400M/index100",
                "provide_safety_model": true,
                "enable_faiss_memory_mapping": true,
                "enable_hdf5": true,
                "use_arrow": false,
                "reorder_metadata_by_ivf_index": true,
                "enable_mclip_option": true,
                "clip_model": "ViT-B/32"
        }
}
```

hdf5 or arrow caching is a good idea to use if:
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

By default the backend will also expose a front end. That front end will by default hit this backend, however you may need to specify whether this is happening over http or https, in this case use the option `--default_backend` to specify the backend url. `--url_column` allows specifying the name of the column url for the front

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

You can also run it with `clip-retrieval front` or back from the python package.

### Development

For development it, go to [front](front) and run `npm install` then `npm start`.

## For development

Either locally, or in [gitpod](https://gitpod.io/#https://github.com/rom1504/clip-retrieval) (do `export PIP_USER=false` there)

Setup a virtualenv:

```
python3 -m venv .env
source .env/bin/activate
pip install -e .
```

to run tests:
```
pip install -r requirements-test.txt
```
then 
```
make lint
make test
```

You can use `make black` to reformat the code

`python -m pytest -x -s -v tests -k "test_runner"` to run a specific test

If you want to use the front through the python backend or frontend, run
```
cd front
npm install
npm run build
cd ..
pip install -e .
```
