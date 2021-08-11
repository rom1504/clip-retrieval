# clip-retrieval
[![pypi](https://img.shields.io/pypi/v/clip-retrieval.svg)](https://pypi.python.org/pypi/clip-retrieval)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rom1504/clip-retrieval/blob/master/notebook/clip-retrieval-getting-started.ipynb)
[![Try it on gitpod](https://img.shields.io/badge/try-on%20gitpod-brightgreen.svg)](https://gitpod.io/#https://github.com/rom1504/clip-retrieval)

Easily computing clip embeddings and building a clip retrieval system with them.

* clip batch allows you to quickly (1500 sample/s on a 3080) compute image and text embeddings and indices
* clip filter allows you to filter out the data using the clip embeddings
* clip back hosts the indices with a simple flask service
* clip service is a simple ui querying the back

End to end this make it possible to build a simple semantic search system.
Interested to learn about semantic search in general ? You can read by [medium post](https://rom1504.medium.com/semantic-search-with-embeddings-index-anything-8fb18556443c) on the topic.

## Install

pip install clip-retrieval

## clip batch

Get some images in an `example_folder`, for example by doing:
```
pip install img2dataset
echo 'https://placekitten.com/200/305' >> myimglist.txt
echo 'https://placekitten.com/200/304' >> myimglist.txt
echo 'https://placekitten.com/200/303' >> myimglist.txt
img2dataset --url_list=myimglist.txt --output_folder=image_folder --thread_count=64 --image_size=256
```
You can also put text files with the same names as the images in that folder, to get the text embeddings.

Then run `clip-retrieval batch --dataset_path image_folder --output_folder indice_folder`

Output folder will contain:
* description_list containing the list of caption line by line
* image_list containing the file path of images line by line
* img_emb.npy containing the image embeddings as numpy
* text_emb.npy containing the text embeddings as numpy
* image.index containing a brute force faiss index for images
* text.index containing a brute force faiss index for texts

## Clip filter

Once the embeddings are computed, you may want to filter out the data by a specific query.
For that you can run `clip-retrieval filter --query "cat" --output_folder "cat/" --indice_folder "indice_folder"`
It will copy the 100 best images for this query in the output folder.
Using the `--num_results` or `--threshold` may be helpful to refine the filter

## Clip back

Then run (output_folder is the output of clip batch)
```bash
echo '{"example_index": "output_folder"}' > indices_paths.json
clip-retrieval back --port 1234 --indices-paths indices_paths.json
```

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

## For development

Either locally, or in [gitpod](https://gitpod.io/#https://github.com/rom1504/img2dataset) (do `export PIP_USER=false` there)

Setup a virtualenv:

```
python3 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -e .
```
