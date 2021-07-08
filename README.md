# clip-retrieval
Easily computing clip embeddings and building a clip retrieval system with them.

* clip batch allows you to quickly (1500 sample/s on a 3080) compute image and text embeddings and indices
* clip filter allows you to filter out the data using the clip embeddings
* clip back hosts the indices with a simple flask service
* clip service is a simple ui querying the back

End to end this make it possible to build a simple semantic search system.
Interested to learn about semantic search in general ? You can read by [medium post](https://rom1504.medium.com/semantic-search-with-embeddings-index-anything-8fb18556443c) on the topic.

## clip batch

First install it by running:
```
python3 -m venv .env
source .env/bin/activate
pip install -U pip
pip install clip-anytorch faiss-cpu fire
```

Then put some images in a `example_folder` and some text with the same name (or use --enable_text=False) then
* `python clip_batch.py  --dataset_path example_folder --output_folder output_folder`

Output folder will contain:
* description_list containing the list of caption line by line
* image_list containing the file path of images line by line
* img_emb.npy containing the image embeddings as numpy
* text_emb.npy containing the text embeddings as numpy
* image.index containing a brute force faiss index for images
* text.index containing a brute force faiss index for texts

## Clip filter

Once the embeddings are computed, you may want to filter out the data by a specific query.
For that you can run `python clip_filter.py --query "dog" --output_folder "dog/" --indice_name "example_index"`
It will copy the 100 best images for this query in the output folder.
Using the `--num_results` or `--threshold` may be helpful to refine the filter

## Clip back

First install it by running:
```bash
python3 -m venv .env
source .env/bin/activate
pip install -U pip
pip install clip-anytorch faiss-cpu fire flask flask_cors flask_restful 
```

Then run (output_folder is the output of clip batch)
```bash
echo '{"example_index": "output_folder"}' > indices_paths.json
python clip_back.py 1234
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