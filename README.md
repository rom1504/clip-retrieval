# clip-retrieval
Easily computing clip embeddings and building a clip retrieval system with them

## clip batch

First install it by running:
```
python3 -m venv .env
source .env/bin/activate
pip install -U pip
pip install clip-by-openai faiss-cpu fire
```

Then put some images in a `example_folder` and some text with the same name (or use --enable_text=False) then
* `python clip_batch.py  --dataset_path example_folder --output_folder output_folder`

