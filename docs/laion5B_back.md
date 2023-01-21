# Run clip-retrieval back with laion5B index

First step, download all the files from https://huggingface.co/datasets/laion/laion5B-index

Second step,
```bash
python3 -m venv .env
source .env/bin/activate
pip install clip-retrieval autokeras==1.0.18 keras==2.8.0 Keras-Preprocessing==1.1.2 tensorflow==2.8.0`
```
(autokeras is optional and needed only for safety filtering)

Then put this
```json
{
        "laion5B": {
                "indice_folder": "laion5B-index",
                "provide_safety_model": true,
                "enable_faiss_memory_mapping": true,
                "use_arrow": true,
                "enable_hdf5": false,
                "reorder_metadata_by_ivf_index": false,
                "columns_to_return": ["url", "caption"],
                "clip_model": "ViT-L/14",
                "enable_mclip_option": true
        }
}
```
in indices.json file

Finally run this
```python
export CUDA_VISIBLE_DEVICES=
clip-retrieval back --provide_violence_detector True --provide_safety_model True  --clip_model="ViT-L/14" --default_backend="http://localhost:1234/" --port 1234 --indices-paths indices.json --use_arrow True --enable_faiss_memory_mapping True --columns_to_return='["url", "caption", "md5"]'
```
