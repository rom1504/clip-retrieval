rm -rf /tmp/folder
time clip-retrieval inference --input_dataset="http://the-eye.eu/eleuther_staging/cah/releases/laion400m/{00000..01000}.tar" --output_folder="/tmp/folder" \
--input_format "webdataset" --subset_size=100000 --enable_metadata=True --write_batch_size=100000 --batch_size=512 --cache_path=None