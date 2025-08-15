## 2.45.0

* Update Python version support to 3.10-3.12
* Fix PyTorch 2.8 compatibility issues with upstream all-clip 1.3.0
* Update dependencies to latest versions
* Update GitHub Actions to latest versions

## 2.44.0

* Support get_tokenizer in clip back inf

## 2.43.0

* Update more deps (fire, pyarrow, pandas, torch)

## 2.42.0

* Update deps

## 2.41.0

* Update scipy requirement from <1.9.2 to <1.11.5
* catch and skip images that fail to load (thanks @heyalexchoi)
* Handle images in multiple folder for files reader and handle uppercase extension (thanks @BIGBALLON)

## 2.40.0

* Add support for the full open clip model name format : ViT-B-32/laion2b_s34b_b79k (thanks @mehdidc @barinov274)

## 2.39.0

* Add DeepSparse backend for CLIP inference (thanks @mgoin)
* fix parquet to arrow script failed when number of samples is small (thanks @luke-han)
* Integration with hugging face ClipModel (thanks @Sofianel5)

## 2.38.0

* Add webp to list of supported files in reader.
* Remove version constraint of fsspec.

## 2.37.0

* Update versions to fix pex and npm build
* Improve errors for empty input folders.
* Default context to fix bug with some requests returning 404

## 2.36.1

* Fix truncate

## 2.36.0

* Make jit=False the default in clip inference
* update webdataset and fsspec
* Add H14 NSFW detector
* Support get tokenizer in clip back  (thanks @nousr)
* enable filtering by image with clip-retrieval filter

## 2.35.1

* update key toggles in inf.main (thanks @nousr)

## 2.35.0

* Slurm distributor (thanks @nousr)
* Autocast for openclip
* support openclip in clip back

## 2.34.2

* Read image data from path in case "image_path" is present

## 2.34.1

* Makes file image reader in clip inference fast

## 2.34.0

* Make it possible to use an embedding as query of the back

## 2.33.0

* add clip-client module for querying backend remotely (thanks @afiaka87 )

## 2.32.0

* use better mclip from https://github.com/FreddeFrallan/Multilingual-CLIP

## 2.31.1

* add clearer way to disable aesthetic scoring in front

## 2.31.0

* aesthetic option

## 2.30.0

* Log error for unsupported input_format (thanks @dmvaldman)
* Add open_clip support (thanks @cat-state)

## 2.29.1

* fix mclip in clip back

## 2.29.0

* add violence detector to clip back

## 2.28.0

* add feature to pass options in config file

## 2.27.0

* safety model for ViT-B/32

## 2.26.0

* replace safety heuristic by safety model

## 2.25.4

* enable back dedup of images

## 2.25.3

* turn off image dedup by default temporarily

## 2.25.2

* fix range search use

## 2.25.1

* add back node build in publish

## 2.25.0

* new arrow provider in clip back
* index combiner script
* parquet to arrow script
* deduplication of results feature

## 2.24.10

* one more fix for text only

## 2.24.9

* fix image_tensor_count vs text_counter count in runner

## 2.24.8

* fix file count check for input format files

## 2.24.7

* going back to autofaiss main

## 2.24.6

* switch to fork of autofaiss

## 2.24.5

* properly close the wandb run at the end

## 2.24.4

* fix pex building

## 2.24.3

* fix version ranges

## 2.24.2

* fix sample_count == 0 issue in logger and handle no text sample properly in main

## 2.24.1

* improve logger by checking the file exists before reading

## 2.24.0

* use zero padding for output file names
* add proper multi gpu support in pyspark distributor
* improve printing of error in logger

## 2.23.3

* fix another small issue with logger reporting

## 2.23.2

* small fix in logger computation

## 2.23.1

* Fix race condition when using mkdir in writer

## 2.23.0

* Refactor clip inference, make it support distributed inference

## 2.22.0

* add use_jit option to back and inference, now True by default, add clip_model option to back

## 2.21.0

* mclip support in clip back and front

## 2.20.0

* replace null bytes while transforming parquet to hdf5
* Use collate_fn to skip corrupt images without using recursion (thanks @afiaka87)
* truncate text inputs in clip back

## 2.19.1

* fix url column option bug

## 2.19.0

* add url column option
* use torch no grad to fix a memleak in clip back

## 2.18.0

* add default backend url in clip back

## 2.17.0

* add option in clip end 2 end to avoid running the back

## 2.16.2

* update for autofaiss

## 2.16.1

* add missing front building in python publish

## 2.16.0

* clip retrieval end2end

## 2.15.1

* minor bug fix about missing .npy extension in output of clip inference

## 2.15.0

* mclip support
* use fsspec to make it possible to output to any fs

## 2.14.3

* add indice deduplication in the output of clip back

## 2.14.2

* use the npy mapping in all cases for ivf reordering since it's fast enough

## 2.14.1

* save ivf_old_to_new_mapping for the text index to use

## 2.14.0

* implement ivf re-ordering for much faster metadata fetching
* add download button in front

## 2.13.1

* fix filterDuplicateUrls issue when there is no url, only images
* fix default columns_to_return

## 2.13.0

* add a simple filter ipynb notebook

## 2.12.0

* implement infinite scroll feature

## 2.11.2

* fix limiting of results in clip back
* fix absence of caption in clip front

## 2.11.1

* fix an issue in clip front handling of default
* limit the number of results to the number available in clip back

## 2.11.0

* add compression by default when creating the hdf5 cache file

## 2.10.0

* add columns_to_return in clip back
* safe mode in front

## 2.9.2

* fix metrics sorting in metrics summary

## 2.9.1

* add download url time and descriptions in metrics summary endpoint

## 2.9.0

* add prometheus endpoint in clip back

## 2.8.1

* properly display errors in clip index

## 2.8.0

* add nb cores option in clip index

## 2.7.1

* add folder name option and catch errors in clip index

## 2.7.0

* package front in npm

## 2.6.0

* implement image url search in clip back

## 2.5.0

* add memory mapping option in clip back : 0 memory usage to load an index!

## 2.4.0

* add copy metadata option to clip index

## 2.3.0

* allows controlling the amount of ram used during the creation process of the index
* add logs in clip back to inform when each thing is loaded
* fix PIL call (thanks @pvl)

## 2.2.0

* expose max_index_memory_usage

## 2.1.0

* --wds_image_key, --wds_caption_key options (thanks @afiaka87)
* implement h5py caching in clip back

## 2.0.4

* fix clip back and filter to use sorted metadatas

## 2.0.3

* fix finding the last batch number (continuing output)

## 2.0.2

* add warn and continue handler to avoid crashing

## 2.0.1

* add missing webdataset dep

## 2.0.0

* webdataset input format
* save in batch
* test files in tests folder
* save metadata as parquet
* use autofaiss in a new clip index
* remove indexing from clip batch and rename to clip inference

## 1.0.1

* fixes

## 1.0.0

* it works
