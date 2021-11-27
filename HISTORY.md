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
