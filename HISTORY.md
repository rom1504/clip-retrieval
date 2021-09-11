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
