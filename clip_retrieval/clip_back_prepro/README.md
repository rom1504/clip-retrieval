## clip back prepro

Clip back preprocessing jobs transform metadata and indices into a form that is easier to use for clip back.

This is helpful to load large datasets in clip back, for example laion5B that has a 800GB index and 900GB of metadata.

### Parquet to arrow

The parquet to arrow script converts many parquet files into a few arrow files.

The benefit of arrow format compared to parquet is that it's possible to memmap it, allowing to use large amount of metadata at no memory cost.

Usage example

```bash
clip-retrieval parquet_to_arrow --parquet_folder "/media/hd2/allmeta/2Ben"\
 --output_arrow_folder "/media/nvme/large_index/metadata/2B-en"\
  --columns_to_return='["url", "caption"]'
```

### Index combiner

The indexer combiner script converts many indices into a single index file, without using memory.

This makes it possible to use a large index at low memory cost (<500MB)

Usage example

```bash
clip_retrieval index_combiner --input_folder "the/indices"\
 --output_folder "output"
```