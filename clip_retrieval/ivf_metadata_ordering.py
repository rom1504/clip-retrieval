"""ivf metadata ordering is a module to reorder a metadata collection by ivf clusters"""

import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
from collections import defaultdict
import heapq
import time
import pandas as pd

import pyarrow.parquet as pq
import h5py
import faiss


def search_to_new_ids(index, query, k):
    """
    this function maps the result ids to the ones ordered by the ivf clusters
    to be used along with a re-ordered metadata
    """
    distances, indices = index.search(query, k)
    opq2 = faiss.downcast_VectorTransform(index.chain.at(0))
    xq = opq2.apply(query)
    _, l = faiss.extract_index_ivf(index).quantizer.search(xq, faiss.extract_index_ivf(index).nprobe)
    il = faiss.extract_index_ivf(index).invlists
    list_sizes = [il.list_size(i) for i in range(il.nlist)]
    starting_offset = []
    c = 0
    for i in list_sizes:
        starting_offset.append(c)
        c += i
    old_id_to_new_id = {}
    for i in l[0]:
        i = int(i)
        ids = il.get_ids(i)
        list_size = il.list_size(int(i))
        items = faiss.rev_swig_ptr(ids, list_size)
        for nit, it in enumerate(items):
            old_id_to_new_id[it] = starting_offset[i] + nit
        il.release_ids(ids=ids, list_no=i)
    ids = np.array([old_id_to_new_id[i] if i != -1 else -1 for i in indices[0]])
    return distances, ids


def get_old_to_new_mapping(index):
    """
    use an ivf index to compute a mapping from initial ids to ids ordered by clusters
    """
    il = faiss.extract_index_ivf(index).invlists
    d = np.ones((index.ntotal,), "int64")
    begin_list = []
    current_begin = 0
    for i in tqdm(range(il.nlist)):
        begin_list.append(current_begin)
        ids = il.get_ids(i)
        list_size = il.list_size(int(i))
        items = faiss.rev_swig_ptr(ids, list_size)
        new_ids = range(current_begin, current_begin + list_size)
        d.put(np.array(items, "int"), np.array(new_ids, "int"))
        il.release_ids(ids=ids, list_no=i)
        current_begin += list_size

    return d


def re_order_parquet(index, input_path, output_path, columns_to_return):
    """
    use external sort to reorder parquet files
    """
    d = get_old_to_new_mapping(index)
    data_dir = Path(input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    current_offset = 0
    current_id = 0
    for parquet_files in tqdm(sorted(data_dir.glob("*.parquet"))):
        df = pd.read_parquet(parquet_files)
        df["new_id"] = d[current_offset : current_offset + len(df)]
        saved_df = df[columns_to_return + ["new_id"]]
        saved_df = saved_df.sort_values("new_id")
        saved_df.to_parquet(output_path + "/meta_" + str(current_id) + ".parquet")
        current_id += 1
        current_offset += len(df)


class Hdf5Sink:
    """
    A hdf5 sink: take as input rows and write them to hdf5 regularly
    """

    def __init__(self, output_hdf5_file, keys):
        self.f = h5py.File(output_hdf5_file, "w")
        self.ds = self.f.create_group("dataset")
        self.buffer = []
        self.keys = keys

    def write(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) == 10**6:
            self._write_buffer()

    def end(self):
        self._write_buffer()
        self.f.close()

    def _write_buffer(self):
        """
        Write a list of rows to hdf5
        """
        if len(self.buffer) == 0:
            return
        df = pd.DataFrame(self.buffer, columns=self.keys)
        for k, v in df.items():
            if k not in self.keys:
                continue
            col = v
            if col.dtype in ("float64", "float32"):
                col = col.fillna(0.0)
            if col.dtype in ("int64", "int32"):
                col = col.fillna(0)
            if col.dtype == "object":
                col = col.fillna("")
            z = col.to_numpy()
            if k not in self.ds:
                self.ds.create_dataset(k, data=z, maxshape=(None,), compression="gzip")
            else:
                prevlen = len(self.ds[k])
                self.ds[k].resize((prevlen + len(z),))
                self.ds[k][prevlen:] = z
        self.buffer = []


class DummySink:
    def __init__(self):
        pass

    def write(self, sample):
        pass

    def end(self):
        pass


def external_sort_parquet(output_sink, input_path):
    """
    create heap
    add to heap 1 batch of each file
    store in dict nb of item in heap for each file
    start getting from heap and pushing to sink
    when nb_item[last_retrieved] == 0 and there is some item left in this file, add a new batch of that file in heap
    """

    h = []
    data_dir = Path(input_path)
    files = [pq.ParquetFile(filename, memory_map=True) for filename in sorted(data_dir.glob("*.parquet"))]
    batches_list = [ffile.iter_batches(batch_size=10**4) for ffile in files]
    index_to_value = {}
    counts = [ffile.metadata.num_rows for ffile in files]
    current_count_per_file = defaultdict(lambda: 0)

    def read_batch(i):
        batch = next(batches_list[i])
        current_count_per_file[i] += batch.num_rows
        df = batch.to_pandas()
        data = zip(df["new_id"], *[df[c] for c in [c for c in df.columns if c != "new_id"]])
        for e in data:
            heapq.heappush(h, (e[0], i))
            index_to_value[e[0]] = e[1:]

    for i in range(len(batches_list)):
        read_batch(i)

    done_count_per_file = defaultdict(lambda: 0)
    c = 0
    begin = time.time()
    while h:
        c += 1
        e, i = heapq.heappop(h)
        v = index_to_value[e]
        del index_to_value[e]
        output_sink.write(v)
        current_count_per_file[i] -= 1
        done_count_per_file[i] += 1
        if current_count_per_file[i] == 0 and done_count_per_file[i] < counts[i]:
            read_batch(i)
        if c % 100000 == 0:
            print(e, c, time.time() - begin, "s")

    output_sink.end()
