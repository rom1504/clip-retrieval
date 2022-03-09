"""the index combiner module is used to combine the index files into a single index file"""

from pathlib import Path
from faiss.contrib.ondisk import merge_ondisk
import faiss
import fire


def index_combiner(input_folder, output_folder):
    index_dir = Path(input_folder)
    block_fnames = sorted(index_dir.glob("*.index"))
    empty_index = faiss.read_index(block_fnames[0], faiss.IO_FLAG_MMAP)
    empty_index.ntotal = 0

    merge_ondisk(empty_index, block_fnames, output_folder + "/merged_index.ivfdata")

    faiss.write_index(empty_index, output_folder + "/populated.index")


if __name__ == "__main__":
    fire.Fire(index_combiner)
