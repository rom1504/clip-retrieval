"""the parquet to arrow module is used to convert the parquet files into arrow files"""

from multiprocessing.pool import ThreadPool
import os
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import fire
import math


def file_to_count(filename):
    with open(filename, "rb") as f:
        parquet_file = pq.ParquetFile(f, memory_map=True)
        return parquet_file.metadata.num_rows


def count_samples(files):
    total_count = 0
    with ThreadPool(10) as p:
        for c in tqdm(p.imap(file_to_count, files), total=len(files)):
            total_count += c
    return total_count


def parquet_to_arrow(parquet_folder, output_arrow_folder, columns_to_return):
    """convert the parquet files into arrow files"""
    os.makedirs(output_arrow_folder, exist_ok=True)
    data_dir = Path(parquet_folder)
    files = sorted(data_dir.glob("*.parquet"))
    number_samples = count_samples(files)
    print("There are {} samples in the dataset".format(number_samples))  # pylint: disable=consider-using-f-string

    schema = pq.read_table(files[0], columns=columns_to_return).schema
    sink = None
    current_batch_count = 0
    batch_counter = 0
    key_format = int(math.log10(number_samples / 10**10)) + 1
    for parquet_files in tqdm(files):
        if sink is None or current_batch_count > 10**10:
            if sink is not None:
                writer.close()
                sink.close()
            file_key = "{true_key:0{key_format}d}".format(  # pylint: disable=consider-using-f-string
                key_format=key_format, true_key=batch_counter
            )
            file_name = f"{output_arrow_folder}/{file_key}.arrow"
            print(f"Writing to {file_name}")
            sink = pa.OSFile(file_name, "wb")
            writer = pa.ipc.new_file(sink, schema)
            current_batch_count = 0
            batch_counter += 1

        print("going to read parquet file: ", parquet_files)
        for i in range(2):
            try:
                table = pq.read_table(parquet_files, columns=columns_to_return, use_threads=False)
            except Exception as e:  # pylint: disable=broad-except
                if i == 1:
                    raise e
                print("Error reading parquet file: ", e)
                print("Retrying once...")
                continue
        writer.write_table(table)
        current_batch_count += table.num_rows
    if sink is not None:
        writer.close()
        sink.close()


if __name__ == "__main__":
    fire.Fire(parquet_to_arrow)
