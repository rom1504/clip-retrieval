"""the parquet to arrow module is used to convert the parquet files into a single arrow file"""

from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import fire

# do many small files instead


def parquet_to_arrow(parquet_folder, output_arrow_file, columns_to_return):
    data_dir = Path(parquet_folder)
    files = sorted(data_dir.glob("*.parquet"))

    schema = pq.read_table(files[0], columns=columns_to_return).schema
    with pa.OSFile(output_arrow_file, "wb") as sink:
        with pa.RecordBatchFileWriter(sink, schema) as writer:
            for parquet_files in tqdm(files):
                table = pq.read_table(parquet_files, columns=columns_to_return)
                writer.write_table(table)


if __name__ == "__main__":
    fire.Fire(parquet_to_arrow)
