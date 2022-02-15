"""The runner combine reader, mapper and writer to produce clip embeddings"""


class Sampler:
    def __init__(self, output_partition_id, output_partition_count):
        self.output_partition_id = output_partition_id
        self.output_partition_count = output_partition_count

    def __call__(self, l):
        return [e for i, e in enumerate(l) if i % self.output_partition_count == self.output_partition_id]


class Runner:
    """Runner class"""

    def __init__(self, reader_builder, mapper_builder, writer_builder, output_partition_count):
        self.reader_builder = reader_builder
        self.mapper_builder = mapper_builder
        self.writer_builder = writer_builder
        self.output_partition_count = output_partition_count

    def __call__(self, i):
        sampler = Sampler(i, self.output_partition_count)
        reader = self.reader_builder(sampler)
        writer = self.writer_builder(i)
        mapper = self.mapper_builder()
        for batch in reader:
            writer(mapper(batch))
        writer.flush()
