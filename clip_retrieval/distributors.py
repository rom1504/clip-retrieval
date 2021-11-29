from abc import ABC
import math


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def default_distributor_builder(distributor_kind):

    if distributor_kind == "sequential":
        return SequentialDistributor()
    elif distributor_kind == "multigpu":
        return MultiGpuDistributor()
    elif distributor_kind == "pyspark":
        return PySparkDistributor()
    elif distributor_kind == "pyspark_multigpu":
        return ComposedDistributor(MultiGpuDistributor(), PySparkDistributor())
    else:
        raise ValueError("Unknown distributor kind: {}".format(distributor_kind))


class Sampler:
    def __init__(self, output_partitions, output_partition_id):
        self.output_partitions = output_partitions
        self.output_partition_id = output_partition_id

    def __call__(self, l):
        batch_size = math.ceil(len(l) / self.output_partitions)

        for i, e in enumerate(l):
            if i % batch_size == self.output_partition_id:
                yield e


class AbstractDistributor(ABC):
    def __call__(self, output_partitions, mapper):
        """Calls mapper with a sampler of the form (sample_id, total_sample)"""
        # would be smart
        raise NotImplementedError()


class SequentialDistributor(AbstractDistributor):
    def __call__(self, output_partitions, mapper):
        for i in range(output_partitions):
            sampler = Sampler(output_partitions, i)
            mapper(sampler)

class ComposedDistributor(AbstractDistributor):
    def __init__(self, internal_distributor, external_distributor):
        self.internal_distributor = internal_distributor
        self.external_distributor = external_distributor

    def __call__(self, output_partitions, mapper):
        def worker(sampler):
            p = len(sampler(range(output_partitions)))
            self.internal_distributor(p, mapper)

        self.external_distributor(output_partitions, worker)


class MultiGpuDistributor(AbstractDistributor):
    def __init__(self):
        import torch
        self.num_workers = torch.cuda.device_count()

    def __call__(self, output_partitions, mapper):
        import multiprocessing as mp
        import torch

        def worker(output_partition_id):
            # to fix
            worker_id = mp.current_process()._identity[0]
            with torch.cuda.device(worker_id):
                mapper(Sampler(output_partitions, output_partition_id))

        with mp.Pool(self.num_workers) as p:
            p.map(worker, range(output_partitions))

class PySparkDistributor(AbstractDistributor):
    def __init__(self, spark_context=None):
        if spark_context is None:
            import pyspark

            self.sc = pyspark.SparkContext.getOrCreate()
        self.sc = spark_context

    def __call__(self, output_partitions, mapper):
        rdd = self.sc.parallelize(range(output_partitions))
        def worker(output_partition_id):
            mapper(Sampler(output_partitions, output_partition_id))
        rdd.foreach(worker)