"""distributors provide way to compute using several gpus and several machines"""

import os

from .worker import worker


class SequentialDistributor:
    def __init__(self, tasks, worker_args):
        self.tasks = tasks
        self.worker_args = worker_args

    def __call__(self):
        """
        call a single `worker(...)` and pass it everything.
        """
        worker(
            tasks=self.tasks,
            **self.worker_args,
        )


class PysparkDistributor:
    """the pyspark distributor uses pyspark for distribution"""

    def __init__(self, tasks, worker_args):
        self.tasks = tasks
        self.worker_args = worker_args

    def __call__(self):
        """
        Parallelize work and call `worker(...)`
        """

        import pyspark  # pylint: disable=import-outside-toplevel
        from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

        spark = SparkSession.getActiveSession()

        if spark is None:
            print("No pyspark session found, creating a new one!")
            spark = (
                SparkSession.builder.config("spark.driver.memory", "16G")
                .master("local[" + str(2) + "]")
                .appName("spark-stats")
                .getOrCreate()
            )

        rdd = spark.sparkContext.parallelize(c=self.tasks, numSlices=len(self.tasks))

        def run(partition_id):
            context = pyspark.TaskContext.get()
            if "gpu" in context.resources():
                gpu = context.resources()["gpu"].addresses[0]
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu

            worker(tasks=[partition_id], **self.worker_args)

        rdd.foreach(run)
