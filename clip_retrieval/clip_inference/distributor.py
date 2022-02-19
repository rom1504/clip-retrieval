"""distributors provide way to compute using several gpus and several machines"""

import os


class SequentialDistributor:
    def __init__(self, runner, output_partition_count):
        self.runner = runner
        self.output_partition_count = output_partition_count

    def __call__(self):
        for i in range(self.output_partition_count):
            self.runner(i)


class PysparkDistributor:
    """the pyspark distributor uses pyspark for distribution"""

    def __init__(self, runner, output_partition_count):
        self.runner = runner
        self.output_partition_count = output_partition_count

    def __call__(self):
        from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel
        import pyspark  # pylint: disable=import-outside-toplevel

        spark = SparkSession.getActiveSession()

        if spark is None:
            print("No pyspark session found, creating a new one!")
            spark = (
                SparkSession.builder.config("spark.driver.memory", "16G")
                .master("local[" + str(2) + "]")
                .appName("spark-stats")
                .getOrCreate()
            )

        df = list(range(self.output_partition_count))
        rdd = spark.sparkContext.parallelize(df, len(df))

        def run(i):
            context = pyspark.TaskContext.get()
            if "gpu" in context.resources():
                gpu = context.resources()["gpu"].addresses[0]
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu
            self.runner(i)

        rdd.foreach(run)
