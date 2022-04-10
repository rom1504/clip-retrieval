"""The logger module allows logging to stdout and wandb"""

from collections import defaultdict
import fsspec
import multiprocessing
import time
import json
import wandb
import queue


class LoggerWriter:
    """the logger writer write stats to json file, for each worker"""

    def __init__(self, partition_id, stats_folder):
        self.partition_id = partition_id
        self.stats_folder = stats_folder

    def start(self):
        ctx = multiprocessing.get_context("spawn")
        self.queue = ctx.Queue()
        self.updater_process = ctx.Process(target=self.updater)
        self.updater_process.start()

    def end(self):
        self.queue.put(None)
        self.updater_process.join()
        self.queue.close()

    def __call__(self, stats):
        self.queue.put(stats)

    def updater(self):
        """updater process that writes stats to file from the queue"""
        stats = defaultdict(lambda: 0)
        fs, relative_path = fsspec.core.url_to_fs(self.stats_folder)
        last_write = None
        while True:
            item = self.queue.get()
            if item is None:
                self.write_stats(stats, fs, relative_path, False)
                return
            for k in item:
                stats[k] += item[k]
            if last_write is None or time.time() - last_write > 5:
                self.write_stats(stats, fs, relative_path, True)
                last_write = time.time()

    def sum(self, stats, new_stats):
        for k in stats.keys():
            stats[k] += new_stats[k]
        return stats

    def write_stats(self, stats, fs, relative_path, wip):
        fs.makedirs(relative_path, exist_ok=True)
        if not wip and fs.exists(relative_path + f"/wip_{self.partition_id}.json"):
            fs.rm(relative_path + f"/wip_{self.partition_id}.json")
        prefix = "wip_" if wip else ""
        with fs.open(relative_path + f"/{prefix}{self.partition_id}.json", "w") as f:
            f.write(json.dumps(stats))


class LoggerReader:
    """the logger reader read stats of all json files and aggregate them"""

    def __init__(self, stats_folder, wandb_project="clip_retrieval", enable_wandb=False):
        self.stats_folder = stats_folder
        self.enable_wandb = enable_wandb
        self.wandb_project = wandb_project
        self.log_interval = 5

    def start(self):
        ctx = multiprocessing.get_context("spawn")
        self.queue = ctx.Queue()
        self.start_time = time.perf_counter()
        self.reader_process = ctx.Process(target=self.reader)
        self.reader_process.start()

    def end(self):
        self.queue.put("end")
        self.reader_process.join()
        self.queue.close()

    def reader(self):
        """reader process that reads stats from files and aggregates them"""
        if self.enable_wandb:
            self.current_run = wandb.init(project=self.wandb_project)
        else:
            self.current_run = None

        last_check = 0
        stats = {}
        fs, relative_path = fsspec.core.url_to_fs(self.stats_folder, use_listings_cache=False)

        fs.makedirs(relative_path, exist_ok=True)

        while True:  # pylint: disable=too-many-nested-blocks
            time.sleep(0.1)
            try:
                self.queue.get(False)
                last_one = True
            except queue.Empty as _:
                last_one = False
            if not last_one and time.perf_counter() - last_check < self.log_interval:
                continue

            last_check = time.perf_counter()

            stats_files = fs.glob(relative_path + "/*.json")
            for k in stats_files:
                filename = k.split("/")[-1]
                if filename[:4] == "wip_" or filename not in stats:
                    for i in range(5):  # pylint: disable=unused-variable
                        try:
                            fs.invalidate_cache()
                            if not fs.exists(k):
                                continue
                            with fs.open(k, "r") as f:
                                stats[filename] = json.loads(f.read())
                            if filename[:4] != "wip_" and "wip_" + filename in stats:
                                del stats["wip_" + filename]
                            break
                        except Exception as e:  # pylint: disable=broad-except
                            if i == 4:
                                print(f"failed to read {k} error : {e}")
                            time.sleep(1)

            stats_aggregated = defaultdict(lambda: 0)
            for k, v in stats.items():
                for k2 in v:
                    stats_aggregated[k2] += v[k2]

            current_time = time.perf_counter()
            total_duration = current_time - self.start_time

            if stats_aggregated["sample_count"] == 0:
                if last_one:
                    self._finish()
                    break
                continue

            stats_aggregated["average_read_duration_per_sample"] = (
                stats_aggregated["read_duration"] / stats_aggregated["sample_count"]
            )
            stats_aggregated["average_inference_duration_per_sample"] = (
                stats_aggregated["inference_duration"] / stats_aggregated["sample_count"]
            )
            stats_aggregated["average_write_duration_per_sample"] = (
                stats_aggregated["write_duration"] / stats_aggregated["sample_count"]
            )
            stats_aggregated["average_total_duration_per_sample"] = (
                stats_aggregated["total_duration"] / stats_aggregated["sample_count"]
            )
            stats_aggregated["sample_per_sec"] = stats_aggregated["sample_count"] / total_duration
            stats_aggregated["total_job_duration"] = total_duration

            to_log = [
                "sample_count",
                "sample_per_sec",
                "total_job_duration",
                "average_read_duration_per_sample",
                "average_inference_duration_per_sample",
                "average_write_duration_per_sample",
                "average_total_duration_per_sample",
            ]
            stats_for_logging = {}
            for k in to_log:
                stats_for_logging[k] = stats_aggregated[k]

            print(
                "\r",
                "sample_per_sec "
                + str(int(stats_for_logging["sample_per_sec"]))
                + " ; sample_count "
                + str(stats_for_logging["sample_count"])
                + " ",
                end="",
            )
            if self.enable_wandb:
                wandb.log(stats_for_logging)

            if last_one:
                self._finish()
                break

    def _finish(self):
        if self.current_run is not None:
            self.current_run.finish()
