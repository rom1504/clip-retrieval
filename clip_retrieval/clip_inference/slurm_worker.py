"""
Read environment variables and distribute work work to this rank.
This script is called by the slurm distributor.

Will launch a worker.
"""

import os
import json

from torch.cuda import set_device

from clip_retrieval.clip_inference.worker import worker


def get_task_list(num_tasks, world_size, global_rank, local_rank):
    """Get the list of tasks to process."""
    tasks_per_worker = num_tasks // world_size

    # Assign a subset of tasks to each worker
    start = global_rank * tasks_per_worker
    end = start + tasks_per_worker

    # If the tasks don't divide evenly
    # then we should redistribute the remainder
    if global_rank < num_tasks % world_size:
        start += global_rank
        end += global_rank + 1
    else:
        start += num_tasks % world_size
        end += num_tasks % world_size

    tasks = list(range(start, end))

    print(f"worker global rank:{global_rank}\tlocal rank: {local_rank}\tprocessing tasks {tasks}")

    return tasks


def slurm_worker():
    """Distribute work to this job and launch a job."""

    # Read environment variables
    # These are set by slurm_distributor or SLURM itself
    num_tasks = int(os.environ["NUM_TASKS"])
    global_rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    # Read the worker args from the file
    with open(os.environ["WORKER_ARGS_PATH"], "r", encoding="utf-8") as worker_args_file:
        worker_args = json.load(worker_args_file)

    # Find the range of tasks to process
    tasks = get_task_list(num_tasks, world_size, global_rank, local_rank)

    # set device
    set_device(local_rank)

    # Launch the worker
    worker(tasks, **worker_args)


if __name__ == "__main__":
    slurm_worker()
