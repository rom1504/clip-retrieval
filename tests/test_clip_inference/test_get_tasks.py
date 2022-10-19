import pytest

from clip_retrieval.clip_inference.slurm_worker import get_task_list


def test_uneven_tasks():
    """Test task distribution for an uneven distribution of tasks/workers."""

    world_size = 3
    num_tasks = 11

    SOLUTION = {
        0: [0, 1, 2, 3],
        1: [4, 5, 6, 7],
        2: [8, 9, 10],
    }

    # Test that the tasks are distributed as evenly as possible
    for global_rank in range(world_size):
        tasks = get_task_list(num_tasks=num_tasks, world_size=world_size, global_rank=global_rank, local_rank=-1)
        assert tasks == SOLUTION[global_rank]


def test_even_tasks():
    """Test task distribution for an even distribution of tasks/workers."""

    world_size = 3
    num_tasks = 9

    SOLUTION = {
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7, 8],
    }

    # Test that the tasks are distributed as evenly as possible
    for global_rank in range(world_size):
        tasks = get_task_list(num_tasks=num_tasks, world_size=world_size, global_rank=global_rank, local_rank=-1)
        assert tasks == SOLUTION[global_rank]
