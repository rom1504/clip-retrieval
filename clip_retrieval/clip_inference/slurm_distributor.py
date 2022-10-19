"""Distribute work using SLURM"""

import os
import time
import json
import subprocess
from datetime import datetime

TIMESTAMP = datetime.now().timestamp()


class SlurmDistributor:
    """distribute work across a collection of slurm jobs"""

    def __init__(self, tasks, worker_args, slurm_args):
        self.num_tasks = len(tasks)
        self.worker_args = worker_args
        self.slurm_args = slurm_args

        self.job_timeout = slurm_args.pop("job_timeout")
        self.verbose_wait = slurm_args.pop("verbose_wait")

    def __call__(self):
        """
        Create a sbatch file, submit it to slurm, and wait for it to finish.
        """
        # pop the cache path from the slurm args to remove it
        cache_path = self.slurm_args.pop("cache_path")

        # create the cache path if it doesn't exist
        if cache_path is None:
            cache_path = os.path.expanduser("~/.cache")

        os.makedirs(cache_path, exist_ok=True)

        # make the filenames unique using the current timestamp
        sbatch_script_path = os.path.join(cache_path, f"sbatch_script_{TIMESTAMP}.sh")

        # save the file to the cache path
        with open(sbatch_script_path, "w", encoding="utf-8") as sbatch_file:
            sbatch_file.write(
                self._generate_sbatch(cache_path=cache_path, slurm_args=self.slurm_args, worker_args=self.worker_args)
            )

        # now we need to run the job
        status = self._run_job(sbatch_script_path)

        # interpret the results
        if status == "success":
            print("job succeeded")
            return True
        elif status == "failed":
            print("job failed")
            return False
        else:
            print("exception occurred")
            return False

    def _run_job(self, sbatch_file):
        """
        Run a job and wait for it to finish.
        """
        try:
            job_id = self._start_job(sbatch_file)

            print(f"waiting for job {job_id}")

            timeout = self.job_timeout

            if timeout is None:
                print("You have not specified a timeout, defaulting to 2 weeks.")
                timeout = 1.21e6

            status = self._wait_for_job_to_finish(job_id=job_id, timeout=timeout)

            if not status:
                print(f"canceling {job_id}")
                subprocess.check_output(["scancel", job_id]).decode("utf8")
                status = self._wait_for_job_to_finish(job_id)
                print("job cancelled")
                return "failed"
            else:
                print("job succeeded")
                return "success"
        except Exception as e:  # pylint: disable=broad-except
            print(e)
            return "exception occurred"

    def _wait_for_job_to_finish(self, job_id, timeout=30):
        t = time.time()
        while 1:
            if time.time() - t > timeout:
                return False
            time.sleep(1)
            if self._is_job_finished(job_id):
                return True

    def _is_job_finished(self, job_id):
        status = subprocess.check_output(["squeue", "-j", job_id]).decode("utf8")

        if self.verbose_wait:
            print(f"job status is {status}")

        return status == "slurm_load_jobs error: Invalid job id specified" or len(status.split("\n")) == 2

    def _start_job(self, sbatch_file):
        """start job"""
        args = ["sbatch"]
        args.append(sbatch_file)
        sbatch_output = subprocess.check_output(args).decode("utf8")
        lines = sbatch_output.split("\n")

        lines = [line for line in lines if "Submitted" in line]
        if len(lines) == 0:
            raise ValueError(f"slurm sbatch failed: {sbatch_output}")

        parsed_sbatch = lines[0].split(" ")
        job_id = parsed_sbatch[3].strip()
        return job_id

    def _write_json_worker_args(self, worker_args, cache_path):
        """write the worker args to a json file"""
        worker_args_path = os.path.join(cache_path, f"worker_args_{TIMESTAMP}.json")
        with open(worker_args_path, "w", encoding="utf-8") as worker_args_file:
            json.dump(worker_args, worker_args_file, indent=4)
        return worker_args_path

    def _generate_sbatch(self, cache_path, slurm_args, worker_args):
        """
        Generate sbatch for a worker.

        sbatch: allows you to specify a configuration and task in a file
            - https://slurm.schedmd.com/sbatch.html
        """
        # write the worker args to a file
        worker_args_path = self._write_json_worker_args(worker_args, cache_path)

        venv = os.environ["VIRTUAL_ENV"]
        scomment = ("--comment " + slurm_args["job_comment"]) if ["job_comment"] is not None else ""
        sbatch_scomment = (
            ("#SBATCH --comment " + slurm_args["job_comment"]) if slurm_args["job_comment"] is not None else ""
        )
        nodelist = ("#SBATCH --nodelist " + slurm_args["nodelist"]) if slurm_args["nodelist"] is not None else ""
        exclude = ("#SBATCH --exclude " + slurm_args["exclude"]) if slurm_args["exclude"] is not None else ""

        return f"""#!/bin/bash
# Define sbatch config, use exclusive to capture all resources in each node
#SBATCH --partition={slurm_args["partition"]}
#SBATCH --job-name={slurm_args["job_name"]}
#SBATCH --output={cache_path}/slurm-%x_%j.out
#SBATCH --nodes={slurm_args["nodes"]}
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-gpu=6
#SBATCH --gres=gpu:8
#SBATCH --exclusive

{sbatch_scomment}
{nodelist}
{exclude}

# Environment variables for the inner script
export NUM_TASKS={self.num_tasks}
export WORLD_SIZE={slurm_args["nodes"] * 8} # 8 gpus per node
export WORKER_ARGS_PATH={worker_args_path}

# Run the internal script
source {venv}/bin/activate
srun --cpu_bind=v --accel-bind=gn {scomment} clip-retrieval inference.slurm_worker
"""
