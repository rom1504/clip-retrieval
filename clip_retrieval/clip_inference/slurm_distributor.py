"""Distribute work using SLURM"""

import os
import time
import subprocess
from multiprocessing.pool import ThreadPool


class SlurmDistributor:
    """distribute work across a collection of slurm nodes"""

    def __init__(self, tasks, worker_args, slurm_args):
        self.worker_args = worker_args
        self.slurm_args = slurm_args

        self.job_timeout = slurm_args.pop("job_timeout")

        # calculate world info for distributing work, assume 1 GPU/node
        self.tasks = tasks
        self.num_tasks = len(self.tasks)
        self.tasks_per_node = self.num_tasks // self.slurm_args["nodes"]

        if self.tasks_per_node <= 0:
            print("There are more nodes than tasks...reducing the number of requested nodes.")

            while self.tasks_per_node <= 0:
                # reduce the number of nodes by one until we no longer have an excess
                self.slurm_args["nodes"] -= 1
                self.tasks_per_node = self.num_tasks // self.slurm_args["nodes"]

            new_worker_count = self.slurm_args["nodes"]
            print(f"Now using only: {new_worker_count} workers")

    def __call__(self):
        """
        Distribute task amongst multiple workers that will be started with sbatch.
        """
        cache_path = self.slurm_args.pop("cache_path")
        if cache_path is None:
            cache_path = os.path.expanduser("~/.cache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)

        task_assignments = {node_id: self._get_worker_tasks(node_id) for node_id in range(self.slurm_args["nodes"])}

        # create the sbatch files to run
        sbatch_files = []
        for node_id, tasks in task_assignments.items():
            # save the sbatch file to the cache
            sbatch_script_path = os.path.join(cache_path, f"sbatch_{node_id}.script")
            sbatch_output_path = os.path.join(cache_path, f"sbatch_{node_id}.out")

            with open(sbatch_script_path, "w", encoding="utf-8") as sbatch_file:
                sbatch_file.write(
                    self._generate_sbatch(
                        tasks=tasks, output_file=sbatch_output_path, cache_folder=cache_path, **self.slurm_args
                    )
                )

            sbatch_files.append(sbatch_script_path)

        # create a thread group to manage all the jobs we are about to start
        all_results = {}
        with ThreadPool(self.slurm_args["nodes"]) as p:
            # create a surrogate function for the task of running jobs
            run_worker = lambda node_id: self._run_job(sbatch_files[node_id])

            for result in p.imap_unordered(run_worker, range(self.slurm_args["nodes"])):
                all_results.update(result)

        print(all_results)

    def _run_job(self, sbatch_file):
        """
        Run a job and wait for it to finish
        """

        job_id = self._start_job(sbatch_file)

        print(f"waiting for job {job_id}")

        timeout = self.job_timeout

        if timeout is None:
            print("You have not specified a timeout, defaulting to 24 hours.")
            timeout = 60 * 60 * 24

        status = self._wait_for_job_to_finish(job_id=job_id, timeout=timeout)

        if not status:
            print(f"canceling {job_id}")
            subprocess.check_output(["scancel", job_id]).decode("utf8")
            status = self._wait_for_job_to_finish(job_id)
            print("job cancelled")

            # TODO: better reporting
            return {job_id: "failed"}
        else:
            print("job succeeded")
            return {job_id: "success"}

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

    def _get_worker_tasks(self, node_id):
        """
        Return a list of the tasks this worker is responsible for.
        """
        # calculate this node's distribution of work
        start_index = self.tasks_per_node * node_id

        # account for uneven work distribution
        stop_index = min(start_index + self.tasks_per_node, self.num_tasks)

        return self.tasks[start_index:stop_index]

    def _get_formated_worker_args(self):
        """
        Format the worker arguments to be used for a CLI command.

        Fire is sensitive to argument parsing:
            - more reading here: https://google.github.io/python-fire/guide/#argument-parsing
        """

        arguments = []

        for key, value in self.worker_args.items():
            # the input_dataset list of strings need special attention
            if key == "input_dataset":
                # wrap inner values with strings
                escaped_values = ", ".join([f'\\"{i}\\"' for i in value])
                value = f"[{escaped_values}]"

            # add arguments
            arguments.append(f'--{key}="{value}"')

        return " ".join(arguments)

    def _generate_sbatch(
        self, tasks, output_file, cache_folder, job_name, partition, nodes, job_comment, nodelist, exclude
    ):
        """
        Generate sbatch for a worker.

        Resources:
        sbatch: allows you to specify a configuration and task in a file
            - https://slurm.schedmd.com/sbatch.html
        gres: for specifying the resources used in a node
            - https://slurm.schedmd.com/gres.html
        """
        ntasks_per_node = 1
        gres = "gpu:1"
        venv = os.environ["VIRTUAL_ENV"]
        scomment = ("--comment " + job_comment) if job_comment is not None else ""
        sbatch_scomment = ("#SBATCH --comment " + job_comment) if job_comment is not None else ""
        nodelist = ("#SBATCH --nodelist " + nodelist) if nodelist is not None else ""
        exclude = ("#SBATCH --exclude " + exclude) if exclude is not None else ""

        return f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --job-name={job_name}
#SBATCH --nodes {nodes}
#SBATCH --ntasks-per-node {ntasks_per_node}
#SBATCH --gres={gres}
#SBATCH --output={output_file}
#SBATCH --exclusive
{sbatch_scomment}
{nodelist}
{exclude}
{self._get_slurm_boilerplate(cache_folder=cache_folder)}
source {venv}/bin/activate
/opt/slurm/sbin/srun {scomment} --cpu_bind=v --accel-bind=gn clip-retrieval inference.worker --tasks="{tasks}" {self._get_formated_worker_args()}
"""

    def _get_slurm_boilerplate(self, cache_folder):
        return f"""
export HF_HOME="{cache_folder}/hf_home"
export WANDB_CACHE_DIR="{cache_folder}/wandb_cache"
"""
