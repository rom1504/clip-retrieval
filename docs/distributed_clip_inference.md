# distributed clip inference

If you want to generate billion of clip embeddings, read this.

This guide is about using pyspark to run clip inference in multiple node and using multiple gpus.

you may also be interested by [distributed img2dataset](https://github.com/rom1504/img2dataset/blob/main/examples/distributed_img2dataset_tutorial.md)

We will be assuming ubuntu 20.04.

## Setup the master node

On the master node:

First download spark:
```bash
wget https://archive.apache.org/dist/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz
tar xf spark-3.2.0-bin-hadoop3.2.tgz
```

Then download clip inference:
```bash
rm -rf clip_retrieval.pex
wget https://github.com/rom1504/clip-retrieval/releases/latest/download/clip_retrieval.tgz -O clip_retrieval.tgz
wget https://github.com/rom1504/clip-retrieval/releases/latest/download/clip_retrieval_torch.tgz -O clip_retrieval_torch.tgz
tar xf clip_retrieval.tgz
tar xf clip_retrieval_torch.tgz
```

If the master node cannot open ports that are visible from your local machine, you can do a tunnel between your local machine and the master node to be able to see the spark ui (at http://localhost:8080)
```bash
ssh -L 8080:localhost:8080 -L 4040:localhost:4040 master_node
```


## Setup the worker nodes

### ssh basic setup

Still in the master node, create a ips.txt with the ips of all the nodes

```bash
ssh-keyscan `cat ips.txt` >> ~/.ssh/known_hosts
```

You may use a script like this to fill your .ssh/config file
```
def generate(ip):
    print(
        f"Host {ip}\n"
        f"        HostName {ip}\n"
        "        User ubuntu\n"
        "        IdentityFile ~/yourkey.pem"
        )

with open("ips.txt") as f:
    lines = f.readlines()
    for line in lines:
        generate(line.strip())
```
python3 generate.py >> ~/.ssh/config

Install pssh with `sudo apt install pssh`

Pick the right username (MASTER_USER) for the master node, and (USER) for the worker nodes, then run this to check your parallel ssh setup:
```bash
USER=rom1504
```

Optionally, if another node than the current one has access to the worker nodes, you may need to add a ssh key to all the nodes with:
```
for IP in `cat ips.txt`
do
        ssh-copy-id -i the_new_id_rsa $USER@$IP
done
```

Check you can connect to all the nodes with:
```
parallel-ssh -l $USER -i -h  ips.txt uname -a
```

##### Install some packages

```bash
parallel-ssh -l $USER -i -h  ips.txt "sudo apt update"
parallel-ssh -l $USER -i -h  ips.txt "sudo apt install openjdk-11-jre-headless libgl1 htop tmux bwm-ng sshfs python3-distutils python3-apt python3.8 -y"
```


#### [Optional] Network setting on aws

put in same VPC and security group and allow inbound

##### Download clip retrieval on all nodes

Download clip retrieval on all node by retrying this N times until parallel ssh says success for all:
```bash

parallel-ssh -i -h ips.txt "rm -rf clip_retrieval.pex"
parallel-ssh -i -h ips.txt "wget https://github.com/rom1504/clip-retrieval/releases/latest/download/clip_retrieval.tgz -O clip_retrieval.tgz"
parallel-ssh -i -h ips.txt "wget https://github.com/rom1504/clip-retrieval/releases/latest/download/clip_retrieval_torch.tgz -O clip_retrieval_torch.tgz"
parallel-ssh -i -h ips.txt "tar xf clip_retrieval.tgz"
parallel-ssh -i -h ips.txt "tar xf clip_retrieval_torch.tgz"
```

##### Download spark on workers

parallel-ssh -l $USER -i -h  ips.txt  "wget https://archive.apache.org/dist/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz"
parallel-ssh -l $USER -i -h  ips.txt  "tar xf spark-3.2.0-bin-hadoop3.2.tgz"

echo '[{"id":{"componentName": "spark.worker","resourceName":"gpu"},"addresses":["0","1","2","3","4","5","6","7"]}]' > gpufile
parallel-scp -h ips.txt gpufile /home/ubuntu/gpufile

#### Start the master node

When you're ready, you can start the master node with:

```bash
./spark-3.2.0-bin-hadoop3.2/sbin/start-master.sh -p 7077
```


#### Start the worker nodes

When you're ready, you can start the worker nodes with:

```bash
parallel-ssh -l $USER -i -h  ips.txt  'SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=8 -Dspark.worker.resourcesFile=/home/ubuntu/gpufile" ./spark-3.2.0-bin-hadoop3.2/sbin/start-worker.sh -c 16 -m 24G "spark://172.31.44.42:7077"'
```

Replace 172.31.44.42 by the master node ip.


#### Stop the worker nodes

When you're done, you can stop the worker nodes with:

```bash
parallel-ssh -l $USER -i -h  ips.txt "rm -rf ~/spark-3.2.0-bin-hadoop3.2/work/*"
parallel-ssh -l $USER -i -h  ips.txt  "pkill java"
```

#### Stop the master node

When you're done, you can stop the master node with:

```bash
pkill java
```


### Running clip inference on it

Once your spark cluster is setup, you're ready to start clip inference in distributed mode.
Make sure to open your spark UI, at http://localhost:8080 (or the ip where the master node is running)

Save this script to inference.py.

Then run `./clip_retrieval.pex/__main__.py inference.py`

```python
from clip_retrieval import clip_inference
import shutil
import os
from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

from pyspark import SparkConf, SparkContext

def create_spark_session():
    # this must be a path that is available on all worker nodes
    
    os.environ['PYSPARK_PYTHON'] = "/home/ubuntu/clip_retrieval.pex/__main__.py"
    spark = (
        SparkSession.builder
        .config("spark.submit.deployMode", "client") \
        .config("spark.executorEnv.PEX_ROOT", "./.pex")
        .config("spark.task.resource.gpu.amount", "1")
        .config("spark.executor.resource.gpu.amount", "8")
        #.config("spark.executor.cores", "16")
        #.config("spark.cores.max", "48") # you can reduce this number if you want to use only some cores ; if you're using yarn the option name is different, check spark doc
        .config("spark.driver.port", "5678")
        .config("spark.driver.blockManager.port", "6678")
        .config("spark.driver.host", "172.31.44.42")
        .config("spark.driver.bindAddress", "172.31.44.42")
        .config("spark.executor.memory", "16G") # make sure to increase this if you're using more cores per executor
        .config("spark.executor.memoryOverhead", "8G")
        .config("spark.task.maxFailures", "100")
        .master("spark://172.31.44.42:7077") # this should point to your master node, if using the tunnelling version, keep this to localhost
        .appName("spark-stats")
        .getOrCreate()
    )
    return spark

spark = create_spark_session()

clip_inference(input_dataset="pipe:aws s3 cp --quiet s3://laion-us-east-1/laion-data/laion2B-data/{000000..231349}.tar -", output_folder="s3://laion-us-east-1/my_test_embedding2", input_format="webdataset", enable_metadata=True, write_batch_size=1000000, num_prepro_workers=8, batch_size=512, cache_path=None, enable_wandb=True, distribution_strategy="pyspark", clip_model="ViT-B/14")
```

## Some benchmarks

Using 1 node with 8 a100 on aws, using s3 as input and output:
* 7000 sample/s on 8 a100 on vit-b / 32 : 2500 for one gpu so it's resizing bottlenecked
* 7000 sample/s on 8 a100 on vit-b / 16 : 1100 sample/s for one gpu so it's still bottlenecked by resizing but much better
* 2500 sample/s on 8 a100 on vit-l / 14 : 312 sample/s for one gpu so it's optimal

on 4 such nodes, the speed are multiplied by 4 which is optimal.
