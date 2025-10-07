# Dynamo Deployment with Aiconfigurator Guide

This guide walks through   
- installing aiconfigurator
- building the Dynamo container  
- generating configuration files（currently, we only support trtllm)   
- deploying Dynamo (single-node and two-node)  
- benchmarking the service and comparison  

Take [qwen3-32b-fp8](https://huggingface.co/Qwen/Qwen3-32B-FP8) model as an example.

> Currently auto configuration / script generation only support trtllm backend

# All-in-one Automation process

we're now supporting automate everything in one script, starting from configuring the deployment, generating the configs, preparing docker image and container, pulling model checkpoints, deploying the service, benchmarking and summarizing. Refer to [Automation](../tools/automation/README.md) for more details.

# Step-by-step Manual Deployment and Performance Alignment

## Methodology
First, we need to define the problem we want to solve clearly. For a given model, we need to 
understand what's the constraint, we use `ISL`, `OSL` to define the target seqeunce length, and 
`TTFT`, `TPOT` to set the SLA constraint. Let's define the problem:  
```
Can we find a config {parellel, concurrency} for given {ISL, OSL, Model, GPU}, which maximizes tokens/s/gpu, under TPOT and TTFT 
constraints.
```
Let's take a look at the pareto frontier,
```text
  Pareto Frontier:
              QWEN3_32B Pareto Frontier: tokens/s/gpu vs tokens/s/user          
    ┌──────────────────────────────────────────────────────────────────────────┐
2250┤ •• disagg                                                                │
    │ ff agg                                                                   │
    │ xx disagg best                                                           │
    │                                                                          │
1875┤  ff                                                                      │
    │   fff                                                                    │
    │     ff                                                                   │
    │      fff••                                                               │
1500┤         f •••                                                            │
    │         ff   ••••••••                                                    │
    │          ffff       •                                                    │
    │              f       •••••••                                             │
1125┤               ff            •                                            │
    │                ff            ••••                                        │
    │                  ffff            ••••x                                   │
    │                     fff              ••••                                │
 750┤                        fff               •                               │
    │                          ffffff           •                              │
    │                                ffffff      ••                            │
    │                                      fffffff ••••••                      │
 375┤                                             ff    •                      │
    │                                               fffffff•••••••••           │
    │                                                      ffffffffff          │
    │                                                                          │
   0┤                                                                          │
    └┬─────────────────┬──────────────────┬─────────────────┬─────────────────┬┘
     0                60                 120               180              240 
tokens/s/gpu                        tokens/s/user                               
```
Here the `TPOT_limit=10ms`. All datapoints shown on the pareto frontier meet `TTFT_limit=1000ms`  
Each point on the pareto frontier can represent a different config {parallel, concurrency}.  
The pareto frontier means, no matter how you change your deployment parallel strategy and 
benchmark with different concurrency, the datapoint will be under the frontier.  
What we need is the highest point on the frontier which is left to `1000ms/TPOT_limit = 100 tokens/s/user`. 
The point tagged as `x` is the one we find. This point indicates the `parellel strategy` as well as the `concurrency` level  
We can find that, the config in this parallel strategy **is potentially only best for this given 
concurrency instead of being generally better**.  
Thus we need corresponding benchmark way to make it work. Set concurrency sweep from 1 to target_concurrency 
predicted by aiconfigurator. E.g., [1 2 4 8 ... target_concurrency]
Compare the result at target_concurrency with `TTFT, TPOT, tokens/s/gpu and previous baseline you have`

> In order to reduce the impact of first batch of requests, we use `concurrency * 10` as `num_requests`
> In order to aovid undefined cache hit rate when benchmarking with random data, we delebrately disable 
cache reuse to make it fair.

## Step-by-step Manual Deployment

**Problem**:  
16 H200 in total. QWen3 32B FP8.  
ISL=4000, OSL=512, TTFT=300ms, TPOT=10ms, optimize tokens/s/gpu

If you would like to deploy by your own, when running the `aiconfigurator cli exp|default`, engine configuration files and executable scripts are automatically generated under the `--save_dir`, in the `topx` folder. The directory structure is:

```
results/QWEN3_32B_isl4000_osl1000_ttft1000_tpot20_904495
├── agg
│   ├── best_config_topn.csv
│   ├── config.yaml
│   ├── pareto.csv
│   ├── top1
│   │   ├── agg
│   │   │   ├── agg_config.yaml
│   │   │   ├── k8s_deploy.yaml
│   │   │   └── node_0_run.sh 
│   │   └── generator_config.yaml
│   ...
├── disagg
│   ├── best_config_topn.csv
│   ├── config.yaml
│   ├── pareto.csv
│   ├── top1
│   │   ├── disagg
│   │   │   ├── decode_config.yaml
│   │   │   ├── k8s_deploy.yaml
│   │   │   ├── node_0_run.sh
│   │   │   └── prefill_config.yaml
│   │   └── generator_config.yaml
│   ...
└── pareto_frontier.png
```

Here, `agg_config.yaml`, `prefill_config.yaml`, and `decode_config.yaml` are TRTLLM engine configuration files, and `node_x_run.sh` are the executable scripts. `k8s_deploy.yaml` is for deployment in k8s. In this guide, we're not using k8s.

For multi-node setups, there will be multiple `node_x_run.sh` scripts (one per node), each invoking the same TRTLLM engine config file. By default, `node_0_run.sh` starts **both the frontend service and the workers, assuming ETCD and NATS are already running on node0, while other nodes only start the workers**. Therefore, in multi-node deployments, please specify `--head_node_ip` to indicate the IP address of node0.

FIXME, info below is all outdated. Need to be updated

Typically, the command is:

````bash
aiconfigurator cli default \
  --system h200_sxm \
  --model QWEN3_32B \
  --isl 5000 \
  --osl 1000 \
  --ttft 2000 \
  --tpot 50 \
  --save_dir results \
  --total_gpus 16 \
  --model_path /workspace/model_hub/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --head_node_ip x.x.x.x
````

Since TRTLLM has many custom parameters, run:

```bash
aiconfigurator cli --help
```

to see all supported options. We haven’t listed all trtllm configurations here. Feel free to modify the generated configuration file to include any additional ones [(trtllm args)](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.20.0/tensorrt_llm/llmapi/llm_args.py).

To customize parameters per worker type, prefix them with `prefill_`, `decode_`, or `agg_`. For example:

```bash
aiconfigurator cli default \
  --system h200_sxm \
  --model QWEN3_32B \
  --isl 5000 \
  --osl 1000 \
  --ttft 2000 \
  --tpot 50 \
  --save_dir results \
  --total_gpus 16 \
  --model_path /workspace/model_hub/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --head_node_ip x.x.x.x \
  --free_gpu_memory_fraction 0.9 \        # default for all worker types
  --prefill_free_gpu_memory_fraction 0.8 \  # for prefill workers
  --decode_free_gpu_memory_fraction 0.5     # for decode workers
```

At runtime, copy the generated `backend_configs` directory to each node and execute the corresponding script:

```bash
# On node0
bash node_0_run.sh

# On other nodes
bash node_x_run.sh
```

> Note: The generated configs are for deploying 1 replica instead of the cluster (defined as total_gpus). We'll bridge this gap in future.

---

## Prerequisites

* Docker with GPU support

---

## 1. Environment Setup

### 1.1 Install aiconfigurator

Use a minimal Ubuntu base image with python installed.

```bash
# Install Git LFS
apt-get update && apt-get install -y git-lfs

# Clone the repo
git clone https://github.com/ai-dynamo/aiconfigurator.git
cd aiconfigurator

# Install build tools and aiconfigurator
pip3 install "."
```

### 1.2 Build the Dynamo Container
In this example, we're using Dynamo 0.5.0, please switch to release/0.5.0 first.
```bash
# other version of trtllm can be used as well
# currently dynamo is at version 0.4.0, indicated in the tag
./container/build.sh \
  --framework TRTLLM \
  --tensorrtllm-pip-wheel tensorrt-llm==1.0.0rc6 \
  --tag dynamo:0.4.0-trtllm-1.0.0rc6
```

> Please refer to [Dynamo Getting Started](https://docs.nvidia.com/dynamo/latest/get_started.html) for detailed dynamo installation

### 1.3 Download model checkpoint
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen3-32B-FP8 --local-dir /raid/hub/qwen3-32b-fp8
```
Please modify based on your own path '/raid/hub/qwen3-32b-fp8'

---

## 2. Running etcd and NATS

On **Node 0**, start etcd and NATS.io:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

---

## 3. Single-Node Deployment

### 3.1 Generate Configuration with aiconfigurator

```bash
aiconfigurator cli \
  --system h200_sxm \
  --version 1.0.0rc3 \
  --isl 5000 \
  --osl 1000 \
  --ttft 1000 \
  --tpot 10 \
  --save_dir ./ \
  --model QWEN3_32B \
  --model_path /workspace/model_hub/qwen3-32b-fp8 \
  --served_model_name Qwen3/Qwen3-32B-FP8 \
  --total_gpus 8 \
  --head_node_ip 0.0.0.0 \
  --generated_config_version 1.0.0rc4 \
  --prefill_free_gpu_memory_fraction 0.9 \
  --free_gpu_memory_fraction 0.7 \
  --decode_free_gpu_memory_fraction 0.5
```
We use 1.0.0rc3 (our latest data) for aiconfigurator and we can support generate configurations for running with trtllm 1.0.0rc4 worker.  
*--model* is for aiconfigurator and *--served_model_name* is for dynamo deployment  
> For other supported configurations, please run `aiconfigurator cli --help`.

### 3.2 Verify Generated Configuration

engine configuration files and executable scripts are automatically generated under the `--save_dir`, in the `backend_configs` folder. The directory structure is:

````
backend_configs/
├── agg/
│   ├── agg_config.yaml
│   └── node_0_run.sh
└── disagg/
│   ├── decode_config.yaml
│   ├── prefill_config.yaml
│   ├── node_0_run.sh
│   ├── node_1_run.sh
│   └── ...
└──
````

### 3.3 Launch the Dynamo Container

```bash
cd ..
docker run --gpus all --net=host --ipc=host \
  -v $(pwd):/workspace/mount_dir \
  -v /raid/hub:/workspace/model_hub/ \
  --rm -it dynamo:0.4.0-trtllm-1.0.0rc4
```


### 3.4 Deploy the service

Inside the container:

```bash
cd /workspace/mount_dir/dynamo/components/backends/trtllm

# Copy generated configs from save_dir
cp -r ${your_save_dir}/QWEN3_32B_isl5000_osl1000_ttft1000_tpot50_*/backend_configs/* ./

# Launch dynamo
bash disagg/node_0_run.sh
```

> **Tip:** If you see a Triton version mismatch error, reinstall Triton:
>
> ```bash
> pip uninstall -y triton
> pip install triton==3.3.1
> ```

### 3.5 Test the Service

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "Qwen3/Qwen3-32B-FP8",
    "messages": [
      { "role": "user", "content": "Introduce yourself" }
    ],
    "stream": true
  }'
```

### 3.6 Benchmark

---

## 4. Two-Node Deployment

### 4.1 Generate Configuration for Two Nodes

```bash
# For head_node_ip, ensure that the IP passed here corresponds to node 0, etcd and NATS.io have already been started on node 0 in Step 2
aiconfigurator cli \
  --system h200_sxm \
  --version 1.0.0rc3 \
  --isl 5000 \
  --osl 1000 \
  --ttft 200 \
  --tpot 8 \
  --save_dir ./ \
  --model QWEN3_32B \
  --model_path /workspace/model_hub/qwen3-32b-fp8 \
  --served_model_name Qwen3/Qwen3-32B-FP8 \
  --total_gpus 16 \
  --head_node_ip NODE_0_IP \
  --generated_config_version 1.0.0rc4 \
  --prefill_free_gpu_memory_fraction 0.8 \
  --free_gpu_memory_fraction 0.7 \
  --decode_free_gpu_memory_fraction 0.5
```

> Note that even if `--total_gpus 16`, the optimal configuration generated by aiconfigurator may not require 16 GPUs. If only 8 GPUs are needed, it may produce just a `node_0_run.sh`, which can then be executed on each node.

Refer to the single node example to run the container on both node 0 and node 1.

### 4.2 Deploy on Node 0
Inside the container:
```bash
cd /workspace/mount_dir/dynamo/components/backends/trtllm
cp -r QWEN3_32B_isl5000_osl1000_ttft200_tpot8_*/backend_configs/* ./
bash disagg/node_0_run.sh
```

### 4.3 Deploy on Node 1
Inside the container:
```bash
cd /workspace/mount_dir/dynamo/components/backends/trtllm
cp -r QWEN3_32B_isl5000_osl1000_ttft200_tpot8_*/backend_configs/* ./
bash disagg/node_1_run.sh
```

### 4.4 Test the Service

```bash
curl http://NODE_0_IP:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "Qwen3/Qwen3-32B-FP8",
    "messages": [
      { "role": "user", "content": "Introduce yourself" }
    ],
    "stream": true
  }'
```

---
