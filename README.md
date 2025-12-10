<h1 align="center">Bigger or Longer? Test-Time-Scaling is More Efficient<br> than Model-Size-Scaling for Visual Grounding  </h1>


This repository releases the official implementation of Bigger or Longer?Test-Time-Scaling is More Efficient than Model-Size-Scaling for Visual Grounding. 

这里差一个文章地址

## Abstract

Visual grounding is an essential capability of Visual Language Models (VLMs) to understand the real physical world. Previous state-of-the-art grounding visual language models tend to go bigger (scale the model size), which makes them very heavy for deployment and slow for inference, making it difficult to apply to edge systems to obtain grounding results in real time. However, we notice that the sizes of visual encoders are the same and they only scale the language parts. Small VLMs fall behind larger VLMs in grounding because of the difference in language understanding capability rather than visual information handling, and we thoroughly verify it. To mitigate the gap, we introduce *EGM*: a method to scale the test-time computation (*#generated tokens*) rather than the model sizes. Going longer is not only deployment-friendly, but also yields better end-to-end latency as the cost of each token is much cheaper. On the RefCOCO benchmark, our **EGM-8B** demonstrates **91.4 IoU** with average 737ms (**5.9$\times$ faster**) latency while **Qwen3-VL-235B** demands 4,320ms to achieve **90.5 IoU**. To validate our approach's generality, we further set up a new amodal grounding setting that requires the model to predict both the visible and occluded parts of the objects. Experiments show our method can consistently and significantly improve the vanilla grounding and amodal grounding capabilities of small models to be on par with or outperform the larger models, thereby improving the efficiency for visual grounding. 

See our website for more details : https://zgq1879.github.io/EGM_website/

### Table of Contents  <!-- omit in toc -->

  - [Bigger or Longer? Test-Time-Scaling is More Efficient than Model-Size-Scaling for Visual Grounding](#bigger-or-longer-test-time-scaling-is-more-efficient-than-model-size-scaling-for-visual-grounding-1)
  - [SFT Training](#sft-training)
  - [RL Training](#rl-training)
  - [Evaluation](#evaluation)
  - [Acknowledgment](#acknowledgment)

## Bigger or Longer? Test-Time-Scaling is More Efficient than Model-Size-Scaling for Visual Grounding

### Motivation of the EGM method

Conventional VLMs **"Go Bigger"** (Left) by scaling up model size, which hinders deployment and latency. In contrast, our EGM **"Go Longer"** (Right) scales test-time inference instead; by outputting more tokens with a smaller model to bridge the understanding gap, we achieve on-par performance with significantly better efficiency.


<div align="center">
  <img src="images/teaser.png" width="90%"/>
</div>

### Performance of the EGM method

Our **EGM** models (2B/4B/8B) demonstrate superior scaling efficiency compared to simply increasing model size ("Bigger"). Notably, **EGM-8B** outperforms the massive Qwen3-VL-235B-Instruct ('-I') and Thinking ('-T') models in accuracy, while respectively achieving significant **5.9×** and **18.9×** speedups in latency.

<div align="center">
  <img src="images/tradeoff.png" width="90%"/>
</div>



## SFT Training

Please refer to `sft/README.md` for SFT training.

## RL Training

We provide the Grounding training and inference workflow for the **EGM-8B** model as the primary example below.

### 1. Installation

```bash
cd qwen3vl_rl
conda create -n verl_egm -y -c nvidia/label/cuda-12.8.0 -c nvidia -c conda-forge python=3.12 cuda-toolkit=12.8
conda activate verl_egm
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "sglang[all]==0.5.2" --no-cache-dir && pip install torch-memory-saver --no-cache-dir
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard 
pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"
wget https://github.com/Efficient-Large-Model/flash-attention-builder/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl 
pip install --no-cache-dir flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install --no-cache-dir flashinfer-python==0.3.1
pip install -e .[vllm]
pip uninstall decord
```

### 2. Model 

| Training Phase | Model | HuggingFace |
| :--- | :--- | :--- |
| Supervised Fine-Tuning (SFT) | EGM-Qwen3-VL-4B-SFT | [Link](https://huggingface.co/JamesZGQ/EGM-4B-SFT) |
| Reinforcement Learning | EGM-Qwen3-VL-4B-v1 | [Link](https://huggingface.co/JamesZGQ/EGM-4B) |
| Supervised Fine-Tuning (SFT) | EGM-Qwen3-VL-8B-SFT | [Link](https://huggingface.co/JamesZGQ/EGM-8B-SFT) |
| Reinforcement Learning | EGM-Qwen3-VL-8B-v1 | [Link](https://huggingface.co/JamesZGQ/EGM-8B) |

### 3. Data Preparation

Please download the training and testing datasets before proceeding.

[Training annotations](https://huggingface.co/datasets/JamesZGQ/EGM_Datasets/tree/main/train_data) |
[Testing annotations](https://huggingface.co/datasets/JamesZGQ/EGM_Datasets/tree/main/eval_data) |
[Images tar1](https://huggingface.co/datasets/JamesZGQ/EGM_Datasets/blob/main/coco.tar) |
[Images tar2](https://huggingface.co/datasets/JamesZGQ/EGM_Datasets/blob/main/coco_flip.tar)

```bash
# Set your environment variables
export BASE_IMG_PATH=${YOUR_BASE_IMG_PATH}
export OUTPUT_DIR=${YOUR_DATA_DIR}
export VAL_DIR=${YOUR_VAL_DIR}
export TRAIN_JSON=${QWEN3_8B_GROUNDING_TRAIN_JSON}

# Run preprocessing scripts
bash examples/data_preprocess/grounding_val.sh
bash examples/data_preprocess/grounding_all.sh
```

### 4. Training

Reinforcement Learning  is conducted based on the SFT checkpoint. The default configuration utilizes 8 GPUs. You may customize the distributed training settings via the `trainer.nnodes` and `trainer.n_gpus_per_node` arguments. The data directory `(DATA_DIR)` should be the same as output directory `(OUTPUT_DIR)` in Data Preparation.

```bash
export WANDB_BASE_URL=${YOUR_WANDB_BASE_URL}   
export WANDB_API_KEY=${YOUR_WANDB_API_KEY} 
export DATA_DIR=${YOUR_DATA_DIR}
export PROJECT_NAME=${YOUR_PROJECT_NAME}
export TASK_NAME=${YOUR_TASK_NAME}
export OUTPUT_DIR=${YOUR_OUTPUT_DIR}
export MODEL_PATH=${YOUR_MODEL_PATH}

bash scripts/grounding_qwen.sh
```

### 5. Inference and Evaluation


To evaluate the model, update sglang with `pip install sglang==0.5.5.post3` and use the command provided below.

**Note:** The RefCOCO benchmark consists of eight distinct JSON files. Consequently, you must run the evaluation script sequentially for each of the 8 files to obtain the complete benchmark results.

```bash
export MODEL_PATH=${YOUR_MODEL_PATH}
export DATA_JSON=${DATA_JSON}
export OUTPUT_DIR=${YOUR_OUTPUT_DIR}
export BASE_IMG_PATH=${YOUR_BASE_IMG_PATH}

bash scripts/sglang_infer.sh
```

We also support evaluation with vLLM, update vLLM with `pip install vllm==0.11.0` and use the command provided below:

```bash
export MODEL_PATH=${YOUR_MODEL_PATH}
export DATA_JSON=${DATA_JSON}
export OUTPUT_DIR=${YOUR_OUTPUT_DIR}
export BASE_IMG_PATH=${YOUR_BASE_IMG_PATH}

bash scripts/vllm_infer.sh
```

### 6. Handling Environment Issues

1. If you encounter runtime errors related to FlashInfer, such as `GLIBC_2.32' not found (required by .../flashinfer/.../sampling.so)`, you can work around this by disabling the FlashInfer sampler: set the environment variable `VLLM_USE_FLASHINFER_SAMPLER=0` or `SGLANG_IS_FLASHINFER_AVAILABLE=false` before launching the training or inference command.

## Evaluation

## Acknowledgment

This repository benefits from [Qwen3VL](https://github.com/QwenLM/Qwen3-VL), [InternVL](https://github.com/OpenGVLab/InternVL), [verl](https://github.com/volcengine/verl) and [verl-internvl](https://github.com/Weiyun1025/verl-internvl).

Thanks for their wonderful works and their efforts to further promote LLM research.

---

