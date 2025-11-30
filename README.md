# EGM: Efficient Grounding Model

This repository releases the official implementation of **EGM** (Efficient Grounding Model). Our approach enables an 8B parameter model to surpass the reasoning performance and efficiency of a 235B model on RefCOCO benchmarks. The training framework is built upon [verl](https://github.com/volcengine/verl) and [verl-internvl](https://github.com/Weiyun1025/verl-internvl).

## Example Usage

As the InternVL and Qwen model series require different training environments, we provide the Grounding training workflow for the **Qwen3-VL-8B** model as the primary example below.

### 1. Installation

```bash
git clone https://github.com/antoinegg1/EGM.git
cd verl
pip install -e .[vllm]
```

### 2. Model 

| Training Phase | Model | HuggingFace |
| :--- | :--- | :--- |
| Supervised Fine-Tuning (SFT) | EGM-Qwen3-VL-8B-SFT | [Link] |
| Reinforcement Learning | EGM-Qwen3-VL-8B-v1 | [Link] |

### 3. Data Preparation

Please download the training and testing datasets from [Link] before proceeding.

```bash
# Set your environment variables
export BASE_IMAGE_PATH=${YOUR_BASE_IMG_PATH}
export OUTPUT_DIR_PATH=${YOUR_DATA_DIR}
export VAL_DIR_PATH=${VAL_DIR_PATH}
export TRAIN_JSON=${QWEN3_8B_GROUNDING_TRAIN_JSON}

# Run preprocessing scripts
bash examples/data_preprocess/grounding_val.sh
bash examples/data_preprocess/grounding_all.sh
```

### 4. Training

Reinforcement Learning  is conducted based on the SFT checkpoint. The default configuration utilizes 8 GPUs. You may customize the distributed training settings via the `trainer.nnodes` and `trainer.n_gpus_per_node` arguments.

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

### 5. Evaluation

To evaluate the model, use the command provided below.

**Note:** The RefCOCO benchmark consists of eight distinct JSON files. Consequently, you must run the evaluation script sequentially for each of the 8 files to obtain the complete benchmark results.

```bash
export MODEL_PATH=${YOUR_MODEL_PATH}
export DATA_JSON=${DATA_JSON}
export OUTPUT_DIR=${YOUR_OUTPUT_DIR}

bash scripts/vllm_infer.sh
```

---

