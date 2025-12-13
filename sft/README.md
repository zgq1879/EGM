## Bigger or Longer? Test-Time-Scaling is More Efficient than Model-Size-Scaling for Visual Grounding
## SFT - Qwen3-VL


### Data Generation

First, download the original RefCOCO train [annotations](https://huggingface.co/datasets/JamesZGQ/EGM_Datasets/blob/main/vanilla_grounding_reasoning_training_dataset_raw.jsonl) and [images](https://huggingface.co/datasets/JamesZGQ/EGM_Datasets/tree/main).

```bash
hf download JamesZGQ/EGM_Datasets --repo-type=dataset --local-dir ./EGM_Datasets
cd EGM_Datasets
cat coco.tar.part_* > coco.tar
tar -xvf coco.tar
tar -xvf coco_flip.tar
```


Then, generates reasoning for vanilla grounding tasks by analyzing how to locate Ground Truth bboxes. 

```bash
cd ..
python data_generation/vanilla_grounding_reasoning_dataset_generation.py --input_jsonl EGM_Datasets/vanilla_grounding_reasoning_training_dataset_raw.jsonl --base_image_dir EGM_Datasets --api_key <openai_api_key> --num_samples 0 --num_threads 64
```

After generating reasoning with GPT, convert the output to Qwen3-VL SFT training format:

```bash
python data_generation/convert_gpt_to_qwen3_sft.py \
    --gpt_answer_dir ./gpt_answers \
    --output_jsonl ./qwen3_sft_train.jsonl
```

**Arguments:**
- `--gpt_answer_dir`: Directory containing GPT analysis JSON files (from the previous step)
- `--output_jsonl`: Output JSONL file path for Qwen3-VL SFT training




### Requirements

You could use follow version of packages:

- `torch==2.6.0`
- `torchvision==0.21.0`
- `transformers==4.57.0.dev0`
- `deepspeed==0.17.1`
- `flash_attn==2.7.4.post1`
- `triton==3.2.0`
- `accelerate==1.7.0`
- `torchcodec==0.2`
- `peft==0.17.1`

### Installation

```
conda create -n qwen -y -c nvidia/label/cuda-12.4.0 -c nvidia -c conda-forge python=3.9 cuda-toolkit=12.4

conda activate qwen

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

pip install transformers==4.57.1 deepspeed==0.17.1 triton==3.2.0 accelerate==1.7.0 torchcodec==0.2 peft==0.17.1 importlib_metadata wandb
```

### SFT Training

Before training, set your environment variables:

```bash
# Set your environment variables
export REFCOCO_ANNOTATION_PATH=../qwen3_sft_train.jsonl
export REFCOCO_DATA_PATH=../EGM_Datasets
export OUTPUT_DIR=/path/to/output/directory

# Example:
# export REFCOCO_ANNOTATION_PATH=/data/grounding/refcoco_train.json
# export REFCOCO_DATA_PATH=/data/images/refcoco
# export OUTPUT_DIR=./output/qwen3_2b_grounding_sft
```

Then run the training scripts:

```bash
cd qwen-vl-finetune

# Train Qwen3-VL 2B
bash scripts/sft_qwen3_2b_grounding.sh

# Train Qwen3-VL 4B
bash scripts/sft_qwen3_4b_grounding.sh

# Train Qwen3-VL 8B
bash scripts/sft_qwen3_8b_grounding.sh
```

**Environment Variables:**

- `REFCOCO_ANNOTATION_PATH`: Path to your RefCOCO grounding annotation JSON file
- `REFCOCO_DATA_PATH`: Path to your image data folder
- `OUTPUT_DIR`: Directory where model checkpoints will be saved (optional, has defaults)
