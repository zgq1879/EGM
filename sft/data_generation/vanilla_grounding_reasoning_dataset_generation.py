

import json
import os
import re
import argparse
import random
from typing import List, Dict, Tuple
from openai import AzureOpenAI
import base64
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_jsonl_data(jsonl_path: str) -> List[Dict]:
    
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_question_and_bbox(sample: Dict) -> Tuple[str, List[int], str, int, int]:
    
    conversations = sample['conversations']
    
    human_msg = conversations[0]['value']
    ref_pattern = r'<ref>(.*?)</ref>'
    match = re.search(ref_pattern, human_msg)
    question = match.group(1) if match else ""
    
    gpt_msg = conversations[1]['value']
    bbox_pattern = r'<box>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</box>'
    bbox_match = re.search(bbox_pattern, gpt_msg)
    
    if bbox_match:
        gt_bbox = [int(bbox_match.group(i)) for i in range(1, 5)]
    else:
        gt_bbox = [0, 0, 0, 0]
    
    image_path = sample['image']
    height = sample['height']
    width = sample['width']
    
    return question, gt_bbox, image_path, height, width

def denormalize_bbox(bbox: List[int], height: int, width: int) -> List[float]:
    
    x1, y1, x2, y2 = bbox
    
    x1_norm = x1 / 1000.0
    y1_norm = y1 / 1000.0
    x2_norm = x2 / 1000.0
    y2_norm = y2 / 1000.0
    
    return [
        x1_norm * width,
        y1_norm * height,
        x2_norm * width,
        y2_norm * height
    ]

def create_annotated_image(image_path: str, gt_bbox: List[float], output_path: str, base_image_dir: str = None) -> str:
    
    try:
        if base_image_dir:
            full_image_path = os.path.join(base_image_dir, image_path)
        else:
            full_image_path = image_path
        
        if not os.path.exists(full_image_path):
            print(f"Warning: Image not found: {full_image_path}")
            return None
        
        img = Image.open(full_image_path)
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]], 
                      outline='green', width=5)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        draw.text((gt_bbox[0], max(0, gt_bbox[1]-30)), 'Ground Truth', fill='green', font=font)
        
        img.save(output_path)
        return output_path
        
    except Exception as e:
        print(f"Failed to create annotated image: {e}")
        return None

def encode_image_to_base64(image_path: str) -> str:
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_with_gpt(client: AzureOpenAI, question: str, image_base64: str, gt_bbox: List[int]) -> str:
    
    
    prompt = f"""
You are looking at an image with a visual grounding task. The image shows a green bounding box indicating the Ground Truth (GT) target object.

Question: "{question}"
Ground Truth bbox (normalized 0-999 coordinates): {gt_bbox}, which is highlighed as a green bbox on the input image.

Please analyze this visual grounding case and explain how to correctly output the Ground Truth bbox according to the text description of the Question, especially when there are multiple similar objects.

Please provide a clear and concise analysis focusing on the visual reasoning process in 1-2 sentences. Don't say 'green bbox' in your answer. The answer you give is going to be used to train a small model, so you should pretend not to know the GT bbox before hand. Just answer the question with reasoning.
"""

    try:
        messages = [
            {"role": "system", "content": "You are an expert in computer vision and visual grounding tasks. Analyze images carefully and provide clear explanations about object localization."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4.1-20250414",
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"GPT analysis error: {e}")
        return f"Analysis failed: {str(e)}"

def create_visualization(image_path: str, gt_bbox: List[float], question: str, gpt_analysis: str, output_path: str):
    
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        fig = plt.figure(figsize=(14, 12))
        
        ax_img = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        
        ax_img.imshow(img)
        
        gt_rect = patches.Rectangle(
            (gt_bbox[0], gt_bbox[1]), 
            gt_bbox[2] - gt_bbox[0], 
            gt_bbox[3] - gt_bbox[1],
            linewidth=4, edgecolor='green', facecolor='none', label='Ground Truth'
        )
        ax_img.add_patch(gt_rect)
        
        ax_img.set_title(f'Question: "{question}"', fontsize=14, pad=15, wrap=True)
        ax_img.legend(loc='upper right', fontsize=11)
        ax_img.axis('off')
        
        ax_text = plt.subplot2grid((3, 1), (2, 0))
        ax_text.axis('off')
        
        text_info = f"GPT-4 Analysis:\n\n{gpt_analysis}"
        
        ax_text.text(0.05, 0.95, text_info, fontsize=10, wrap=True, 
                    verticalalignment='top', horizontalalignment='left',
                    transform=ax_text.transAxes,
                    bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    except Exception as e:
        print(f"Failed to create visualization: {e}")
        return None

def process_single_sample(sample_data: Tuple) -> Dict:
    
    idx, sample, args = sample_data
    
    client = AzureOpenAI(
        azure_endpoint=args.azure_endpoint,
        api_version="2025-02-01-preview",
        api_key=args.api_key
    )
    
    question, gt_bbox_norm, image_path, height, width = extract_question_and_bbox(sample)
    
    gt_bbox_abs = denormalize_bbox(gt_bbox_norm, height, width)
    
    if args.base_image_dir:
        full_image_path = os.path.join(args.base_image_dir, image_path)
    else:
        full_image_path = image_path
    
    if not os.path.exists(full_image_path):
        print(f"Warning: Image not found: {full_image_path}")
        return None
    
    temp_img_path = os.path.join(args.temp_img_dir, f"temp_{idx:06d}.jpg")
    annotated_path = create_annotated_image(image_path, gt_bbox_abs, temp_img_path, args.base_image_dir)
    
    if annotated_path is None:
        return None
    
    image_base64 = encode_image_to_base64(annotated_path)
    
    gpt_analysis = analyze_with_gpt(client, question, image_base64, gt_bbox_norm)
    
    if os.path.exists(annotated_path):
        os.remove(annotated_path)
    
    gpt_answer_path = os.path.join(args.gpt_answer_dir, f"analysis_{idx:06d}.txt")
    with open(gpt_answer_path, 'w', encoding='utf-8') as f:
        f.write(f"Question: {question}\n")
        f.write(f"GT BBox (normalized): {gt_bbox_norm}\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"GPT-4 Analysis:\n")
        f.write(f"{'='*80}\n\n")
        f.write(gpt_analysis)
    
    result_data = {
        'index': idx,
        'question': question,
        'image_path': image_path,
        'gt_bbox_normalized': gt_bbox_norm,
        'gt_bbox_absolute': [int(x) for x in gt_bbox_abs],
        'image_height': height,
        'image_width': width,
        'gpt_analysis': gpt_analysis
    }
    
    json_path = os.path.join(args.gpt_answer_dir, f"analysis_{idx:06d}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    return {
        'idx': idx,
        'full_image_path': full_image_path,
        'gt_bbox_abs': gt_bbox_abs,
        'gpt_analysis': gpt_analysis,
        'question': question
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze RefCOCO grounding questions with GPT')
    parser.add_argument('--input_jsonl', 
                       default='xxx',
                       help='Input JSONL file path')
    parser.add_argument('--base_image_dir',
                       default='xxx',
                       help='Base directory for images')
    parser.add_argument('--output_dir', default='./vis', help='Output directory for visualizations')
    parser.add_argument('--gpt_answer_dir', default='./gpt_answers', help='Directory to save GPT answers')
    parser.add_argument('--temp_img_dir', default='./temp_img', help='Directory for temporary annotated images')
    parser.add_argument('--api_key', 
                       default='xxx',
                       help='OpenAI API key')
    parser.add_argument('--azure_endpoint',
                       default='https://llm-proxy.perflab.nvidia.com',
                       help='Azure OpenAI endpoint URL')
    parser.add_argument('--num_samples', type=int, default=0, help='Number of samples to process (0 for all, ignored if start_index is set)')
    parser.add_argument('--start_index', type=int, default=None, help='Start index for processing samples (0-based, inclusive)')
    parser.add_argument('--end_index', type=int, default=None, help='End index for processing samples (0-based, exclusive)')
    parser.add_argument('--num_threads', type=int, default=64, help='Number of parallel threads')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--skip_visualization', type=bool, default=True, help='Skip creating visualization images (only save GPT answers)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.gpt_answer_dir, exist_ok=True)
    os.makedirs(args.temp_img_dir, exist_ok=True)
    
    print(f"Loading data from {args.input_jsonl}...")
    all_data = load_jsonl_data(args.input_jsonl)
    print(f"Loaded {len(all_data)} samples")
    
    if args.start_index is not None or args.end_index is not None:
        start_idx = args.start_index if args.start_index is not None else 0
        end_idx = args.end_index if args.end_index is not None else len(all_data)
        
        start_idx = max(0, min(start_idx, len(all_data)))
        end_idx = max(start_idx, min(end_idx, len(all_data)))
        
        samples = list(enumerate(all_data[start_idx:end_idx], start=start_idx))
        print(f"Processing samples from index {start_idx} to {end_idx-1} ({len(samples)} samples)")
    elif args.num_samples > 0 and args.num_samples < len(all_data):
        random.seed(args.random_seed)
        samples = random.sample(list(enumerate(all_data)), args.num_samples)
        print(f"Randomly selected {args.num_samples} samples")
    else:
        samples = list(enumerate(all_data))
        print(f"Processing all {len(samples)} samples")
    
    thread_data = [(idx, sample, args) for idx, sample in samples]
    
    print(f"Processing samples with {args.num_threads} threads...")
    results = []
    failed_count = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        future_to_data = {executor.submit(process_single_sample, data): data for data in thread_data}
        
        for future in tqdm(as_completed(future_to_data), total=len(thread_data), desc="Processing"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    failed_count += 1
            except Exception as e:
                print(f"\nError processing sample: {e}")
                failed_count += 1
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Successfully processed: {len(results)}/{len(samples)}")
    print(f"❌ Failed: {failed_count}/{len(samples)}")
    print(f"⏱️  Total time: {elapsed_time:.1f}s")
    print(f"🚀 Average speed: {len(results)/elapsed_time:.2f} samples/second")
    
    if not args.skip_visualization:
        print("\nCreating final visualizations...")
        for result in tqdm(results, desc="Creating visualizations"):
            idx = result['idx']
            full_image_path = result['full_image_path']
            gt_bbox_abs = result['gt_bbox_abs']
            gpt_analysis = result['gpt_analysis']
            question = result['question']
            
            vis_output_path = os.path.join(args.output_dir, f"visualization_{idx:06d}.png")
            create_visualization(full_image_path, gt_bbox_abs, question, gpt_analysis, vis_output_path)
    else:
        print("\n⏩ Skipping visualization generation (--skip_visualization flag is set)")
    
    summary = {
        'input_file': args.input_jsonl,
        'total_samples_in_file': len(all_data),
        'start_index': args.start_index if args.start_index is not None else 0,
        'end_index': args.end_index if args.end_index is not None else len(all_data),
        'num_processed': len(results),
        'num_failed': failed_count,
        'output_dir': args.output_dir,
        'gpt_answer_dir': args.gpt_answer_dir,
        'processing_time_seconds': elapsed_time,
        'samples_per_second': len(results)/elapsed_time if elapsed_time > 0 else 0,
        'visualization_skipped': args.skip_visualization
    }
    
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Processing completed!")
    print(f"📁 Visualizations saved in: {args.output_dir}")
    print(f"📄 GPT answers saved in: {args.gpt_answer_dir}")
    print(f"📊 Summary file: {summary_path}")

if __name__ == "__main__":
    main()

