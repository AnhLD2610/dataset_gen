
import os
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"

import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
from tqdm import tqdm
import shutil
from dataclasses import dataclass, asdict

# Constants
ERROR_MARKER = "❌_GENERATION_FAILED_❌"
SUCCESS_MARKER = "✅_GENERATION_SUCCESS_✅"


@dataclass
class GenerationResult:
    """Structured result object - simplified for inference only"""
    question_idx: int
    sample_idx: int
    prompt: str
    generate_output: str
    num_tokens: int
    deepseek_generated_output: str
    generation_status: str
    error_message: Optional[str]
    
    def to_dict(self) -> Dict:
        """Convert to dict"""
        return asdict(self)


# ============================================================================
# DATA PREPARATION
# ============================================================================

def format_prompt_with_chat_template(tokenizer, prompt: str) -> str:
    """Format prompt with chat template and instruction tail"""
    tail = r" Please reason step by step, and put your final answer within \boxed{}."
    messages = [
        {"role": "user", "content": prompt + tail}
    ]
    
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text
    except Exception as e:
        print(f"⚠️  Chat template failed: {e}, using fallback")
        return f"User: {prompt + tail}\n\nAssistant:"


def prepare_samples(data: List[Dict], tokenizer, num_samples: int) -> List[Dict]:
    """Prepare all samples for generation - keep original fields + format prompt"""
    samples = []
    
    for item in data:
        prompt = item.get('prompt', '')
        if not prompt:
            continue
        
        question_idx = item.get('question_idx', 0)
        generate_output = item.get('generate_output', '')
        num_tokens = item.get('num_tokens', 0)
        
        # Format prompt with chat template
        formatted_prompt = format_prompt_with_chat_template(tokenizer, prompt)
        
        for s_idx in range(num_samples):
            samples.append({
                'question_idx': question_idx,
                'sample_idx': s_idx,
                'prompt': prompt,  # Original prompt
                'formatted_prompt': formatted_prompt,  # Formatted for generation
                'generate_output': generate_output,
                'num_tokens': num_tokens,
            })
    
    return samples


# ============================================================================
# RESULT CREATION
# ============================================================================

def create_error_result(sample: Dict, error_msg: str) -> Dict:
    """Create error result dictionary"""
    result = GenerationResult(
        question_idx=sample['question_idx'],
        sample_idx=sample['sample_idx'],
        prompt=sample.get('prompt', ''),
        generate_output=sample.get('generate_output', ''),
        num_tokens=sample.get('num_tokens', 0),
        deepseek_generated_output=f"{ERROR_MARKER}: {error_msg}",
        generation_status=ERROR_MARKER,
        error_message=error_msg,
    )
    
    return result.to_dict()


def create_success_result(sample: Dict, deepseek_output: str) -> Dict:
    """Create success result with DeepSeek generated text"""
    
    result = GenerationResult(
        question_idx=sample['question_idx'],
        sample_idx=sample['sample_idx'],
        prompt=sample['prompt'],
        generate_output=sample['generate_output'],
        num_tokens=sample['num_tokens'],
        deepseek_generated_output=deepseek_output,
        generation_status=SUCCESS_MARKER,
        error_message=None,
    )
    
    return result.to_dict()


# ============================================================================
# GENERATION PROCESSING
# ============================================================================

def process_single_sample(sample: Dict, llm: LLM, sampling_params: SamplingParams) -> Dict:
    """Process single sample - simple inference with formatted prompt"""
    try:
        output = llm.generate([sample['formatted_prompt']], sampling_params)[0]
        generated_text = output.outputs[0].text
        
        if not generated_text:
            return create_error_result(sample, "Empty generation")
        
        return create_success_result(sample, generated_text)
    except Exception as e:
        return create_error_result(sample, f"Generation failed: {str(e)}")


def process_batch(batch_samples: List[Dict], llm: LLM, sampling_params: SamplingParams) -> List[Dict]:
    """Process batch with fallback to individual processing"""
    
    try:
        # Use formatted prompts for generation
        prompts = [s['formatted_prompt'] for s in batch_samples]
        outputs = llm.generate(prompts, sampling_params)
        
        results = []
        for sample, output in zip(batch_samples, outputs):
            generated_text = output.outputs[0].text
            
            if not generated_text:
                results.append(create_error_result(sample, "Empty generation"))
            else:
                results.append(create_success_result(sample, generated_text))
        
        return results
    
    except Exception as e:
        print(f"⚠️  Batch failed: {e}, processing individually...")
        
        results = []
        for sample in batch_samples:
            try:
                result = process_single_sample(sample, llm, sampling_params)
                results.append(result)
            except Exception as single_e:
                results.append(create_error_result(sample, f"Failed: {single_e}"))
        
        return results


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def load_checkpoints(checkpoint_dir: str) -> Tuple[int, List[Dict]]:
    """Load existing checkpoints and return (start_batch, results)"""
    if not os.path.exists(checkpoint_dir):
        return 0, []
    
    checkpoint_files = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_") and f.endswith(".parquet")
    ])
    
    if not checkpoint_files:
        return 0, []
    
    print(f"🔄 Loading {len(checkpoint_files)} checkpoints...")
    
    all_results = []
    max_batch = 0
    
    for ckpt_file in checkpoint_files:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        df = pd.read_parquet(ckpt_path)
        all_results.extend(df.to_dict('records'))
        
        # Extract batch number
        batch_num = int(ckpt_file.split("_")[1].replace(".parquet", ""))
        max_batch = max(max_batch, batch_num)
    
    return max_batch + 1, all_results


def save_checkpoint(results: List[Dict], checkpoint_dir: str, batch_idx: int):
    """Save checkpoint for given batch"""
    if not results:
        return
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{batch_idx}.parquet")
    df = pd.DataFrame(results)
    df.to_parquet(checkpoint_path, index=False)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # Configuration
    CONFIG = {
        'INPUT_JSON': "filtered_dataset.json",
        'OUTPUT_DIR': "gen_data_deepseek_final",
        'MODEL_NAME': "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        'NUM_SAMPLES': 1,
        'TENSOR_PARALLEL': 4,
        'BATCH_SIZE': 16,
        'GPU_MEMORY': 0.95,
        'MAX_QUESTIONS': None,
        'CHECKPOINT_EVERY': 10000,
    }
    
    # Setup paths
    OUTPUT_DIR = CONFIG['OUTPUT_DIR']
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "final_output.parquet")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print("="*80)
    print("SIMPLE INFERENCE - NO ENTROPY CALCULATION")
    print("="*80)
    print(f"Tensor Parallel: {CONFIG['TENSOR_PARALLEL']} GPUs")
    print(f"Batch size: {CONFIG['BATCH_SIZE']}")
    print(f"Checkpoint every: {CONFIG['CHECKPOINT_EVERY']} batches")
    print("="*80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
    print("Tokenizer loaded")
    
    # Load data
    print("\nLoading data...")
    with open(CONFIG['INPUT_JSON']) as f:
        data = json.load(f)
    
    if CONFIG['MAX_QUESTIONS']:
        data = data[:CONFIG['MAX_QUESTIONS']]
    
    print(f"Loaded {len(data)} questions")
    
    # Prepare samples
    print("Preparing samples...")
    samples = prepare_samples(data, tokenizer, CONFIG['NUM_SAMPLES'])
    print(f"Total: {len(samples)} samples")
    
    if not samples:
        print("No samples to process!")
        return
    
    # Load existing checkpoints
    start_batch, all_results = load_checkpoints(CHECKPOINT_DIR)
    print(f"Already processed: {len(all_results)}, resuming from batch {start_batch}")
    
    # Initialize vLLM
    print("\nInitializing vLLM...")
    llm = LLM(
        model=CONFIG['MODEL_NAME'],
        max_model_len=20000,
        gpu_memory_utilization=CONFIG['GPU_MEMORY'],
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=8192,
        tensor_parallel_size=CONFIG['TENSOR_PARALLEL'],
    )
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16000,
    )
    
    # Process batches
    print("\nProcessing batches...")
    start_time = time.time()
    
    num_batches = (len(samples) + CONFIG['BATCH_SIZE'] - 1) // CONFIG['BATCH_SIZE']
    checkpoint_buffer = []
    
    for batch_idx in tqdm(range(0, num_batches), initial=start_batch, ncols=100):
        if batch_idx < start_batch:
            continue
        
        batch_start = batch_idx * CONFIG['BATCH_SIZE']
        batch_end = min(batch_start + CONFIG['BATCH_SIZE'], len(samples))
        batch = samples[batch_start:batch_end]
        
        batch_results = process_batch(batch, llm, sampling_params)
        all_results.extend(batch_results)
        checkpoint_buffer.extend(batch_results)
        
        # Save checkpoint
        if (batch_idx + 1) % CONFIG['CHECKPOINT_EVERY'] == 0:
            save_checkpoint(checkpoint_buffer, CHECKPOINT_DIR, batch_idx + 1)
            checkpoint_buffer = []
    
    # Final checkpoint
    if checkpoint_buffer:
        save_checkpoint(checkpoint_buffer, CHECKPOINT_DIR, num_batches)
    
    generation_time = time.time() - start_time
    
    # Save final results
    print("\nSaving results...")
    df = pd.DataFrame(all_results)
    df = df.sort_values(['question_idx', 'sample_idx']).reset_index(drop=True)
    df.to_parquet(FINAL_OUTPUT, index=False)
    
    OUTPUT_JSON = FINAL_OUTPUT.replace('.parquet', '.json')
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(df.to_dict('records'), f, indent=2, ensure_ascii=False)
    
    # Cleanup
    print("\nCleaning up checkpoints...")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    
    # Report
    total = len(all_results)
    success = sum(1 for r in all_results if r['generation_status'] == SUCCESS_MARKER)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Total: {total} | Success: {success} ({success/total*100:.1f}%)")
    print(f"Time: {generation_time/60:.1f}min | Throughput: {total/generation_time:.2f} samples/sec")
    print(f"\nOutput: {FINAL_OUTPUT}")
    print("="*80)


if __name__ == "__main__":
    main()