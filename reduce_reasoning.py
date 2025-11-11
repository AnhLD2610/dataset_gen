import json
import re
import time
from openai import OpenAI
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Initialize OpenAI client
api_key = ""
client = OpenAI(api_key=api_key)

# Configuration
INPUT_FILE = 'filtered_dataset.json'
OUTPUT_FILE = 'reduced_reasoning_results.json'
CHECKPOINT_FILE = 'reduce_checkpoint.json'

# Parameters
MAX_SAMPLES = None  # Set to None to process all, or a number to limit
MODEL = 'gpt-4.1-2025-04-14'
TEMPERATURE = 0.6
MAX_TOKENS = 16384
NUM_WORKERS = 4  # Number of parallel threads
MAX_RETRIES = 3  # Number of retries on failure
RETRY_DELAY = 2  # seconds between retries

# Thread-safe lock for checkpoint updates
checkpoint_lock = threading.Lock()

# Task prompt template
TASK_PROMPT = '''You are a reasoning cutoff marker.

## Task:
For the given problem and reasoning, identify the earliest point where the reasoning approach is correct and sufficient to reach the answer. 

- Stop reasoning at that point by inserting </think>.  
- Do not continue to fully solve or try to compute the answer unless it is necessary to know the approach.  
- Copy exactly all text from <think> up to that point.  
- The output should be a verbatim substring of the reasoning up to the cutoff starting from <think>.  

### Rules:
- Do NOT rewrite, rephrase, summarize, or restate anything.
- Do NOT add or modify any text.
- The output must be a verbatim substring of the raw reasoning.
- Your only action is to decide **where** to place </think>.
- If your output contains any new text not in the raw reasoning, it is invalid.

### Output format (MUST match exactly):
<reduce_think>
[verbatim substring of the raw reasoning from <think> to your chosen </think>]
</reduce_think>

## Input:
{problem}
## Reasoning
{reasoning}
'''

def format_output_with_think_tags(text):
    """Add <think> and </think> tags if missing"""
    if not text:
        return text
    
    text = text.strip()
    
    # Check if already has both tags
    has_opening = '<think>' in text
    has_closing = '</think>' in text
    
    if has_opening and has_closing:
        return text
    
    # Add missing tags
    if not has_opening and not has_closing:
        # No tags at all, wrap entire text
        return f"<think>\n{text}\n</think>"
    elif not has_opening:
        # Has closing but no opening
        return f"<think>\n{text}"
    else:
        # Has opening but no closing
        return f"{text}\n</think>"


def extract_reasoning(generate_output):
    """Extract reasoning between <think> and </think> tags"""
    # Find content between <think> and </think>
    match = re.search(r'<think>(.*?)</think>', generate_output, re.DOTALL)
    if match:
        return '<think>' + match.group(1) + '</think>'
    return None

def load_checkpoint():
    """Load checkpoint to resume processing"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'processed_indices': [], 'results': []}

def save_checkpoint(checkpoint):
    """Save checkpoint"""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False)

def process_sample_with_retry(sample, index):
    """Process a single sample with GPT API with retry mechanism"""
    question_idx = sample['question_idx']
    problem = sample['prompt']
    generate_output = sample['generate_output']
    
    # Extract reasoning
    reasoning = extract_reasoning(generate_output)
    if not reasoning:
        return {
            'question_idx': question_idx,
            'index': index,
            'error': 'No reasoning tags found',
            'num_tokens': sample['num_tokens']
        }
    
    # Create prompt
    prompt = TASK_PROMPT.format(problem=problem, reasoning=reasoning)
    
    # Retry loop
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            
            gpt_output = response.choices[0].message.content
            
            # Format output with think tags if missing
            formatted_output = format_output_with_think_tags(gpt_output)
            
            return {
                'question_idx': question_idx,
                'index': index,
                'problem': problem,
                'original_reasoning': reasoning,
                'reduced_reasoning': formatted_output,
                'num_tokens': sample['num_tokens'],
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return {
                    'question_idx': question_idx,
                    'index': index,
                    'error': f'Failed after {MAX_RETRIES} retries: {str(e)}',
                    'num_tokens': sample['num_tokens']
                }
    
    return {
        'question_idx': question_idx,
        'index': index,
        'error': 'Unexpected error in retry loop',
        'num_tokens': sample['num_tokens']
    }

def main():
    # Load data
    print("="*80)
    print("MULTITHREADED GPT PROCESSING WITH AUTO-FORMAT & RETRY")
    print("="*80)
    print(f"Worker threads: {NUM_WORKERS}")
    print(f"Max retries: {MAX_RETRIES}")
    print(f"Model: {MODEL}")
    print("="*80)
    
    print("\nLoading dataset...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_samples = len(data) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(data))
    print(f"Total samples: {total_samples}")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_indices = set(checkpoint['processed_indices'])
    results = checkpoint['results']
    
    print(f"Already processed: {len(processed_indices)}")
    
    # Get samples that need processing
    samples_to_process = [(data[i], i) for i in range(total_samples) if i not in processed_indices]
    
    if not samples_to_process:
        print("All samples already processed!")
    else:
        print(f"\nProcessing {len(samples_to_process)} samples with {NUM_WORKERS} workers...")
        
        # Process with multithreading
        try:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(process_sample_with_retry, sample, idx): idx
                    for sample, idx in samples_to_process
                }
                
                # Process results as they complete
                with tqdm(total=len(samples_to_process), desc="Processing") as pbar:
                    for future in as_completed(future_to_index):
                        idx = future_to_index[future]
                        result = future.result()
                        
                        # Thread-safe checkpoint update
                        with checkpoint_lock:
                            results.append(result)
                            processed_indices.add(idx)
                            
                            # Save checkpoint every 10 samples
                            if len(results) % 10 == 0:
                                checkpoint = {
                                    'processed_indices': list(processed_indices),
                                    'results': results
                                }
                                save_checkpoint(checkpoint)
                        
                        pbar.update(1)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Saving checkpoint...")
            checkpoint = {
                'processed_indices': list(processed_indices),
                'results': results
            }
            save_checkpoint(checkpoint)
            print("Checkpoint saved.")
            return
    
    # Save final results
    print(f"\n\nSaving {len(results)} results...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {OUTPUT_FILE}")
    
    # Calculate statistics
    successful = sum(1 for r in results if 'error' not in r)
    failed = len(results) - successful
    total_tokens = sum(r.get('usage', {}).get('total_tokens', 0) for r in results if 'usage' in r)
    
    print(f"\nStatistics:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total tokens: {total_tokens:,}")
    
    # Clean up checkpoint file
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

if __name__ == "__main__":
    main()

