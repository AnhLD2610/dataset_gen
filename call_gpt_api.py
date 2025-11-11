import json
from openai import OpenAI

client = OpenAI(api_key='YOUR_API_KEY_HERE')
MODEL = 'gpt-4.1-2025-04-14'
with open('filtered_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

def process_sample(sample, index):
    prompt = sample['prompt']
    original_output = sample['generate_output']
    question_idx = sample['question_idx']
    
    print(f"\n{'='*80}")
    print(f"Processing sample {index + 1}, question_idx: {question_idx}")
    print(f"Original tokens: {sample['num_tokens']}")
    
    try:
        response = client.chat.completions.create(
            model=MODEL,  # or "gpt-4", "gpt-3.5-turbo"
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=16384
        )
        
        gpt_output = response.choices[0].message.content
        
        print(f"\nGPT Response length: {len(gpt_output)} chars")
        print(f"Usage: {response.usage}")
        
        return {
            'question_idx': question_idx,
            'prompt': prompt,
            'original_output': original_output,
            'gpt_output': gpt_output,
            'num_tokens': sample['num_tokens'],
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None


# Example: Process first 3 samples
results = []
# for i in range(min(3, len(data))):
#     result = process_sample(data[i], i)
#     if result:
#         results.append(result)

# Save results
if results:
    output_file = 'gpt_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\nResults saved to {output_file}")

