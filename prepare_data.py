from datasets import load_dataset
import json


dataset = load_dataset("open-r1/Mixture-of-Thoughts", "math", split="train")

data = dataset

print(f"Total samples: {len(data)}")

filtered_data = []
question_idx = 0

for item in data:
    if item['num_tokens'] <= 16000:
        messages = item['messages']
        
        prompt = None
        generate_output = None
        
        for msg in messages:
            if msg['role'] == 'user':
                prompt = msg['content']
            elif msg['role'] == 'assistant':
                generate_output = msg['content']
        
        # Only add if both prompt and output exist
        if prompt and generate_output:
            filtered_data.append({
                'question_idx': question_idx,
                'prompt': prompt,
                'generate_output': generate_output,
                'num_tokens': item['num_tokens'],
            })
            question_idx += 1

print(f"Filtered samples (num_tokens <= 16000): {len(filtered_data)}")

# Save to JSON file
output_file = 'filtered_dataset.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"Data saved to {output_file}")

# Print sample
if filtered_data:
    print("\nSample entry:")
    print(f"Prompt length: {len(filtered_data[0]['prompt'])}")
    print(f"Output length: {len(filtered_data[0]['generate_output'])}")
    print(f"Num tokens: {filtered_data[0]['num_tokens']}")
