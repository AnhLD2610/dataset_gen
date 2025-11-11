import json
import re
# Đọc file JSON
with open("/home/anhld48/Working/entropy_inference_data/gpt_data_final/filtered_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

i = 0
count = 0

def extract_reasoning(generate_output):
    """Extract reasoning between <think> and </think> tags"""
    # Find content between <think> and </think>
    match = re.search(r'<think>(.*?)</think>', generate_output, re.DOTALL)
    if match:
        return '<think>' + match.group(1) + '</think>'
    return None

for item in data:
    i += 1
    count += item['num_tokens']
    print(extract_reasoning(item['generate_output']))
    break 
print(count/i)
print(len(data))
print(data[0].keys())


# for item in data:
#     if item['evaluation'] != 'CORRECT':
#         print(item['reasoning'][:2000])
#         break 
# # Nếu file là list, lấy phần tử đầu tiên
# if isinstance(data, list) and len(data) > 0:
#     first_example = data[1]
# elif isinstance(data, dict):
#     first_example = data
# else:
#     first_example = None

# # In ra các trường (keys)
# if first_example:
#     print("🔑 Các trường (keys):")
#     print(list(first_example.keys()))
#     print("\n🧩 First example:")
#     # print(first_example['full_generated_text'])
#     print(json.dumps(first_example, indent=2, ensure_ascii=False))
# else:
#     print("⚠️ Không tìm thấy example hợp lệ.")

# print(len(data))