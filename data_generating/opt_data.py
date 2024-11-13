import json

input_file = 'generated_questions_unique.jsonl'  # Replace with your actual input file path
output_file = 'filtered_question_not_similar.jsonl'

# Read and filter the data from the jsonl file
filtered_data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        filtered_data.append({"class": item["class"], "question": item["question"]})

# Write the filtered data to a new jsonl file
with open(output_file, 'w', encoding='utf-8') as f:
    for item in filtered_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Filtered data saved to '{output_file}'")
