import json

# Base URL format
base_url = "https://raw.githubusercontent.com/theshi-1128/rag_dataset/refs/heads/master/updated_file_{}.md"

# Generate URLs from updated_file_4.md to updated_file_172.md
web_paths = [base_url.format(i) for i in range(1, 173)]

# Specify the output file path (change this to your desired file path)
output_path = "../dataset/web_paths_split_full_manual_refine_table2text_final.json"

# Write the list to a JSON file
with open(output_path, "w") as json_file:
    json.dump({"web_paths": web_paths}, json_file, indent=4)

print(f"URLs have been written to {output_path}")
