from transformers import AutoModelForCausalLM

# Step 1: Load the base model and SFT model
base_model_path = '/root/autodl-tmp/qwen2_5-14b-instruct'  # Replace with your base model path
sft_model_path = '/root/autodl-tmp/output/qwen-14b-instruct/v5-20241108-223319/checkpoint-200'    # Replace with your SFT model path

base_model = AutoModelForCausalLM.from_pretrained(base_model_path).to('cuda:0')
sft_model = AutoModelForCausalLM.from_pretrained(sft_model_path).to('cuda:1')


# Step 2: Extract the SFT model's state dictionary
sft_state_dict = sft_model.state_dict()

# Step 3: Merge the SFT weights into the base model
base_state_dict = base_model.state_dict()

for key in sft_state_dict:
    if key in base_state_dict:
        base_state_dict[key] = sft_state_dict[key]

# Update the base model with the new state dictionary
base_model.load_state_dict(base_state_dict)

# Step 4: Save the merged model
merged_model_path = '/root/autodl-tmp/sft_qwen_2_5-14b'  # Replace with your desired save path
base_model.save_pretrained(merged_model_path)
print("Successfully merged!")