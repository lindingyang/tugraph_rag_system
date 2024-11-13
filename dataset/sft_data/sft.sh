# 单机多卡
CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=3 \
swift sft \
    --custom_register_path "/root/autodl-tmp/sft+rag/dataset/custom.py" \
    --model_type qwen-14b-instruct \
    --model_id_or_path "/root/autodl-tmp/qwen2_5-14b-instruct" \
    --dataset "/root/autodl-tmp/sft+rag/dataset/sft_data/api_glm4_flash_sft_data.jsonl" \
    --logging_steps 10 \
    --learning_rate 1e-5 \
    --output_dir '/root/autodl-tmp/output' \
    --num_train_epochs 1 \
    --batch_size 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --lora_target_modules ALL \
    --deepspeed default-zero3