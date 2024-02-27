#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1.5-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

 deepspeed --master_port=$((RANDOM + 10000)) --include localhost:0,3 geochat/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path /share/data/drive_3/kartik/llava-v1.5-7b \
    --version $PROMPT_VERSION \
    --data_path /share/data/drive_3/kartik/HRVQA_jsons/grounding_refering_regional.json \
    --image_folder /share/softwares/kartik/final_images_llava  \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter /share/data/drive_3/kartik/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir /share/data/drive_3/kartik/GeoChat/checkpoints/supervised_finetuning/random \
    --num_train_epochs 1 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 2 \
    --save_total_limit 9 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 16 \
    --report_to wandb
