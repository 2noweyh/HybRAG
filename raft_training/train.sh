#!/bin/bash
# bash raft_training/train.sh 2 Qwen2.5-14B webqsp v1
# bash raft_training/train.sh 2 Meta-Llama-3-8B webqsp vfinal both

# bash raft_training/train.sh 2 Llama-2-7B webqsp vfinal both
# bash raft_training/train.sh 2 Llama-2-7B webqsp vfinal dense

# bash raft_training/train.sh 2 Llama-2-7B-chat webqsp vfinal both
# bash raft_training/train.sh 2 Llama-2-7B-chat webqsp vfinal dense

# bash raft_training/train.sh 2 Meta-Llama-3-8B webqsp vfinal both
# bash raft_training/train.sh 2 Meta-Llama-3-8B webqsp vfinal dense

# bash raft_training/train.sh 2 Meta-Llama-3-8B-Instruct webqsp vfinal both
# bash raft_training/train.sh 2 Meta-Llama-3-8B-Instruct webqsp vfinal dense

num_epochs=$1
model=$2

dataset=$3
version=$4
mode=$5

num_val=1
output_dir="./models/${dataset}/${model}_${version}_${mode}"
trn_data_path="/home/noweyh/hw_work/RAG/hybRAG/data/${dataset}/rerank_train_${version}_${mode}.json"
dev_data_path="/home/noweyh/hw_work/RAG/hybRAG/data/${dataset}/rerank_eval_${version}_${mode}.json"
# dev_data_path="/home/noweyh/hw_work/RAG/hybRAG/data/${dataset}/rerank_eval_${version}_dense.json"
# trn_data_path="/home/noweyh/hw_work/RAG/hybRAG/data/${dataset}/rerank_train_${version}.json"
# dev_data_path="/home/noweyh/hw_work/RAG/hybRAG/data/${dataset}/rerank_eval_${version}.json"

if [ "$model" = "Flan-T5-XXL" ]; then
    base_model="google/flan-t5-xxl"
    lora_target_modules='[q, k, v, o, wi_0, wi_1, wo, lm_head]'
    prompt_template_name="alpaca"
elif [ "$model" = "Flan-T5-XL" ]; then
    base_model="google/flan-t5-xl"
    lora_target_modules='[q, k, v, o, wi_0, wi_1, wo, lm_head]'
    prompt_template_name="alpaca"
elif [ "$model" = "Phi-2" ]; then
    base_model="microsoft/phi-2"
    lora_target_modules='[Wqkv, out_proj, fc1, fc2, linear]'
    prompt_template_name="alpaca"
elif [ "$model" = "Mistral-7B-Instruct-v0.2" ]; then #이것만 나중에 Prompt 만들기 
    base_model="mistralai/Mistral-7B-Instruct-v0.2"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="mistral"
elif [ "$model" = "Qwen2.5-14B" ]; then # 실행예정
    base_model="Qwen/Qwen2.5-14B-Instruct"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="llama"
elif [ "$model" = "Qwen3-14B-Instruct" ]; then # 실행예정
    base_model="Qwen/Qwen3-14B-Instruct"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="alpaca"
elif [ "$model" = "Qwen1.5-7B" ]; then
    base_model="Qwen/Qwen1.5-7B"
    lora_target_modules='[c_attn, c_proj, w1, w2, lm_head]'  # 예시, 실제 확인 필요
    prompt_template_name="qwen"
elif [ "$model" = "Llama-2-13B-chat" ]; then 
    base_model="meta-llama/Llama-2-13b-chat-hf"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="llama"
elif [ "$model" = "Llama-2-13B" ]; then
    base_model="meta-llama/Llama-2-13b-hf"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="llama"
elif [ "$model" = "Llama-2-7B-chat" ]; then 
    base_model="meta-llama/Llama-2-7b-chat-hf"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="llama"
elif [ "$model" = "Llama-2-7B" ]; then
    base_model="meta-llama/Llama-2-7b-hf" 
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="llama"
elif [ "$model" = "Meta-Llama-3-8B" ]; then #현재 가장 우수
    base_model="meta-llama/Meta-Llama-3-8B"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="llama"
elif [ "$model" = "Meta-Llama-3-8B-Instruct" ]; then 
    base_model="meta-llama/Meta-Llama-3-8B-Instruct"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="llama"  # or "llama3"
else
    base_model=""
    lora_target_modules=""
    prompt_template_name=""
fi

    # --num_train_epochs 3 \
    # --per_device_train_batch_size 4 \
    # --per_device_eval_batch_size 4 \
    # --gradient_accumulation_steps 16 \
    # --learning_rate 2e-5 \
#  --warmup_ratio 0.03 \
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_LAUNCH_BLOCKING=1 
CUDA_VISIBLE_DEVICES=0 python raft_training/train.py \
    --base_model $base_model \
    --trn_data_path $trn_data_path \
    --dev_data_path $dev_data_path \
    --output_dir $output_dir \
    --batch_size 4 \
    --micro_batch_size 2 \
    --num_epochs $num_epochs \
    --cutoff_len 1080 \
    --val_set_size $num_val \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules "$lora_target_modules" \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name $prompt_template_name \
    --lr_scheduler 'cosine' \
    --optim "adamw_torch" \
    --warmup_ratio 0.03 \