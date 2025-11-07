#!/bin/bash
# bash raft_inference/inference.sh Qwen2.5-14B webqsp v1 both
# bash raft_inference/inference.sh Llama-2-7B webqsp v7 both

# bash raft_inference/inference.sh Meta-Llama-3-8B webqsp vfinal both
# bash raft_inference/inference.sh Meta-Llama-3-8B webqsp vfinal dense

# bash raft_inference/inference.sh Meta-Llama-3-8B-Instruct webqsp v5
# bash raft_inference/inference.sh Meta-Llama-3-8B-Instruct webqsp vfinal both

model=$1
dataset=$2
version=$3
mode=$4

model_path="./models/${dataset}/${model}_${version}_${mode}"
# data_path="/home/noweyh/hw_work/RAG/hybRAG/data/${dataset}/rerank_test_${version}.json"
data_path="/home/noweyh/hw_work/RAG/hybRAG/data/${dataset}/rerank_test_${version}_${mode}.json"
# data_path="/home/noweyh/hw_work/RAG/hybRAG/data/${dataset}/rerank_test_${version}_dense.json"
output_path="./outputs/${dataset}/${model}_${version}_result_${mode}.jsonl"
# output_path="./outputs/${dataset}/${model}_${version}_result_both-dense.jsonl"

# Base model & template 지정
if [ "$model" = "Flan-T5-XXL" ]; then
    base_model="google/flan-t5-xxl"
    prompt_template_name="alpaca"
elif [ "$model" = "Flan-T5-XL" ]; then
    base_model="google/flan-t5-xl"
    prompt_template_name="alpaca"
elif [ "$model" = "Phi-2" ]; then
    base_model="microsoft/phi-2"
    prompt_template_name="alpaca"
elif [ "$model" = "Mistral-7B-Instruct-v0.2" ]; then #이것만 나중에 Prompt 만들기 
    base_model="mistralai/Mistral-7B-Instruct-v0.2"
    prompt_template_name="mistral"
elif [ "$model" = "Qwen2.5-14B" ]; then # 실행예정
    base_model="Qwen/Qwen2.5-14B-Instruct"
    prompt_template_name="llama"
elif [ "$model" = "Qwen3-14B-Instruct" ]; then # 실행예정
    base_model="Qwen/Qwen3-14B-Instruct"
    prompt_template_name="alpaca"
elif [ "$model" = "Qwen1.5-7B" ]; then
    base_model="Qwen/Qwen1.5-7B"
    prompt_template_name="qwen"
elif [ "$model" = "Llama-2-13B-chat" ]; then 
    base_model="meta-llama/Llama-2-13b-chat-hf"
    prompt_template_name="llama"
elif [ "$model" = "Llama-2-13B" ]; then
    base_model="meta-llama/Llama-2-13b-hf"
    prompt_template_name="llama"
elif [ "$model" = "Llama-2-7B-chat" ]; then 
    base_model="meta-llama/Llama-2-7b-chat-hf"
    prompt_template_name="llama"
elif [ "$model" = "Llama-2-7B" ]; then
    base_model="meta-llama/Llama-2-7b-hf" 
    prompt_template_name="llama"
elif [ "$model" = "Meta-Llama-3-8B" ]; then #현재 가장 우수
    base_model="meta-llama/Meta-Llama-3-8B"
    prompt_template_name="llama"
elif [ "$model" = "Meta-Llama-3-8B-Instruct" ]; then 
    base_model="meta-llama/Meta-Llama-3-8B-Instruct"
    prompt_template_name="llama"  # or "llama3"
elif [ "$model" = "Llama-3.1-8B" ]; then # 실행예정
    base_model="meta-llama/Llama-3.1-8B"
    prompt_template_name="llama"
elif [ "$model" = "Llama-3.1-8B-Instruct" ]; then # 실행예정
    base_model="meta-llama/Llama-3.1-8B-Instruct"
    prompt_template_name="llama"
else
    base_model=""
    prompt_template_name=""
fi
# CUDA_LAUNCH_BLOCKING=0 CUDA_VISIBLE_DEVICES=1 python inference.py \
CUDA_VISIBLE_DEVICES=0 python raft_inference/inference.py \
    --base_model $base_model \
    --lora_weights $model_path \
    --data_path $data_path \
    --output_data_path $output_path \
    --prompt_template $prompt_template_name \

