model=$1
dataset=$2
version=$3
mode=$4

model_path="./models/${dataset}/${model}_${version}_${mode}"
data_path="/home/noweyh/hw_work/RAG/hybRAG/data/${dataset}/rerank_test_${version}_${mode}.json"
output_path="./outputs/${dataset}/${model}_${version}_result_${mode}.jsonl"



if [ "$model" = "Flan-T5-XXL" ]; then
    base_model="google/flan-t5-xxl"
    prompt_template_name="alpaca"
elif [ "$model" = "Flan-T5-XL" ]; then
    base_model="google/flan-t5-xl"
    prompt_template_name="alpaca"
elif [ "$model" = "Phi-2" ]; then
    base_model="microsoft/phi-2"
    prompt_template_name="alpaca"
elif [ "$model" = "Mistral-7B-Instruct-v0.2" ]; then 
    base_model="mistralai/Mistral-7B-Instruct-v0.2"
    prompt_template_name="mistral"
elif [ "$model" = "Qwen2.5-14B" ]; then 
    base_model="Qwen/Qwen2.5-14B-Instruct"
    prompt_template_name="llama"
elif [ "$model" = "Qwen3-14B-Instruct" ]; then
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
elif [ "$model" = "Meta-Llama-3-8B" ]; then 
    base_model="meta-llama/Meta-Llama-3-8B"
    prompt_template_name="llama"
elif [ "$model" = "Meta-Llama-3-8B-Instruct" ]; then 
    base_model="meta-llama/Meta-Llama-3-8B-Instruct"
    prompt_template_name="llama" 
elif [ "$model" = "Llama-3.1-8B" ]; then 
    base_model="meta-llama/Llama-3.1-8B"
    prompt_template_name="llama"
elif [ "$model" = "Llama-3.1-8B-Instruct" ]; then 
    base_model="meta-llama/Llama-3.1-8B-Instruct"
    prompt_template_name="llama"
else
    base_model=""
    prompt_template_name=""
fi


CUDA_VISIBLE_DEVICES=0 python raft_inference/inference.py \
    --base_model $base_model \
    --lora_weights $model_path \
    --data_path $data_path \
    --output_data_path $output_path \
    --prompt_template $prompt_template_name \

