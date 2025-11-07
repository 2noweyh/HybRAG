export MODEL_PATH='ckpts/webqsp_rag_v1'

CUDA_VISIBLE_DEVICES=0 python3  raft_inference/main_infer.py \
    --model_path $MODEL_PATH \
    --strategy Parallel \
    --batch_size 4 \
    --max_tokens 1024 \
    --save_dir data/webqsp \
    --is_bf16

# batch_size 16
# max_tokens 1024
# bash raft_inference/infer.sh