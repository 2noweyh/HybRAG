model=$1
dataset=$2
version=$3
mode=$4

data_path="./outputs/${dataset}/${model}_${version}_result_${mode}.jsonl"
log_path="./outputs/${dataset}/${model}_${version}_eval_metrics_${mode}.json"
output_path="./outputs/${dataset}/${model}_${version}_eval_results_${mode}.jsonl"


python raft_evaluation/evaluation.py \
  --dataset webqsp \
  --input_path $data_path \
  --tb_logdir  $log_path \
  --save_path $output_path