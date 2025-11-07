# bash raft_evaluation/evaluation.sh Qwen2.5-14B webqsp v1 dense
# bash raft_evaluation/evaluation.sh Meta-Llama-3-8B webqsp v0921 both
# bash raft_evaluation/evaluation.sh Meta-Llama-3-8B-Instruct webqsp vfinal both


# bash raft_evaluation/evaluation.sh Llama-2-7B-chat webqsp vrearev_2 both

model=$1
dataset=$2
version=$3
mode=$4
# mode="both-dense"

data_path="./outputs/${dataset}/${model}_${version}_result_${mode}.jsonl"
log_path="./outputs/${dataset}/${model}_${version}_eval_metrics_${mode}.json"
output_path="./outputs/${dataset}/${model}_${version}_eval_results_${mode}.jsonl"


python raft_evaluation/evaluation_v2.py \
  --dataset webqsp \
  --input_path $data_path \
  --tb_logdir  $log_path \
  --save_path $output_path