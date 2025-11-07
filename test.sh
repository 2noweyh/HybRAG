# vfinal 경로 5개 
# vfinal10 경로 10개 
# vrearev , alpaca + []
# vrearev_2, alpaca + ,
# vrearev_3, llama + ,
# vrearev_4, llama + , +dense5(15)
# vrearev_5, 4에서 path만



# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v11.py --dataset webqsp --version vrearev_4 --mode both --split trn
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v11.py --dataset webqsp --version vrearev_4 --mode both --split tst
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v11.py --dataset webqsp --version vrearev_4 --mode both --split val

# bash raft_training/train.sh 2 Llama-2-7B-chat webqsp vrearev_4 both
# bash raft_inference/inference.sh Llama-2-7B-chat webqsp vrearev_4 both
# bash raft_evaluation/evaluation.sh Llama-2-7B-chat webqsp vrearev_4 both

# bash raft_training/train.sh 2 Llama-2-7B webqsp vrearev_4 both
# bash raft_inference/inference.sh Llama-2-7B webqsp vrearev_4 both
# bash raft_evaluation/evaluation.sh Llama-2-7B webqsp vrearev_4 both

# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v11.py --dataset webqsp --version vrearev_5 --mode path --split trn
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v11.py --dataset webqsp --version vrearev_5 --mode path --split tst
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v11.py --dataset webqsp --version vrearev_5 --mode path --split val

# bash raft_training/train.sh 2 Llama-2-7B-chat webqsp vrearev_5 path
# bash raft_inference/inference.sh Llama-2-7B-chat webqsp vrearev_5 path
# bash raft_evaluation/evaluation.sh Llama-2-7B-chat webqsp vrearev_5 path

# bash raft_training/train.sh 2 Meta-Llama-3-8B-Instruct webqsp vrearev_4 both
# bash raft_inference/inference.sh Meta-Llama-3-8B-Instruct webqsp vrearev_4 both
# bash raft_evaluation/evaluation.sh Meta-Llama-3-8B-Instruct webqsp vrearev_4 both

# bash raft_training/train.sh 2 Qwen2.5-14B webqsp vrearev_4 both
# bash raft_inference/inference.sh Qwen2.5-14B webqsp vrearev_4 both
# bash raft_evaluation/evaluation.sh Qwen2.5-14B webqsp vrearev_4 both

# DA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v11.py --dataset webqsp --version vrearev_3 --mode both --split trn
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v11.py --dataset webqsp --version vrearev_3 --mode both --split tst
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v11.py --dataset webqsp --version vrearev_3 --mode both --split val

# bash raft_training/train.sh 2 Llama-2-7B webqsp vrearev both
# bash raft_training/train.sh 2 Llama-2-7B-chat webqsp vrearev_3 both

# bash raft_inference/inference.sh Llama-2-7B webqsp vrearev both
# bash raft_inference/inference.sh Llama-2-7B-chat webqsp vrearev_3 both

# bash raft_evaluation/evaluation.sh Llama-2-7B webqsp vrearev both
# bash raft_evaluation/evaluation.sh Llama-2-7B-chat webqsp vrearev_3 both


# DA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v11.py --dataset webqsp --version vrearev --mode both --split trn
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v11.py --dataset webqsp --version vrearev --mode both --split tst
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v11.py --dataset webqsp --version vrearev --mode both --split val

# bash raft_training/train.sh 2 Llama-2-7B webqsp vrearev both
# bash raft_training/train.sh 2 Llama-2-7B-chat webqsp vrearev both

# bash raft_inference/inference.sh Llama-2-7B webqsp vrearev both
# bash raft_inference/inference.sh Llama-2-7B-chat webqsp vrearev both

# bash raft_evaluation/evaluation.sh Llama-2-7B webqsp vrearev both
# bash raft_evaluation/evaluation.sh Llama-2-7B-chat webqsp vrearev both

# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal10 --mode both --split trn
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal10 --mode both --split tst
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal10 --mode both --split val
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal10 --mode dense --split trn
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal10 --mode dense --split tst
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal10 --mode dense --split val


# bash raft_training/train.sh 2 Llama-2-7B webqsp vfinal10 both
# bash raft_training/train.sh 2 Llama-2-7B webqsp vfinal10 dense

# bash raft_training/train.sh 2 Llama-2-7B-chat webqsp vfinal10 both
# bash raft_training/train.sh 2 Llama-2-7B-chat webqsp vfinal10 dense

# bash raft_training/train.sh 2 Meta-Llama-3-8B webqsp vfinal10 both
# bash raft_training/train.sh 2 Meta-Llama-3-8B webqsp vfinal10 dense

# bash raft_training/train.sh 2 Meta-Llama-3-8B-Instruct webqsp vfinal10 both
# bash raft_training/train.sh 2 Meta-Llama-3-8B-Instruct webqsp vfinal10 dense


# bash raft_inference/inference.sh Llama-2-7B webqsp vfinal10 both
# bash raft_inference/inference.sh Llama-2-7B webqsp vfinal10 dense

# bash raft_inference/inference.sh Llama-2-7B-chat webqsp vfinal10 both
# bash raft_inference/inference.sh Llama-2-7B-chat webqsp vfinal10 dense

# bash raft_inference/inference.sh Meta-Llama-3-8B webqsp vfinal10 both
# bash raft_inference/inference.sh Meta-Llama-3-8B webqsp vfinal10 dense

# bash raft_inference/inference.sh Meta-Llama-3-8B-Instruct webqsp vfinal10 both
# bash raft_inference/inference.sh Meta-Llama-3-8B-Instruct webqsp vfinal10 dense



# bash raft_evaluation/evaluation.sh Llama-2-7B webqsp vfinal10 both
# bash raft_evaluation/evaluation.sh Llama-2-7B webqsp vfinal10 dense

# bash raft_evaluation/evaluation.sh Llama-2-7B-chat webqsp vfinal10 both
# bash raft_evaluation/evaluation.sh Llama-2-7B-chat webqsp vfinal10 dense

# bash raft_evaluation/evaluation.sh Meta-Llama-3-8B webqsp vfinal10 both
# bash raft_evaluation/evaluation.sh Meta-Llama-3-8B webqsp vfinal10 dense

# bash raft_evaluation/evaluation.sh Meta-Llama-3-8B-Instruct webqsp vfinal10 both
# bash raft_evaluation/evaluation.sh Meta-Llama-3-8B-Instruct webqsp vfinal10 dense



# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal10-15 --mode both --split trn
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal10-15 --mode both --split tst
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal10-15 --mode both --split val
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal10-15 --mode dense --split trn
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal10-15 --mode dense --split tst
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal10-15 --mode dense --split val


# bash raft_training/train.sh 2 Llama-2-7B webqsp vfinal10-15 both
# bash raft_training/train.sh 2 Llama-2-7B webqsp vfinal10-15 dense

# bash raft_training/train.sh 2 Llama-2-7B-chat webqsp vfinal10-15 both
# bash raft_training/train.sh 2 Llama-2-7B-chat webqsp vfinal10-15 dense

# 여기부터 안함
# bash raft_training/train.sh 2 Meta-Llama-3-8B webqsp vfinal10-15 both
# bash raft_training/train.sh 2 Meta-Llama-3-8B webqsp vfinal10-15 dense

# bash raft_training/train.sh 2 Meta-Llama-3-8B-Instruct webqsp vfinal10-15 both
# bash raft_training/train.sh 2 Meta-Llama-3-8B-Instruct webqsp vfinal10-15 dense


# bash raft_inference/inference.sh Llama-2-7B webqsp vfinal10-15 both
# bash raft_inference/inference.sh Llama-2-7B webqsp vfinal10-15 dense

# bash raft_inference/inference.sh Llama-2-7B-chat webqsp vfinal10-15 both
# bash raft_inference/inference.sh Llama-2-7B-chat webqsp vfinal10-15 dense

# bash raft_inference/inference.sh Meta-Llama-3-8B webqsp vfinal10-15 both
# bash raft_inference/inference.sh Meta-Llama-3-8B webqsp vfinal10-15 dense

# bash raft_inference/inference.sh Meta-Llama-3-8B-Instruct webqsp vfinal10-15 both
# bash raft_inference/inference.sh Meta-Llama-3-8B-Instruct webqsp vfinal10-15 dense



# bash raft_evaluation/evaluation.sh Llama-2-7B webqsp vfinal10-15 both
# bash raft_evaluation/evaluation.sh Llama-2-7B webqsp vfinal10-15 dense

# bash raft_evaluation/evaluation.sh Llama-2-7B-chat webqsp vfinal10-15 both
# bash raft_evaluation/evaluation.sh Llama-2-7B-chat webqsp vfinal10-15 dense

# bash raft_evaluation/evaluation.sh Meta-Llama-3-8B webqsp vfinal10-15 both
# bash raft_evaluation/evaluation.sh Meta-Llama-3-8B webqsp vfinal10-15 dense

# bash raft_evaluation/evaluation.sh Meta-Llama-3-8B-Instruct webqsp vfinal10-15 both
# bash raft_evaluation/evaluation.sh Meta-Llama-3-8B-Instruct webqsp vfinal10-15 dense



# ==== Evaluation results on webqsp ====
# Accuracy: 25.4526
# Hit: 42.8133
# F1: 31.1769
# Precision: 82.3096
# Recall: 25.4526

# ==== Evaluation results on webqsp ====
# Accuracy: 35.4209
# Hit: 54.7297
# F1: 44.3010
# Precision: 111.6093
# Recall: 35.4209

# ==== Evaluation results on webqsp ====
# Accuracy: 24.1399
# Hit: 41.0319
# F1: 29.8719
# Precision: 77.6413
# Recall: 24.1399

# ==== Evaluation results on webqsp ====
# Accuracy: 34.9865
# Hit: 53.1327
# F1: 43.6016
# Precision: 112.4693
# Recall: 34.9865

# ==== Evaluation results on webqsp ====
# Accuracy: 35.7537
# Hit: 53.9926
# F1: 43.2199
# Precision: 109.2752
# Recall: 35.7537

# ==== Evaluation results on webqsp ====
# Accuracy: 47.9927
# Hit: 67.5676
# F1: 58.8154
# Precision: 141.7690
# Recall: 47.9927

# ==== Evaluation results on webqsp ====
# Accuracy: 39.4763
# Hit: 57.1867
# F1: 48.0776
# Precision: 120.8231
# Recall: 39.4763

# ==== Evaluation results on webqsp ====
# Accuracy: 47.4184
# Hit: 66.5848
# F1: 58.4733
# Precision: 136.6093
# Recall: 47.4184




# CUDA_VISIBLE_DEVICES=0 python data_preprocess/translation_pruning_v11.py --dataset webqsp --version vrearev_6 --mode dense --split trn
# CUDA_VISIBLE_DEVICES=0 python data_preprocess/translation_pruning_v11.py --dataset webqsp --version vrearev_6 --mode dense --split tst
# CUDA_VISIBLE_DEVICES=0 python data_preprocess/translation_pruning_v11.py --dataset webqsp --version vrearev_6 --mode dense --split val

# bash raft_training/train.sh 2 Llama-2-7B-chat webqsp vrearev_6 dense
# bash raft_inference/inference.sh Llama-2-7B-chat webqsp vrearev_6 dense
# bash raft_evaluation/evaluation.sh Llama-2-7B-chat webqsp vrearev_6 dense

# CUDA_VISIBLE_DEVICES=0 python data_preprocess/translation_pruning_v11.py --dataset cwq --version vrearev_4 --mode both --split trn
# CUDA_VISIBLE_DEVICES=0 python data_preprocess/translation_pruning_v11.py --dataset cwq --version vrearev_4 --mode both --split tst
# CUDA_VISIBLE_DEVICES=0 python data_preprocess/translation_pruning_v11.py --dataset cwq --version vrearev_4 --mode both --split val

# bash raft_training/train.sh 2 Llama-2-7B-chat cwq vrearev_4 both
# bash raft_inference/inference.sh Llama-2-7B-chat cwq vrearev_4 both
# bash raft_evaluation/evaluation.sh Llama-2-7B-chat cwq vrearev_4 both

bash raft_training/train.sh 2 Meta-Llama-3-8B-Instruct cwq vrearev_4 both
bash raft_inference/inference.sh Meta-Llama-3-8B-Instruct cwq vrearev_4 both
bash raft_evaluation/evaluation.sh Meta-Llama-3-8B-Instruct cwq vrearev_4 both

# CUDA_VISIBLE_DEVICES=0 python data_preprocess/translation_pruning_v11.py --dataset cwq --version vrearev_5 --mode path --split trn
# CUDA_VISIBLE_DEVICES=0 python data_preprocess/translation_pruning_v11.py --dataset cwq --version vrearev_5 --mode path --split tst
# CUDA_VISIBLE_DEVICES=0 python data_preprocess/translation_pruning_v11.py --dataset cwq --version vrearev_5 --mode path --split val

# bash raft_training/train.sh 2 Meta-Llama-3-8B-Instruct cwq vrearev_5 path
# bash raft_inference/inference.sh Meta-Llama-3-8B-Instruct cwq vrearev_5 path
# bash raft_evaluation/evaluation.sh Meta-Llama-3-8B-Instruct cwq vrearev_5 path

# CUDA_VISIBLE_DEVICES=0 python data_preprocess/translation_pruning_v11.py --dataset cwq --version vrearev_6 --mode dense --split trn
# CUDA_VISIBLE_DEVICES=0 python data_preprocess/translation_pruning_v11.py --dataset cwq --version vrearev_6 --mode dense --split tst
# CUDA_VISIBLE_DEVICES=0 python data_preprocess/translation_pruning_v11.py --dataset cwq --version vrearev_6 --mode dense --split val

# bash raft_training/train.sh 2 Meta-Llama-3-8B-Instruct cwq vrearev_6 dense
# bash raft_inference/inference.sh Meta-Llama-3-8B-Instruct cwq vrearev_6 dense
# bash raft_evaluation/evaluation.sh Meta-Llama-3-8B-Instruct cwq vrearev_6 dense

# bash raft_training/train.sh 2 Meta-Llama-3-8B-Instruct webqsp vrearev_5 path
# bash raft_inference/inference.sh Meta-Llama-3-8B-Instruct webqsp vrearev_5 path
# bash raft_evaluation/evaluation.sh Meta-Llama-3-8B-Instruct webqsp vrearev_5 path

# bash raft_training/train.sh 2 Meta-Llama-3-8B-Instruct webqsp vrearev_6 dense
# bash raft_inference/inference.sh Meta-Llama-3-8B-Instruct webqsp vrearev_6 dense
# bash raft_evaluation/evaluation.sh Meta-Llama-3-8B-Instruct webqsp vrearev_6 dense

# bash raft_training/train.sh 2 Qwen2.5-14B cwq vrearev_4 both
# bash raft_inference/inference.sh Qwen2.5-14B cwq vrearev_4 both
# bash raft_evaluation/evaluation.sh Qwen2.5-14B cwq vrearev_4 both
