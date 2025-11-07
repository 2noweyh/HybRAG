CUDA_VISIBLE_DEVICES=1 python main.py ReaRev \
  --entity_dim 50 \
  --num_epoch 200 \
  --batch_size 8 \
  --eval_every 2 \
  --lm sbert \
  --num_iter 2 \
  --num_ins 3 \
  --num_gnn 3 \
  --dataset webqsp \
  --experiment_name prn_webqsp-rearev-sbert-dgl \
  --data_folder ./data/webqsp \
  --warmup_epoch 80 \
  --use_dgl True

CUDA_VISIBLE_DEVICES=1 python extract_triples.py \
  --checkpoint ./checkpoints/prn_webqsp-rearev-sbert-dgl-final_hw.ckpt \
  --dataset webqsp \
  --data_folder ./data/webqsp \
  --split test \
  --topk 5 \
  --use_dgl True \
  --save_path ./outputs/webqsp_triples.jsonl
