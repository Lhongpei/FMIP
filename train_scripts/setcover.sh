#!/bin/bash

python train.py \
  --storage_path ./result_cache \
  --dataset_root /data/GM4MILP/instances_final \
  --dataset_name setcover \
  --dataset_cache dataset_cache \
  --training_split train \
  --validation_split valid \
  --test_split test \
  --validation_examples -1 \
  --inference_type batch \
  --c_d_weight 1.0 \
  --batch_size 16 \
  --num_epochs 300 \
  --learning_rate 2e-4 \
  --weight_decay 1e-4 \
  --lr_scheduler cosine-decay \
  --num_workers 8 \
  --num_discrete_sample 10 \
  --num_continuous_sample 10 \
  --num_discrete_guide_step 10 \
  --num_continuous_guide_step 10 \
  --discrete_flow_mode uniform \
  --flow_schedule linear \
  --diffusion_steps 1000 \
  --inference_steps 50 \
  --inference_schedule cosine \
  --inference_trick ddim \
  --sequential_sampling 3 \
  --parallel_sampling 64 \
  --n_layers 12 \
  --hidden_dim 64 \
  --project_name GMIP \
  --mode primal \
  --do_train\
  --only_discrete\
  --use_focal_loss \
  # --ckpt_path result_cache/models/setcover/2025-04-24_08-16-22/checkpoints/last.ckpt