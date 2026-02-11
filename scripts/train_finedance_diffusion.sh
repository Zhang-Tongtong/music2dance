#!/bin/bash


# export CUDA_VISIBLE_DEVICES=0

EXP_NAME="mld_finedance_h2_f8_r8"
DATA_DIR="./data/finedance"
CFG_PATH="./config_files/config_hydra/motion_primitive/finedance_h2_f8_r8.yaml"
VAE_PATH="./mvae/mvae_finedance_h2_f8_r8/checkpoint_200000.pt"

echo "============================================"
echo "Training Music-to-Dance Diffusion Model"
echo "============================================"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_DIR"
echo "VAE: $VAE_PATH"
echo "Using python -m (allows underscore args)"
echo "============================================"


python -m mld.train_mld_music \
    --exp_name $EXP_NAME \
    --data_args.dataset finedance \
    --data_args.data_dir $DATA_DIR \
    --data_args.cfg_path $CFG_PATH \
    --data_args.enforce_gender male \
    --data_args.enforce_zero_beta 1 \
    --data_args.body_type smplx \
    --denoiser_args.mvae_path $VAE_PATH \
    --denoiser_args.rescale_latent 1 \
    --denoiser_args.train_rollout_type full \
    --denoiser_args.train_rollout_history rollout \
    --denoiser_args.model_type transformer \
    --denoiser_args.diffusion_args.diffusion_steps 1000 \
    --denoiser_args.diffusion_args.noise_schedule cosine \
    # --denoiser_args.diffusion_args.sigma_small 1 \
    --train_args.batch_size 512 \
    --train_args.learning_rate 0.0001 \
    --train_args.anneal_lr 1 \
    --train_args.grad_clip 1.0 \
    --train_args.ema_decay 0.999 \
    --train_args.use_amp 1 \
    --train_args.stage1_steps 100000 \
    --train_args.stage2_steps 100000 \
    --train_args.stage3_steps 100000 \
    --train_args.log_interval 1000 \
    --train_args.save_interval 50000 \
    --train_args.val_interval 10000 \
    --train_args.weight_latent_rec 1.0 \
    --train_args.weight_feature_rec 1.0 \
    --train_args.weight_joints_delta 10000.0 \
    --train_args.weight_transl_delta 10000.0 \
    --train_args.weight_orient_delta 10000.0 \
    --train_args.weight_joints_consistency 0.0 \
    --train_args.weight_smpl_joints_rec 0.0 \
    --device cuda \
    # --torch_deterministic 1 \
    --seed 0 \
    --track 1 \
    --wandb_project_name mld_music_to_dance \
    --wandb_entity interaction \
    denoiser_args.model_args:denoiser_transformer_args \
    --denoiser_args.model_args.h_dim 512 \
    --denoiser_args.model_args.ff_size 1024 \
    --denoiser_args.model_args.num_layers 8 \
    --denoiser_args.model_args.num_heads 4 \
    --denoiser_args.model_args.dropout 0.1 \
    --denoiser_args.model_args.activation gelu \
    --denoiser_args.model_args.cond_mask_prob 0.1 \
    --denoiser_args.model_args.music_dim 35 \
    --denoiser_args.model_args.history_shape 2 276 \
    --denoiser_args.model_args.noise_shape 1 256

echo "============================================"
echo "Training completed!"
echo "============================================"