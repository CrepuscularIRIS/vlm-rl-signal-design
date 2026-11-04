#!/bin/bash
set -euo pipefail
TAG=$1; PHASE=$2; GPUS=$3; TARGET=32; BATCH=128; WARMUP=1
if [ "$PHASE" = smoke ]; then TARGET=1; BATCH=32; WARMUP=0; fi
P=$PROJECT; C=$P/sapo_vlm_20261101; OUT=$P/outputs/sapo_$TAG
export CAPTION_GRPO_CODE=$C CAPTION_LAZY_MM=1
export SAPO_OUTPUT=$OUT SAPO_PHASE=$PHASE
export CAPTION_ROLLOUT_LOG=$OUT/rollouts.jsonl CAPTION_UPDATE_LOG=$OUT/updates.jsonl
export CUDA_VISIBLE_DEVICES=$GPUS RAY_ADDRESS=local RAY_TMPDIR=/tmp/sa_$TAG
export RAY_DEDUP_LOGS=0 PYTHONUNBUFFERED=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export CAPTION_ENGINE_KWARGS='enable_prefix_caching=true,disable_log_stats=false'
export CAPTION_MM_CACHE_GB=2 CAPTION_MM_PREFETCH=2 CAPTION_MM_PREFETCH_WINDOW=8
export MALLOC_ARENA_MAX=2 RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=34359738368
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$ROOT/cache/hf
export PATH=$ROOT/envs/easyr1/bin:$PATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
cd $EASYR1_DIR
mkdir "$OUT"
VAL=true; SAVE=5; MODEL_ONLY=false
if [ "$PHASE" = smoke ]; then VAL=false; SAVE=1; MODEL_ONLY=true; fi
python "$C/easyr1_caption.py" config=examples/config.yaml \
 data.train_files=$ROOT/datasets/grpo_caption_v51_4k/train.parquet \
 data.val_files=$ROOT/datasets/grpo_caption_v51_4k/val_nosft.parquet \
 data.prompt_key=prompt data.image_key=images data.answer_key=reward_model \
 data.max_prompt_length=4096 data.max_response_length=7168 data.max_pixels=589824 data.min_pixels=1024 \
 data.rollout_batch_size=$BATCH data.mini_rollout_batch_size=32 data.val_batch_size=192 data.seed=42 data.filter_overlong_prompts=false \
 algorithm.adv_estimator=rloo algorithm.gamma=0.95 algorithm.lam=0.95 algorithm.kl_coef=0.0 algorithm.disable_kl=true \
 algorithm.online_filtering=false algorithm.filter_key=overall algorithm.filter_low=0.01 algorithm.filter_high=0.99 \
 worker.actor.global_batch_size=32 worker.actor.max_grad_norm=1.0 worker.actor.clip_ratio_low=0.2 worker.actor.clip_ratio_high=0.2 \
 worker.actor.loss_avg_mode=token worker.actor.ppo_epochs=1 \
 worker.actor.model.model_path=$P/outputs/sft_v22_turnwise_merged worker.actor.model.trust_remote_code=true \
 worker.actor.optim.lr=1.0e-6 worker.actor.optim.lr_warmup_steps=$WARMUP worker.actor.optim.lr_scheduler_type=constant \
 worker.actor.on_policy_reuse_log_probs=false worker.actor.offload.offload_params=false worker.actor.offload.offload_optimizer=false \
 worker.actor.micro_batch_size_per_device_for_update=1 worker.actor.micro_batch_size_per_device_for_experience=1 \
 worker.rollout.n=1 worker.rollout.seed=42 worker.rollout.temperature=0.7 worker.rollout.top_p=1.0 \
 worker.rollout.limit_images=10 worker.rollout.gpu_memory_utilization=0.45 worker.rollout.tensor_parallel_size=1 \
 worker.rollout.max_num_batched_tokens=32768 worker.rollout.disable_tqdm=true \
 worker.reward.reward_function=$C/reward_caption.py:compute_score \
 trainer.total_epochs=1 trainer.max_steps=$TARGET trainer.max_try_make_batch=100 \
 trainer.project_name=caption_sapo trainer.experiment_name=sapo_$TAG trainer.logger='["console"]' \
 trainer.n_gpus_per_node=2 trainer.nnodes=1 trainer.val_freq=25 trainer.val_before_train=$VAL trainer.val_generations_to_log=0 \
 trainer.save_freq=$SAVE trainer.save_limit=2 trainer.save_model_only=$MODEL_ONLY trainer.save_checkpoint_path=$OUT/ckpt
