#!/bin/bash
# grpo_v53_dl (user 10-27): RLOO control vs RLOO + first-action fork supervision, protocol v5.2i in both arms.
# Everything else = B4 (b4_6k_r3) recipe: A-SFT start, 6k pool one pass (47 x 128 q x 16), actor mini-batch 32 q
# (4 optimizer updates / step), lr 1e-6 constant, grad norm 1.0, no KL, clip .2/.3, micro batch 1/2, single seed.
# Arm differences only via env: CAPTION_DL_BETA (0 = control, no forks), CAPTION_DL_K forks per action,
# CAPTION_CROP_LAMBDA (reward R = Y(1 - lambda N), both arms). CAPTION_DL_H0 = decision states per optimizer step.
# usage: LAMBDA=0.01 BETA=0.00390625 bash grpo_v53_dl.sh TAG GPUS [extra overrides, e.g. trainer.max_steps=1]
set -e
TAG=$1; GPUS=$2
shift 2 2>/dev/null || true
[ -n "$TAG" ] && [ -n "$GPUS" ] && [ -n "$LAMBDA" ] && [ -n "$BETA" ] || { echo "usage: LAMBDA=.. BETA=.. $0 TAG GPUS"; exit 2; }
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HOME=$ROOT/cache/hf
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd $EASYR1_DIR
export RAY_TMPDIR=/tmp/ray_v53_$TAG
export CAPTION_ENGINE_KWARGS="enable_prefix_caching=true,disable_log_stats=false"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CAPTION_GRPO_CODE=$PROJECT/grpo_v53_dl
export CUDA_VISIBLE_DEVICES=$GPUS
OUT=$PROJECT/outputs/v53_$TAG
mkdir -p $OUT
export CAPTION_ROLLOUT_LOG=$OUT/rollouts.jsonl
export CAPTION_DL_LOG=$OUT/forks.jsonl
export CAPTION_CROP_LAMBDA=$LAMBDA
export CAPTION_DL_BETA=$BETA
export CAPTION_DL_K=${K:-4}
export CAPTION_DL_H0=32
D=$ROOT/datasets
NGPU=$(echo $GPUS | tr ',' '\n' | wc -l)
export PATH=$ROOT/envs/easyr1/bin:$PATH
echo "[v53] tag=$TAG gpus=$GPUS lambda=$LAMBDA beta=$BETA k=$CAPTION_DL_K h0=$CAPTION_DL_H0 code_sha=$(cat $CAPTION_GRPO_CODE/*.py | sha256sum | cut -c1-16)"
python $CAPTION_GRPO_CODE/easyr1_caption.py \
    config=examples/config.yaml \
    data.train_files=$D/grpo_caption_v51_6k/train.parquet \
    data.val_files=$D/grpo_caption_v51_6k/val_nosft.parquet \
    data.prompt_key=prompt data.image_key=images data.answer_key=reward_model \
    data.max_prompt_length=4096 data.max_response_length=7168 \
    data.max_pixels=589824 data.min_pixels=1024 \
    data.rollout_batch_size=128 data.val_batch_size=192 data.seed=42 \
    data.filter_overlong_prompts=false \
    algorithm.adv_estimator=rloo algorithm.kl_coef=0.0 algorithm.disable_kl=true \
    algorithm.online_filtering=false \
    worker.actor.global_batch_size=32 worker.actor.max_grad_norm=1.0 \
    worker.actor.clip_ratio_low=0.2 worker.actor.clip_ratio_high=0.3 \
    worker.actor.model.model_path=$PROJECT/outputs/sft_v22_turnwise_merged \
    worker.actor.model.trust_remote_code=true \
    worker.actor.optim.lr=1.0e-6 worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.optim.lr_scheduler_type=constant \
    worker.actor.on_policy_reuse_log_probs=false \
    worker.actor.offload.offload_params=false worker.actor.offload.offload_optimizer=false \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.rollout.n=16 worker.rollout.temperature=0.7 worker.rollout.top_p=1.0 \
    worker.rollout.limit_images=10 worker.rollout.gpu_memory_utilization=0.45 \
    worker.rollout.tensor_parallel_size=1 worker.rollout.max_num_batched_tokens=32768 \
    worker.rollout.disable_tqdm=true \
    worker.reward.reward_function=$CAPTION_GRPO_CODE/reward_caption.py:compute_score \
    trainer.total_epochs=1 trainer.max_steps=null \
    trainer.project_name=caption_v53 trainer.experiment_name=v53_$TAG \
    trainer.logger='["console"]' trainer.n_gpus_per_node=$NGPU trainer.nnodes=1 \
    trainer.val_freq=25 trainer.val_before_train=true trainer.val_generations_to_log=0 \
    trainer.save_freq=2 trainer.save_limit=3 trainer.save_model_only=false \
    trainer.save_checkpoint_path=$OUT/ckpt \
    "${@:1}"
