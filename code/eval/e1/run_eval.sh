#!/bin/bash
# Baseline re-evaluation with the fixed runners (docs/40 I1-I3, I13). One condition per GPU.
# usage: run_eval.sh {direct|native|caption|caption_ao} TAG MODEL GPU OUT_DIR [extra runner args...]
# direct/native: eval_base.py (full question text, all tool calls, 32k context / 8192 total generation)
# caption:       eval_v4.py with code dir grpo_v51fix_codex (bbox refusal, same 32k / 8192 budget)
set -e
MODE=$1; TAG=$2; MODEL=$3; GPU=$4; OUT=$5; shift 5
P=$PROJECT; C=$P/evidence/baselines_v52_20261024/code_v2
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HOME=$ROOT/cache/hf VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=$GPU
export DATA_DIR=$ROOT/datasets EASYR1_DIR=$EASYR1_DIR PROJECT_DIR=$P
export PATH=$ROOT/envs/easyr1/bin:$PATH
cd $P
if [ "$MODE" = caption ]; then
  export CAPTION_GRPO_CODE=$P/grpo_v51fix_codex
  python $C/eval_v4.py --model $MODEL --code-dir $P/grpo_v51fix_codex --tag $TAG --out-dir $OUT "$@"
elif [ "$MODE" = caption_ao ]; then  # H1/H2 answer-only protocol (grpo_v51_ansonly + bbox refusal)
  export CAPTION_GRPO_CODE=$P/grpo_v51fix_codex/notool CAPTION_CROP_LAMBDA=0  # ansonly reward_caption reads it at import; eval only uses _norm_answer
  python $C/eval_v4.py --model $MODEL --code-dir $(dirname $C)/code_ao --tag $TAG --out-dir $OUT "$@"
else
  export CAPTION_GRPO_CODE=$P/grpo_v51fix_codex/notool  # verl/__init__ needs a code dir; no-op patch
  python $C/eval_base.py --mode $MODE --model $MODEL --code-dir $P/grpo_v51fix_codex --tag $TAG --out-dir $OUT "$@"
fi
