#!/bin/bash
# 10-29: merge GiGPO/base_grpo step 32, then E1 caption eval of the five k8 step-32 models on 4 GPUs.
P=$PROJECT; cd $P; E=$P/evidence/k8_20261028
R=$P/evidence/baselines_v52_20261024/code_v2/run_eval.sh; F=$P/evidence/baselines_v52_20261024/full
log(){ echo "$(date "+%F %T") $*" >> $E/launch.log; }
ev(){ log "eval start $1 gpu=$3"; bash $R caption $1 $2 $3 $F > $F/$1.log 2>&1; log "eval $1 exit=$?"; }
merge(){ log "merge start $1"; $ROOT/envs/easyr1/bin/python $EASYR1_DIR/scripts/model_merger.py --local_dir $2/global_step_32/actor > $E/merge_$1.log 2>&1; log "merge $1 exit=$?"; }
H=ckpt/global_step_32/actor/huggingface
ev k8_grpo $P/outputs/v52mc_k8_grpo/$H 0 &
ev k8_dapo $P/outputs/v52mc_k8_dapo/$H 1 &
ev k8_gspo $P/outputs/v52mc_k8_gspo/$H 2 &
merge k8_gigpo $P/outputs/gigpo_k8_gigpo/ckpt &
merge k8_base_grpo $P/outputs/v52mc_k8_base_grpo/ckpt
wait %4
ev k8_gigpo $P/outputs/gigpo_k8_gigpo/$H 3 &
wait %1
ev k8_base_grpo $P/outputs/v52mc_k8_base_grpo/$H 0
wait
log "eval_k8 done"
