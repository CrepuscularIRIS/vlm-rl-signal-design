#!/bin/bash
# GPU0: H1 (answer-only SFT) then H2 (answer-only SFT + GRPO step 160), answer-only protocol. Name kept: the B4 queue waits on it.
cd $PROJECT; E=evidence/baselines_v52_20261024
bash $E/code/run_eval.sh caption_ao H1_ao_sft outputs/sft_v22_answeronly_merged 0 $E/full > $E/full/H1.log 2>&1; rc=$?
echo "$(date "+%F %T") queue: H1 exit $rc, start H2" >> $E/launch.log
bash $E/code/run_eval.sh caption_ao H2_ao_rl160 outputs/grpo_v3_AO_base/global_step_160/actor/huggingface 0 $E/full > $E/full/H2.log 2>&1; rc=$?
echo "$(date "+%F %T") queue: H2 exit $rc" >> $E/launch.log
