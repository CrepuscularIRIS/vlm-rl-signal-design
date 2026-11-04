#!/bin/bash
# 10-27 mentor baseline classes 1/2 gaps: base direct answer, SFT CoT / direct answer / native zoom. One per GPU, E1 552 x 4.
P=$PROJECT; E=$P/evidence/baselines_v52_20261024; R=$E/code_v2/run_eval.sh
BASE=$ROOT/models/Qwen3-VL-4B-Instruct; SFT=$P/outputs/sft_v22_turnwise_merged
run() { echo "$(date "+%F %T") start $2 gpu$4" >> $E/launch.log; bash $R $1 $2 $3 $4 $E/full > $E/full/$2.log 2>&1; echo "$(date "+%F %T") $2 exit=$?" >> $E/launch.log; }
run answer B0_answer $BASE 0 &
run direct B3_cot $SFT 1 &
run answer B3_answer $SFT 2 &
run native B3_native $SFT 3 &
wait
