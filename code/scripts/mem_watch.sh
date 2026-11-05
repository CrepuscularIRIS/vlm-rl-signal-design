#!/bin/bash
# Sample container cgroup memory.current (bytes) every 5 s until $1 (a pid) exits; prints peak GiB at the end.
F=/sys/fs/cgroup/memory.current; [ -f $F ] || F=/sys/fs/cgroup/memory/memory.usage_in_bytes
OUT=${2:-/tmp/mem_watch.log}; peak=0
while kill -0 $1 2>/dev/null; do v=$(cat $F); [ $v -gt $peak ] && peak=$v; echo "$(date +%T) $v" >> $OUT; sleep 5; done
echo "peak_GiB $(python3 -c "print(round($peak/2**30,1))")" >> $OUT
