#!/bin/bash
# Stage 3 on GPU $1, run alone: warm timings, alternating a and b, three rounds.
MC=$(cat ~/disco-eb-p100/ROOT)/modax-check-03
G=${1:-0}
cd "$MC"
for i in 1 2 3; do for e in a b; do
  RUN_TIMEOUT=1200 bash scripts/mrun.sh $e $G $MC/scripts/timing.py > logs/timing_${e}${i}.txt 2>&1
done; done
echo done > logs/chain3.done
