#!/bin/bash
# Stage 1: bitwise solves (a, b, a again, b again), kernel-source dumps, timings. GPU from $1.
MC=$(cat ~/disco-eb-p100/ROOT)/modax-check
G=${1:-0}
cd "$MC"
for run in a:1 b:1 a:2 b:2; do
  e=${run%%:*}; i=${run##*:}
  RUN_TIMEOUT=2400 bash scripts/mrun.sh $e $G $MC/scripts/solves.py $MC/out/solves_${e}${i}.npz > logs/solves_${e}${i}.txt 2>&1
done
for e in a b; do
  RUN_TIMEOUT=1800 bash scripts/mrun.sh $e $G $MC/scripts/dump_hier.py $MC/out/dump_$e > logs/dump_$e.txt 2>&1
done
for i in 1 2 3; do for e in a b; do
  RUN_TIMEOUT=1200 bash scripts/mrun.sh $e $G $MC/scripts/timing.py > logs/timing_${e}${i}.txt 2>&1
done; done
echo done > logs/chain1.done
