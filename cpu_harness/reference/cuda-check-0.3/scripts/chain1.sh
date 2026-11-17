#!/bin/bash
# Stage 1 on GPU $1: target checks, bitwise solves (a, b, a again, b again), kernel-source dumps.
MC=$(cat ~/disco-eb-p100/ROOT)/modax-check-03
G=${1:-0}
cd "$MC"
for e in a b; do
  RUN_TIMEOUT=1200 bash scripts/mrun.sh $e $G $MC/scripts/check_target.py > logs/target_$e.txt 2>&1
done
for run in a:1 b:1 a:2 b:2; do
  e=${run%%:*}; i=${run##*:}
  RUN_TIMEOUT=3000 bash scripts/mrun.sh $e $G $MC/scripts/solves.py $MC/out/solves_${e}${i}.npz > logs/solves_${e}${i}.txt 2>&1
done
for e in a b; do
  RUN_TIMEOUT=2400 bash scripts/mrun.sh $e $G $MC/scripts/dump_hier.py $MC/out/dump_$e > logs/dump_$e.txt 2>&1
done
echo done > logs/chain1.done
