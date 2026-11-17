#!/bin/bash
# Stage 2 on GPU $1: each side's own modax test suite, a then b.
MC=$(cat ~/disco-eb-p100/ROOT)/modax-check-03
G=${1:-1}
cd "$MC"
for e in a b; do
  RUN_TIMEOUT=${SUITE_TIMEOUT:-5400} bash scripts/mrun.sh $e $G pytest tests -v -rA --benchmark-disable -p no:cacheprovider > logs/pytest_$e.txt 2>&1
done
echo done > logs/chain2.done
