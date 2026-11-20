#!/bin/bash
# Stage 2: the modax test suite in a, then in b, same GPU.
MC=$(cat ~/disco-eb-p100/ROOT)/modax-check
G=${1:-0}
cd "$MC"
for e in a b; do
  RUN_TIMEOUT=${SUITE_TIMEOUT:-5400} bash scripts/mrun.sh $e $G pytest tests -v -rA --benchmark-disable -p no:cacheprovider > logs/pytest_$e.txt 2>&1
done
echo done > logs/chain2.done
