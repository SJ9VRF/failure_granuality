#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
python code/run_fgt8_main.py
python code/run_fgt8_tax.py
for n in 0 0.2 0.4; do
  for b in 40 80 160; do
    python code/run_fgt8_phase_batch.py "$n" "$b"
  done
done
python code/run_fgt8_stress_axes.py
python code/run_fgt8_support_alt.py
python code/analyze_fgt8.py
python code/analyze_fgt8_extra.py
python code/fgt8_stats_main.py
python code/generate_auto_results.py
