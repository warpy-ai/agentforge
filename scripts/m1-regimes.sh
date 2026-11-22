#!/usr/bin/env bash
# Replay the four M0 regimes and the TraceLab-shaped workload with every residency policy.
# Regime A: FP16 KV, the device saturates on decode traffic.  B: FP8 KV with a scarce, slow host
# tier.  C: FP8 KV without contention.  D: FP8 KV, 80 sessions, 2 TB host tier (the thesis case).
# Usage: scripts/m1-regimes.sh [seed ...]   (regime D and TraceLab seeds; default 1 11 23 42); prints markdown.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build -q --release -p agentforge-cli
bin=target/release/agentforge
seeds=("$@")
[ ${#seeds[@]} -eq 0 ] && seeds=(1 11 23 42)
run() {
	local name=$1; shift
	echo "### Regime $name"
	echo
	"$bin" sim run --tp 8 --steps 20 --policy all "$@"
	echo
}
# The regimes as the design doc defines them (docs/design/0001-approach.md, "First simulation results").
run "A (FP16 KV, 64 sessions, 2 TB host, H200 x8, seed 1)" --machine h200 --kv-precision fp16 --sessions 64 --seed 1
run "B (FP8 KV, 80 sessions, 128 GB host, H200 x8, seed 11)" --machine h200 --kv-precision fp8 --sessions 80 --host-gb 128 --seed 11
run "C (FP8 KV, 80 sessions, 128 GB host, B200 x8, seed 11)" --machine b200 --kv-precision fp8 --sessions 80 --host-gb 128 --seed 11
for seed in "${seeds[@]}"; do
	run "D (FP8 KV, 80 sessions, 2 TB host, H200 x8, seed $seed)" --machine h200 --kv-precision fp8 --sessions 80 --seed "$seed"
done
for seed in "${seeds[@]}"; do
	echo "### TraceLab program (FP8 KV, 40 sessions, 2 TB host, H200 x8, seed $seed)"
	echo
	"$bin" sim run --machine h200 --tp 8 --kv-precision fp8 --sessions 40 --program tracelab --policy all --seed "$seed"
	echo
done
