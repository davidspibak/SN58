#!/usr/bin/env bash
# A private duel with the validator's own engine: KING's weights against CHALLENGER's, on fresh scenes
# from the duel seed range, scored as the validator scores (per-task rates, +3 points to win).
#   local_duel.sh KING.safetensors CHALLENGER.safetensors OUT [RUNS_PER_TASK=15] [WORKERS=1]
# Needs: .venv-orch's vector-orchestrator on PATH (or ORCH=path/to/it), POLICY_PYTHON (the training
# env's python, which serves both policies), ROBOTWIN_BENCH_ROOT and ROBOTWIN_BENCH_PYTHON (the
# RoboTwin-Vector checkout and its conda env's python). Running it again resumes the duel.
set -euo pipefail
king=$(realpath "$1"); chal=$(realpath "$2"); out=$(realpath -m "$3")
runs=${4:-15}; workers=${5:-1}
orch=${ORCH:-vector-orchestrator}
: "${POLICY_PYTHON:?set POLICY_PYTHON to the training env python}"
: "${ROBOTWIN_BENCH_ROOT:?set ROBOTWIN_BENCH_ROOT to the RoboTwin-Vector checkout}"
: "${ROBOTWIN_BENCH_PYTHON:?set ROBOTWIN_BENCH_PYTHON to its conda env python}"
device=${POLICY_DEVICE:-cuda:0}

mkdir -p "$out/king" "$out/challenger"
policy() {  # DIR WEIGHTS: a policy.yaml serving WEIGHTS
  printf 'api: 1\npolicy: vector_runtime.policy:VectorPolicy\nkwargs: {weights: "%s", device: "%s"}\n' \
    "$2" "$device" > "$1/policy.yaml"
}
policy "$out/king" "$king"
policy "$out/challenger" "$chal"
revision() { sha256sum "$1" | cut -c1-40; }  # a local directory stands for this "commit"
args=(--store "$out/store" --run-dir "$out/runs" --policy-python "$POLICY_PYTHON" --workers "$workers"
      --free-gb 0 --local "local/king=$out/king" --local "local/challenger=$out/challenger")

[ -f "$out/store/manifest.json" ] || "$orch" store init --root "$out/store"
# The first duel in an empty store is a genesis: it crowns the king. One run per task keeps it short.
if [ ! -f "$out/.crowned" ]; then
  "$orch" duel "${args[@]}" --challenger "local/king@$(revision "$king")" --runs-per-task 1
  touch "$out/.crowned"
fi
# A made-up block hash deals scenes the way a real duel's chain entropy does; kept for resumes.
[ -f "$out/seed_hash" ] || python3 -c "import secrets; print('0x' + secrets.token_hex(32))" > "$out/seed_hash"
"$orch" duel "${args[@]}" --challenger "local/challenger@$(revision "$chal")" \
  --runs-per-task "$runs" --seed-block 1 --seed-block-hash "$(cat "$out/seed_hash")" | tee "$out/verdict.json"
