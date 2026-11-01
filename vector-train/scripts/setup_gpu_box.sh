#!/usr/bin/env bash
# The two Python environments on a Linux GPU box (run from the directory holding vector-train/):
#   .venv-train  vector-runtime's pinned model stack (torch 2.8.0, CUDA 12.8 wheels: RTX 50xx too)
#                plus vector-protocol and this package. It trains, and serves policies in duels.
#   .venv-orch   the duel engine (`vector-orchestrator`), as its README sets it up.
# The simulator is separate: RoboTwin-Vector's own install script and conda env.
#   bash vector-train/scripts/setup_gpu_box.sh
set -euo pipefail
PY=${PYTHON:-python3}
[ -d vector-orchestrator ] || git clone https://github.com/robotensor/vector-orchestrator
[ -d vector-train ] || { echo "copy vector-train/ here first" >&2; exit 2; }

"$PY" -m venv .venv-train
.venv-train/bin/pip install -q --upgrade pip
.venv-train/bin/pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.venv-train/bin/pip install -q -e "vector-orchestrator/packages/vector-runtime[model]" \
    -e vector-orchestrator/packages/vector-protocol -e vector-train
.venv-train/bin/python -c "import torch; assert torch.cuda.is_available(); print('train env:', torch.__version__, torch.cuda.get_device_name(0))"

"$PY" -m venv .venv-orch
.venv-orch/bin/pip install -q --upgrade pip
.venv-orch/bin/pip install -q -e vector-orchestrator -e vector-orchestrator/packages/vector-protocol \
    -e vector-orchestrator/packages/vector-runtime
.venv-orch/bin/vector-orchestrator --help > /dev/null && echo "orchestrator env: ok"
