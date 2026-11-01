#!/usr/bin/env bash
# Download dataset/<task>/<collection>.zip from TianxingChen/RoboTwin2.0 for the 16 Vector tasks at
# one pinned revision, check each archive against the Hub's sha256, extract it beside itself (only
# if every path in it is safe) and delete the archive.
#   fetch_data.sh DEST [COLLECTION=franka_randomized_500] [PARALLEL=4]
# COLLECTION is franka_randomized_500 (63.6 GB) or franka_clean_50 (3.6 GB).
set -euo pipefail
dest=$(realpath -m "$1")
coll=${2:-franka_randomized_500}
par=${3:-4}
rev=3dc3b798668feb99ac61cc9086d84cbcc3d79186
hub=https://huggingface.co
tasks="beat_block_hammer move_stapler_pad place_bread_skillet place_burger_fries place_can_basket
place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan
place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand place_shoe"
mkdir -p "$dest"

fetch() {
  local task=$1 dir="$dest/$1" zip="$dest/$1/$coll.zip" want got
  if [ -d "$dir/$coll/data" ]; then echo "$task: already there"; return 0; fi
  mkdir -p "$dir"
  want=$(curl -sf "$hub/api/datasets/TianxingChen/RoboTwin2.0/tree/$rev/dataset/$task" | python3 -c \
    "import json,sys; print(next(e['lfs']['oid'] for e in json.load(sys.stdin) if e['path'].endswith('/$coll.zip')))")
  if [ ! -f "$zip" ] || [ "$(sha256sum "$zip" | cut -d' ' -f1)" != "$want" ]; then
    curl -L --fail --retry 5 --retry-delay 5 -C - -sS -o "$zip" \
      "$hub/datasets/TianxingChen/RoboTwin2.0/resolve/$rev/dataset/$task/$coll.zip"
  fi
  got=$(sha256sum "$zip" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then echo "$task: SHA256 MISMATCH ($got, want $want)"; return 1; fi
  python3 - "$zip" <<'EOF'
import sys, zipfile
from pathlib import Path
z = Path(sys.argv[1])
with zipfile.ZipFile(z) as zf:
    bad = [n for n in zf.namelist() if n.startswith(("/", "\\")) or ".." in Path(n).parts or ":" in n]
    if bad:
        sys.exit(f"{z}: unsafe paths {bad[:3]}")
    zf.extractall(z.parent)
EOF
  rm -f "$zip"
  echo "$task: ok"
}
export -f fetch
export dest coll rev hub
printf '%s\n' $tasks | xargs -P "$par" -I{} bash -c 'fetch {}'
