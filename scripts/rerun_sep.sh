#!/usr/bin/env bash
# GPU re-run of ALL KWS BSS stages. This repo has no MossFormer code.
# Use extract branch sep: https://github.com/powerli2024/extract
set -euo pipefail
: "${EXTRACT_ROOT:=/root/extract}"
: "${EXTRACT_BRANCH:=sep}"
: "${VM_OUT:=/root/autodl-tmp/kws_sep}"
: "${SPLITS:=pos,neg}"

if [[ ! -d "$EXTRACT_ROOT" ]]; then
  echo "[ERR] missing $EXTRACT_ROOT" >&2
  echo "  git clone -b $EXTRACT_BRANCH https://github.com/powerli2024/extract.git $EXTRACT_ROOT" >&2
  exit 1
fi
if [[ ! -f "$EXTRACT_ROOT/.sep-only" ]]; then
  echo "[ERR] $EXTRACT_ROOT is not extract@$EXTRACT_BRANCH (no .sep-only)." >&2
  echo "  cd $EXTRACT_ROOT && git fetch origin && git checkout $EXTRACT_BRANCH" >&2
  exit 1
fi
if [[ ! -x "$EXTRACT_ROOT/run_sep.sh" && ! -f "$EXTRACT_ROOT/run_sep.sh" ]]; then
  echo "[ERR] missing $EXTRACT_ROOT/run_sep.sh" >&2
  exit 1
fi
cd "$EXTRACT_ROOT"
chmod +x run_sep.sh run_all.sh run_stage.sh || true
export VM_OUT SPLITS
echo "[INFO] extract@$EXTRACT_BRANCH  VM_OUT=$VM_OUT"
./run_sep.sh --splits "$SPLITS" --vm-out "$VM_OUT" "$@"
echo "[INFO] next: python scripts/rebuild_best_sep.py --pos-neg $VM_OUT"
