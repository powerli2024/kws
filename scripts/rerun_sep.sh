#!/usr/bin/env bash
# Optional GPU re-run of BSS + oracle-CER scoring. No MMS-FA.
# This repo does not vendor MossFormer weights. Use extract/VM on AutoDL.
set -euo pipefail
: "${EXTRACT_ROOT:=/root/extract}"
: "${VM_OUT:=/root/autodl-tmp/kws_rerun}"
: "${SPLITS:=pos,neg}"
# Skip gated s5–s8 by default: they are salvage, not the frozen matrix.
STAGES="${STAGES:-collect,s1,s2,s3,s4,compare,eval}"
cd "$EXTRACT_ROOT"
chmod +x run_all.sh run_stage.sh || true
./run_all.sh --stages "$STAGES" --splits "$SPLITS" --vm-out "$VM_OUT"
echo "Then: python scripts/rebuild_best_sep.py --pos-neg \$VM_OUT"
