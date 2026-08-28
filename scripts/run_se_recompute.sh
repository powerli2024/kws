#!/usr/bin/env bash
set -euo pipefail
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

# Full raw/SE recompute.  Spectral is an executable plumbing/control arm;
# production neural SE should use SE_BACKEND=command and SE_COMMAND below.
REPO_DIR="${REPO_DIR:-/root/kws}"
POS_NEG="${POS_NEG:-/root/autodl-tmp/kws_sep}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/kws_se_route}"
ASR_MODEL_DIR="${ASR_MODEL_DIR:-/root/autodl-tmp/Qwen3-ASR-1.7B}"
S1_ARM="${S1_ARM:-s1_onnx_full}"
S7_ARM="${S7_ARM:-auto}"
SE_BACKEND="${SE_BACKEND:-spectral}"
SE_COMMAND="${SE_COMMAND:-}"
SE_BATCH_COMMAND="${SE_BATCH_COMMAND:-}"
PRECOMPUTED_SE_DIR="${PRECOMPUTED_SE_DIR:-}"
SPEAKER_BACKEND="${SPEAKER_BACKEND:-eres2netv2}"
SPEAKER_MODEL_DIR="${SPEAKER_MODEL_DIR:-}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EXPECTED_UIDS="${EXPECTED_UIDS:-1838}"
WITH_NLL="${WITH_NLL:-1}"
RESUME="${RESUME:-1}"
EXPORT_BEST="${EXPORT_BEST:-1}"
DATA_DIR="${DATA_DIR:-}"
BASELINE_DIR="${BASELINE_DIR:-$WORK_DIR/best_sep_s1_to_s7_raw}"

test -d "$REPO_DIR"
test -d "$POS_NEG/pos"
test -d "$POS_NEG/neg"
test -d "$ASR_MODEL_DIR"
cd "$REPO_DIR"

python -c "import editdistance,numpy,pypinyin,qwen_asr,soundfile,torch,yaml"
python scripts/audit_sep_input.py \
  --pos-neg "$POS_NEG" \
  --expected-uids "$EXPECTED_UIDS"

args=(
  python scripts/run_se_route_eval.py
  --pos-neg "$POS_NEG"
  --work-dir "$WORK_DIR"
  --model-dir "$ASR_MODEL_DIR"
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
  --expected-uids "$EXPECTED_UIDS"
  --s1-arm "$S1_ARM"
  --s7-arm "$S7_ARM"
  --se-backend "$SE_BACKEND"
  --speaker-backend "$SPEAKER_BACKEND"
)

if [[ "$SE_BACKEND" == "command" ]]; then
  if [[ -n "$SE_BATCH_COMMAND" ]]; then
    args+=(--se-batch-command "$SE_BATCH_COMMAND")
  else
    test -n "$SE_COMMAND"
    args+=(--se-command "$SE_COMMAND")
  fi
elif [[ "$SE_BACKEND" == "precomputed" ]]; then
  test -n "$PRECOMPUTED_SE_DIR"
  args+=(--precomputed-se-dir "$PRECOMPUTED_SE_DIR")
fi
if [[ -n "$SPEAKER_MODEL_DIR" ]]; then
  args+=(--speaker-model-dir "$SPEAKER_MODEL_DIR")
fi
if [[ "$WITH_NLL" == "1" ]]; then
  args+=(--with-nll)
fi
if [[ -n "$DATA_DIR" ]]; then
  args+=(--data-dir "$DATA_DIR")
fi
if [[ "$RESUME" == "1" ]]; then
  args+=(--resume)
else
  args+=(--overwrite-scores)
fi
if [[ "$EXPORT_BEST" == "1" ]]; then
  args+=(--export-best)
fi

"${args[@]}"

python - "$WORK_DIR/report.json" "$EXPECTED_UIDS" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
report = json.load(open(path, encoding="utf-8"))
assert report["coverage"]["n_uid"] == expected, report["coverage"]
assert report["arms"]["s1_to_s7_safe_se"]["n_ok"] == expected
assert report["proposed_vs_raw_route"]["n_worsened"] == 0
assert report["production_approved"] is False
print("[PASS] full coverage, paired no-regression and explicit downstream holdout")
PY

if [[ -n "$DATA_DIR" ]]; then
  test -d "$DATA_DIR"
  test -d "$BASELINE_DIR"
  python scripts/eval_cmd_cosine.py \
    --data-dir "$DATA_DIR" \
    --backend "$SPEAKER_BACKEND" \
    --device "$DEVICE" \
    --baseline raw_route \
    --dir "raw_route=$BASELINE_DIR" \
    --dir "safe_se=$WORK_DIR/best_sep_s1_s7_safe_se" \
    --out "$WORK_DIR/cmd_cosine.json" \
    --scores "$WORK_DIR/cmd_cosine_scores.jsonl"
fi

echo "[DONE] $WORK_DIR/report.md"
