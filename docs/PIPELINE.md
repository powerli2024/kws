# Pipeline: extract@sep then kws

kws does **not** run MossFormer. Redo all BSS first in
[powerli2024/extract](https://github.com/powerli2024/extract) branch **`sep`**.

```text
datasetA/{pos,neg}.jsonl + kws_*.wav
        │
        ▼
extract@sep   ./run_sep.sh
        │     s1–s8 Qwen CER oracle, no MMS-FA, no duration skip-sep
        ▼
$VM_OUT/kws_handoff.json
$VM_OUT/{pos,neg}/s*/index.jsonl
$VM_OUT/best_sep/index.jsonl + {pos,neg}/{uid}.wav
        │
        ▼
kws   rebuild_best_sep → build_eres_sidecar → T0–T4 picks
        → export several best_sep_groups → eval_cmd_cosine (local rank)
        → (later, extract@main) Presence veto on those groups
```

Contest Presence / mix ASR is a **separate clone**: `/root/extract` on **`main`**. This BSS clone is `/root/extract-sep`. Do not `git checkout` between them. They share `/root/autodl-tmp` and conda env `ve`.

## AutoDL

```bash
cd /root
git clone -b sep https://github.com/powerli2024/extract.git extract-sep
cd /root/extract-sep && chmod +x *.sh run_sep.sh pick_python.sh
export DATA_DIR=/root/autodl-tmp/datasetA
export VM_OUT=/root/autodl-tmp/kws_sep
export MOSS_CKPT_DIR=/root/autodl-tmp/checkpoints
export ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B
./setup_env.sh          # 装进 conda env ve（与 /root/extract 同一 PYTHON_BIN）
source ./env.sh         # 或: conda activate ve && source ve/.env_ve
./download_models.sh
./check_env.sh
./run_sep.sh --limit 20
./run_sep.sh
# enroll for main VE:
mkdir -p /root/autodl-tmp/pos_neg
ln -sfn /root/autodl-tmp/kws_sep/best_sep /root/autodl-tmp/pos_neg/best_sep
```

Then here. First compare every stage/threshold arm; `rebuild_best_sep.py` is
only a downstream rehydration of the already selected `best_sep`, not the
stage-comparison command:

```bash
python scripts/compare_all_stages.py \
  --pos-neg /root/autodl-tmp/kws_sep \
  --expected-uids 1838
# Slow confirmation when thr_a/thr_b appear duplicated:
python scripts/compare_all_stages.py \
  --pos-neg /root/autodl-tmp/kws_sep \
  --expected-uids 1838 --hash-wav

# Only after reviewing reports/all_stage_comparison.{json,md}:
python scripts/rebuild_best_sep.py --pos-neg /root/autodl-tmp/kws_sep
python scripts/analyze_dual_zero.py --pos-neg /root/autodl-tmp/kws_sep
python scripts/run_kws_eval.py \
  --pos-neg /root/autodl-tmp/kws_sep \
  --data-dir /root/autodl-tmp/datasetA \
  --backend eres2netv2 \
  --out-root /root/autodl-tmp/kws_sep/best_sep_groups
```

For formal comparison, export each selected stage/thr as an **independent
route**. Do not use `best_sep`'s per-UID across-stage oracle as an experimental
arm. A gated route uses its declared parent only for UIDs outside that gate:

```bash
python scripts/export_stage_routes.py \
  --pos-neg /root/autodl-tmp/kws_sep \
  --out-root /root/autodl-tmp/kws_sep/stage_routes \
  --route s1_onnx_full \
  --route s2_cv_full \
  --route s5_onnx_then_cv_gate/thr_a
```

Each output is a standalone VE enrollment root:
`stage_routes/{stage_or_stage__thr}/index.jsonl` plus `pos/` and `neg/` WAVs.
Its index also retains the selected source row's complete `streams` field for
later q_kw/FA experiments.

To inspect the best-performing real audio for the same UID across all s1--s8
streams, with copied WAVs deduplicated by SHA-256:

```bash
python scripts/rank_same_uid_audio.py \
  --pos-neg /root/autodl-tmp/kws_sep \
  --expected-uids 1838 --top-k 20
```

This writes `reports/same_uid_audio_rank.jsonl` (per-UID audio ranking) and
`reports/same_uid_audio_rank.{json,md}` (fixed-route and stage win rankings).
If a copied byte-identical WAV has different CER in different stage indexes,
the default robust policy uses their median and records every reference in
`reports/same_uid_audio_score_conflicts.jsonl`. Use
`--score-conflict-policy fail` only when auditing ASR reproducibility.
The per-UID cross-stage winner is an offline CER ceiling only; use
`export_stage_routes.py` for deployable independent routes.

Sidecar: raw KWS is `$DATA_DIR/{kws_rel}` (never the AutoDL `kws_path`). Original BSS stream is `{uid}_peak.wav`.
T2 without `reports/sidecars/cos_to_raw.jsonl` is not an L2 result.

Handoff contract: extract `KWS_HANDOFF.md`. `mms_fa` must be false.
Selector: within-stage min CER, original wins ties; across-stage min CER, prefer non-original on ties.
