# kws — speaker-first KWS enrollment purification

Repo for proposal v2: CER is a **constraint**, not the enroll objective. E1 is
CER oracle (original wins ties); E2 ranks same-CER candidates by calibrated
`q_kw`. MMS-FA remains excluded from the frozen T0--T4 production selector;
an isolated, fail-closed research branch is documented in
[`docs/FA_EXPERIMENT.md`](docs/FA_EXPERIMENT.md).

## Core metric

Per UID, the selector is lexicographic: **minimum-CER eligible set first, then
maximum `q_kw`** computed from length-normalized forced-decode token NLL. For a
whole `best_sep` group, enroll↔CMD cosine is an offline screen; frozen Presence
and the real contest score are the final adoption veto. Exact formulas, tie
rules, calibration and forbidden metrics are defined in
[`docs/CORE_METRIC.md`](docs/CORE_METRIC.md).

**This repo does not run MossFormer.** Redo all BSS first on
[powerli2024/extract](https://github.com/powerli2024/extract) branch **`sep`**
(`./run_sep.sh` → `$VM_OUT/kws_handoff.json`). Then point `--pos-neg` here at that tree.
Details: `docs/PIPELINE.md`.

Evidence on the current `best_sep` (n=1838):

- Original winners: **1162**
- Among them, original CER=0 **and** at least one sep track CER=0: **1045 (89.9%)** — kind=**evidence**
- Original uniquely CER=0 (sep all worse): **89 (7.7%)** — kind=**evidence**
- Recomputed oracle vs existing best_sep: **0 mismatches**

So “63% original” is mostly a **tie-break**, not proof that BSS is useless. **Skip-sep is a small branch. The bet is L2 on the dual-zero set.**

## Do first / do not

| Do first | Do not |
|---|---|
| Redo BSS on extract@sep (`./run_sep.sh`) | Invent a second separator in this repo |
| Dual-zero stats (done on current dump) | MMS-FA enroll pick |
| **Fill ERes sidecar from existing wavs** | Wait for a missing-wav myth (`kws_path=/root/...` is a red herring) |
| Export **several** `best_sep_*` groups | Rank groups by mean oracle CER |
| Rank groups with **enroll↔CMD cosine** (pos vs neg) | Treat `cos(*, raw)` as purity |
| T2 = `q_kw` + catastrophe; skip-sep only orig-unique-zero | Rank by `cos(*, raw)` or heuristic p_music |
| Later: frozen Presence on extract@main using those groups | Wire Presence / `FORCE_CALIB` in this repo |

## Layout

```text
configs/experiment_matrix.yaml
src/kws/          oracle, L2, sidecar, wav_paths, eres, cmd_eval, export_groups
scripts/          build_eres_sidecar, run_t0_t4, export_best_sep_groups, eval_cmd_cosine, run_kws_eval
docs/             BEST_SEP_EVAL.md is the local eval contract
reports/          sidecars/, t0_t4_picks.jsonl, eval/cmd_cosine.json
```

## Commands

```bash
pip install -r requirements.txt
# real encoder (same family as contest Presence):
#   python -m pip install -U modelscope torch soundfile

python -m pytest -q
python scripts/review_checklist.py
python scripts/rebuild_best_sep.py --pos-neg d:\media\pos_neg --allow-legacy

# Full E2 eval. q_kw must be a calibrated [0,1] known-wake confidence sidecar.
# NLL may rank tracks, but cannot activate the absolute-confidence reject gate.
python scripts/run_kws_eval.py --backend eres2netv2 \
  --data-dir d:\media\datasetA --pos-neg d:\media\pos_neg \
  --qkw-jsonl d:\media\q_kw_forced_decode.jsonl --require-e2

# Build the raw forced-decode NLL sidecar first (ranking only, not calibrated q_kw):
python scripts/score_qkw_nll.py \
  --pos-neg d:\media\pos_neg \
  --model-dir d:\models\Qwen3-ASR-1.7B \
  --limit 20 \
  --out reports/sidecars/q_kw_nll_smoke.jsonl \
  --meta reports/sidecars/q_kw_nll_smoke_meta.json \
  --overwrite
# Full run uses a separate output/signature and can resume safely.
python scripts/score_qkw_nll.py \
  --pos-neg d:\media\pos_neg \
  --model-dir d:\models\Qwen3-ASR-1.7B \
  --overwrite
# Interrupted full run: repeat the same command with --resume instead.

# Experimental registration-text crop boundaries (does not alter T0--T4):
python scripts/score_qwen3_fa.py \
  --model-dir /root/autodl-tmp/Qwen3-ForcedAligner-0.6B \
  --pos-neg /root/autodl-tmp/pos_neg --overwrite
# Experimental MMS phonetic route evidence, in a separate torch 2.7 env:
python scripts/score_mms_fa.py \
  --uroman-dir /root/autodl-tmp/uroman \
  --pos-neg /root/autodl-tmp/pos_neg --overwrite

# Plumbing only: FFT does not produce a frozen-threshold rank.
python scripts/run_kws_eval.py --backend fft --limit 20
```

GPU BSS (required before claiming a **new** enroll dump): clone extract **`sep`**, then
`bash scripts/rerun_sep.sh`. Do not use extract `main` `./ve.sh` for BSS.

`run_kws_eval.py` rebuilds `reports/best_sep_enriched.jsonl` by default, so an
AutoDL BSS rerun cannot silently reuse the report committed with this repo.
Use `--reuse-enriched` only when the `pos_neg` tree is known identical.

T2 text sidecar: pass `--qkw-jsonl` with exactly one of `q_kw` (calibrated
`[0,1]` confidence) or `nll` per UID. `nll` is negated for ranking only;
the two-high-text reject gate requires calibrated `q_kw`.

ERes sidecar: `scripts/build_eres_sidecar.py` writes
`{"uid": "...", "cos_to_raw": {"original": ..., "spk1": ..., "spk2": ...}}`.
Empty dicts and whole-row fallback are rejected. If the file is passed, every
uid must be present. `--strict-text` hard-fails when it is missing.

SE-labelled groups are intentionally disabled until a frozen post-SE ASR safety
score is wired. The repository will not export a copied baseline as an SE result.

## Rank vs later veto

**This branch:** several `pos_neg/best_sep_groups/{raw_kws,t0,t2,...}` + EER/AUC/gap
on `cos(enroll, CMD)`. See `docs/BEST_SEP_EVAL.md`.

**Later, not here:** extract `main` frozen Presence (τ zh 0.29305 / en 0.357868),
swap only enroll. Do not `FORCE_CALIB`.
