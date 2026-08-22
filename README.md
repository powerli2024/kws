# kws — speaker-first KWS enrollment purification

Repo for proposal v2: CER is a **constraint**, not the enroll objective. Selector is **oracle CER** (original wins ties). **No MMS-FA.**

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
| Enriched per-track index | Duration skip-sep (`dur≤1.8s`) |
| T2: eres `cos(track, raw)` on dual-zero | Treat `cos(*, raw)` as purity |
| T1: conditional SE on dirty original | Global SE (T4) as default |
| Frozen Presence veto | `FORCE_CALIB` on locked mix VE |

## Layout

```text
configs/experiment_matrix.yaml
src/kws/          oracle, skip-sep, L2, catastrophe, window min-cos, need_se
scripts/          analyze_dual_zero, rebuild_best_sep, run_t0_t4, review_checklist
docs/             frozen matrix, metrics, survey, review
reports/          dual_zero.json, best_sep_enriched.jsonl
```

## Commands

```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/review_checklist.py
python scripts/analyze_dual_zero.py --pos-neg /path/to/pos_neg
python scripts/rebuild_best_sep.py --pos-neg /path/to/pos_neg
python scripts/run_t0_t4.py
```

GPU BSS (required before claiming a new enroll dump): clone extract **`sep`**, then
`bash scripts/rerun_sep.sh` (wraps `/root/extract/run_sep.sh`).
Do not use extract `main` `./ve.sh` for this.

T2 needs a sidecar jsonl: each row `{"uid": "...", "cos_to_raw": {"original": 0.9, "spk1": 0.8, "spk2": 0.1}}` (or `scores` / `cos`). Empty dicts and whole-row fallback are rejected. If the file is passed, every uid must be present; a partial file is an error, not a mixed T0/L2 run. Without `--cos-jsonl`, T2 degrades to T0 and records `n_l2_degraded_no_cos`. Hard-fail with `--strict-cos`.

## Adopt rule

Change default enroll only if frozen Presence FRR or FAR improves and the other does not worsen, and CER stays inside the constraint. Details: `docs/EXPERIMENT_MATRIX.md`.
