# kws — speaker-first KWS enrollment purification

Repo for proposal v2: CER is a **constraint**, not the enroll objective. Selector is **oracle CER** (original wins ties). **No MMS-FA.**

Evidence on the current `best_sep` (n=1838):

- Original winners: **1162**
- Among them, original CER=0 **and** at least one sep track CER=0: **1045 (89.9%)** — kind=**evidence**
- Original uniquely CER=0 (sep all worse): **89 (7.7%)** — kind=**evidence**
- Recomputed oracle vs existing best_sep: **0 mismatches**

So “63% original” is mostly a **tie-break**, not proof that BSS is useless. **Skip-sep is a small branch. The bet is L2 on the dual-zero set.**

## Do first / do not

| Do first | Do not |
|---|---|
| Dual-zero stats (done) | Train a new separator |
| Enriched per-track index (done) | MMS-FA enroll pick |
| T2: eres `cos(track, raw)` on dual-zero | Treat `cos(*, raw)` as purity |
| T1: conditional SE on dirty original | Global SE (T4) as default |
| Frozen Presence veto | `FORCE_CALIB` on locked mix VE |
| | Duration skip-sep (`dur≤1.8s`) |

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

GPU BSS re-run (optional, AutoDL extract/VM, still CER oracle): `bash scripts/rerun_sep.sh`

T2 needs a sidecar jsonl: each row `{"uid": "...", "cos_to_raw": {"original": 0.9, "spk1": 0.8, "spk2": 0.1}}` (or `scores` / `cos`). Empty dicts and whole-row fallback are rejected. If the file is passed, every uid must be present; a partial file is an error, not a mixed T0/L2 run. Without `--cos-jsonl`, T2 degrades to T0 and records `n_l2_degraded_no_cos`. Hard-fail with `--strict-cos`.

## Adopt rule

Change default enroll only if frozen Presence FRR or FAR improves and the other does not worsen, and CER stays inside the constraint. Details: `docs/EXPERIMENT_MATRIX.md`.
