# Decisions (evidence / inference / hypothesis)

## Dual-zero (evidence)

From existing stage indexes, **no GPU re-run**:

| Quantity | n | rate |
|---|---|---|
| best_sep | 1838 | — |
| original winners | 1162 | 63.2% |
| original winners with orig CER=0 | 1134 | 97.6% of original winners |
| **dual-zero** (orig CER=0 and ≥1 sep CER=0) | **1045** | **89.9% of original winners** (508 with one sep at 0, 537 with two) |
| orig unique zero (sep all CER>0) | 89 | 7.7% of original winners |
| recomputed CER oracle ≠ best_sep | 0 | 0 |

`oracle_of` prefers original on ties, so those 1045 original enrolls are **tie-break artifacts**. Kind=**evidence**.

## Skip-sep (inference)

Pre-proposal hope: many clean originals → skip BSS. Data: unique-zero is only 89 clips. **Skip-sep after scores is a 7.7% branch, not the main path.** Dual-zero must go to L2, not skip. Kind=**inference** from the table.

Do **not** skip BSS because duration ≤ 1.8 s (old VB; enroll CER got worse). Kind=**evidence** from a prior ONNX run.

Pre-BSS skip with residual models: default **OFF** until listen-100. Kind=**hypothesis**.

## Conflicts with a naive “re-run sep”

1. Current `pos_neg` s1–s8 **already skip MMS-FA** (Qwen CER oracle). A full redo is extract@sep `./run_sep.sh` (same selector, new `kws_handoff.json`), then kws T0–T4 — not revive VB MMS.
2. `cos(e*, e_raw)` is catastrophe, not purity.
3. Presence τ stays locked; no `FORCE_CALIB` on `ve_mix_novad`.
4. T4 global SE is ablation; do not ship it if it merely ties T1.

## Stop-loss

- Oracle mismatch > 0 → stop.
- T2 never differs from T0 because cos sidecar missing → do not claim L2 failed; run `scripts/build_eres_sidecar.py` (wavs already exist).
- T1/T3 `need_se=False` FRR/FAR up → detector false positive.
- Presence FAR up while FRR down → reject.
