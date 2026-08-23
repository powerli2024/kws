# Selector contract (after review of the v3 scheme)

`cos(candidate, raw)` is a **catastrophe gate**, not a purity score. Official T2
rank is known-wake continuous confidence `q_kw` (or `-NLL`). Heuristic `p_music`
does not enter the official score.

## What we accepted from the review

| Claim | Decision |
|---|---|
| T2 `score=cos(c,raw)` is biased toward original | **Accept.** λ=0 made this almost a no-op. Removed as rank. |
| CER slack 0.05 is useless on 4-char wakes | **Accept.** L1 slack is now **0** (same CER only). SE still uses 0.05 regression slack. |
| Heuristic p_music is not a music detector | **Accept.** Not official L2. YAMNet/PANNs stay surveyed until calibrated. |
| SE export skipped the safety gate | **Accept.** Export refuses SE if it cannot compute `cos(se,pre)`. |
| Need reject-enroll when two high-text tracks are different speakers | **Accept.** `chosen=reject` when `q_kw` high on ≥2 sep tracks and pair-cos is low. |
| E0 / E1 / E2 / Oracle first | **Accept as groups.** Oracle = offline `max cos(enroll,cmd)` — not deployable. |
| CMD eval needs lang split + CI | **Accept.** Wilson on FRR/FAR; 474 neg ≈ 0.211 pp per error. |
| TF-Locoformer / DeepFilterNet3 / train a new sep | **Defer to extract@sep bake-off (E3).** kws still does not run MossFormer or train. |
| Forced-decode NLL | **Contract landed** (`--qkw-jsonl`). The ASR dump is not in this repo yet. |

## What we rejected or narrowed

- **Do not add a second separator inside kws.** Extra candidates belong on extract `sep`. If Oracle does not beat E1, *then* change candidates — the review’s stop-loss.
- **Do not treat datasetA `id` as `speaker_id`.** On the copy we have, neg `id` is already unique (474/474). The reported `id=364 ×87` is **not** in `d:\media\datasetA`. Still: no speaker/session/device labels.
- **Do not ship T2 conclusions** until a real `q_kw` sidecar exists. Cos-only T2 is now an explicit degrade.
- **Window veto** uses ≥0.8 s windows and **does not reject** if speech &lt; 1.2 s. 0.6 s ERes min-cos is not assumed reliable.

## Arms vs bake-off

| Bake-off | kws group | Selector |
|---|---|---|
| E0 | `e0_raw` | raw datasetA KWS |
| E1 | `e1_t0` | CER oracle (current production) |
| E2 | `e2_qkw` | same-CER + `q_kw` + catastrophe | 
| E3 | extract@sep extra model | not implemented here |
| E4 / E6 | `t1_spectral` / `t4_spectral` | SE only with encoder safety |
| Oracle | `oracle_cmd` | labeled CMD cosine — upper bound |

T0–T4 remain the one-factor ablation. T4 still ignores `q_kw`.
