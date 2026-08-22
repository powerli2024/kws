# Code review checklist

Run: `python scripts/review_checklist.py`

| Check | Design requirement | Code |
|---|---|---|
| Selector | oracle CER, original wins ties | `kws.oracle.oracle_of`; rebuild mismatch count must be 0 |
| Not MMS-FA | skip MMS-FA path | no `MmsFa` / `mms_fa_scorer`; enriched field `mms_fa: false` |
| Candidates | original / spk1 / spk2 (+ cascade r1/r2) | `is_sep_stream`; stage indexes already store these |
| Cascade input | parent wavs, not raw KWS | documented; GPU re-run delegates to extract s3/s4 |
| Skip-sep | unique-zero after scores; no duration | `skip_sep_after_scores`; `skip_bss_before_sep(enabled=False)` |
| L1/L2 | CER slack 0.05; cos−λ p_music | `select_l1_l2` |
| Catastrophe | 0.90–0.95 grid | `catastrophe.py` / SE safety |
| Window min-cos | skip <0.8 s; 0.6 / 0.3 | `window_mincos.py` |
| T4 | ablation | `run_t0_t4.py` arm T4 |
| Presence | veto only | `presence_protocol.enroll_go` |
| Dual-zero stats | required data support | `reports/dual_zero.json` |

Recomputed CER oracle vs `best_sep`: **oracle_mismatch=0** (evidence).
