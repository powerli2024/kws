# Metric availability

| Metric | Role | When | Not |
|---|---|---|---|
| `cos(e*, e_raw)` | **Catastrophe** | Always computable | Not “purer”. Threshold from data, grid 0.90–0.95 |
| `cos(e*, e_cmd_crop)` | Train/dev check only | Offline, pos CMD with a target window | Test-time unknown pos/neg; CMD still dirty |
| Frozen Presence FRR/FAR | **Veto / adopt** | Eval subset, freeze CMD + τ + encoder | Online enroll metric |
| Hard-neg `cos(e*, e_other)` | Optional observation | Only if different uid ⇒ different speaker | datasetA does not guarantee this |
| `p_music` | Residual trigger | After calibration | Speaker quality |
| DNSMOS BAK | Residual trigger / SE rank | After calibration | Speaker quality |
| Window min-cos | Short burst residue | dur ≥ 0.8 s, ≥2×0.6 s windows | Skipped on short KWS; percentile not absolute cos |
| DNSMOS OVRL / NISQA | Artifact / quality log | Cautious | Not L2 objective |
| High-frequency ratio / nonstationarity | Artifact log | Cautious | |
| SI-SDR | Synthetic overlay only | MUSAN diagnostic | Real KWS, no dry ref |
| CER | **Constraint** | mean ≤ 0.03, CER=0 drop ≤ 2 pp | Unique argmin when 94% are already 0 |

## Catastrophe calibration

1. Collect `cos(sep, raw)` on **orig-unique-zero** items (BSS failed on text; embedding should often drop).
2. Collect `cos(winner, raw)` on original winners (≈1 if winner is original).
3. Choose grid value near the 5th percentile of a “must not fire” slice; confirm FPR < 10% there.
4. Until that slice exists, default τ=0.92 is a **placeholder**, kind=hypothesis.

## Window min-cos landing

- `window_starts`: skip if `dur < 0.8s`; else 0.6 s windows, 0.3 s hop, always include the tail window.
- Embed each window with **eres** (FFT in `calibrate_window_mincos.py` is plumbing only).
- `min` of pairwise cosine.
- Flag if min-cos < Pτ of the **same-corpus** distribution, τ ∈ {5,10,15,20}.
- Lock τ on 100 listen labels (obvious music/other-speaker burst).
- Short KWS: **do not impute**; leave the metric missing.

## Presence dataset requirements

- Same pos/neg CMD wavs as the locked mix run.
- Frozen `configs` τ: zh 0.29305, en 0.357868, encoder eres2netv2.
- Change **only enroll wav**.
- Do not `FORCE_CALIB` on `ve_mix_novad`.
- Report FRR, contest RR (neg reject), FAR=1−RR, CER constraint.
