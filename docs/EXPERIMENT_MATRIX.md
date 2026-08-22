# Frozen experiment matrix (proposal §5)

This file is the contract. Code that disagrees is a bug.

## What each arm can claim

| Arm | Select | SE | Question it can answer | Question it cannot answer |
|---|---|---|---|---|
| **T0** | CER oracle, original wins ties | none | Current enroll Presence FRR/FAR / contest | Whether a different track is purer |
| **T1** | T0 | conditional | Does residual-triggered SE help CER-oracle enroll? | Whether the track choice was wrong |
| **T2** | L1 CER slack 0.05 + L2 `cos(track, raw) − λ p_music` | none | On **dual-zero** items, can speaker/residual break the original tie-break without hurting CER? | Whether SE is useful |
| **T3** | T2 | conditional | Combine track change + SE | Which factor moved Presence if both change — run T1/T2 first |
| **T4** | T0 | always | Negative control: global SE should lose | Anything about need_se if T4 wins (then the detector is wrong) |

Implementation: `CER_ORACLE_ARMS = {T0,T1,T4}`, `L2_ARMS = {T2,T3}`. A cosine sidecar that prefers a sep track must move T2 and must not move T4.

Primary keys: **frozen Presence FRR and FAR** (contest RR = 1 − FAR on neg).  
Constraints: mean CER ≤ 0.03; CER=0 rate drop ≤ 2 pp.  
`cos(e*, e_raw)` is catastrophe-only. `p_music` / DNSMOS BAK are residual triggers.

## Triggers and safety gates

**Skip-sep after scores (implemented):** skip using sep wavs only if original uniquely has CER=0. Dual-zero → do not skip; go to L2.

**Skip-BSS before MossFormer:** default **OFF**. May turn on only after residual calibration. **Duration is never a trigger** (old VB `dur≤1.8s` skip raised enroll CER).

**need_se:** `p_music` / SNR grids in `configs/experiment_matrix.yaml`; target trigger rate 15–30%.  
SE safety: drop SE if `cos(se, pre) < τ` (grid 0.90–0.95) or CER rises by > 0.05.

**Window min-cos:** skip if duration < 0.8 s; ≥2 windows of 0.6 s, hop 0.3 s; anomaly = below a corpus percentile (5/10/15/20), locked with 100 listen labels.

## Eval subsets

| Subset | Use |
|---|---|
| all KWS (n=1838) | CER constraint, catastrophe distribution |
| original winners | skip-sep / need_se-on-original |
| dual-zero | T2 must move here or T2 is a no-op |
| orig-unique-zero | text-safe skip-sep rate |
| listen-100 | lock p_music / window percentile / need_se |
| Presence pos/neg CMD | **veto**; freeze τ zh=0.29305 en=0.357868 |
| hard-neg cos | only if different uid ⇒ different speaker is guaranteed; else unused |

## Do first / do not do

**Do first:** extract@sep `./run_sep.sh` (all BSS) → dual-zero stats → enriched best_sep (per-track CER, no MMS) → T2 on dual-zero with eres cos → T1 conditional SE on original winners with high p_music → Presence veto.

BSS code lives only in [extract `sep`](https://github.com/powerli2024/extract/tree/sep). See `docs/PIPELINE.md`.

**Do not:** train a new separator; MMS-FA enroll pick; test-time `cos(e, e_cmd_crop)`; SI-SDR on real KWS as a main metric; BAK↑ as speaker success; `FORCE_CALIB` on locked mix VE.

## Stop-loss

- Recomputed CER oracle ≠ `best_sep` oracle → stop, indexes drifted.
- T2 never differs from T0 → do not run T3; L2 has no gradient (missing cos sidecar).
- T1/T3 `need_se=False` subset FRR or FAR up → detector false-positive, do not ship.
- T4 beats T1 → do not ship T4; retune need_se.
- Presence FAR up while FRR down → reject (iso-FAR not held).
