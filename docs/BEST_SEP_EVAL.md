# Evaluating which `best_sep` is cleaner

Enroll quality is **not** “lower mean oracle CER”. On the current dump ~94% of
winners are already CER=0, and 89.9% of original winners are **dual-zero**.
Ranking by CER just restates the tie-break.

## Two layers (do not mix)

| Layer | Where | Question | Adopt? |
|---|---|---|---|
| **KWS-local (this repo)** | `scripts/eval_cmd_cosine.py` | Does `cos(embed(enroll), embed(CMD))` separate pos CMD from neg CMD more cleanly? | Rank groups; pick which trees to send downstream |
| **Contest veto (later)** | extract `main` `eval_enroll_swap.py` | Frozen Presence FRR/FAR with locked τ | Only then change the contest enroll default |

This branch **does not** run the mix Presence gate. It **writes several `best_sep` trees** and ranks them with CMD cosine.

Selector contract (v3): `cos(track, raw)` is **catastrophe only**. T2 ranks by `q_kw` (known-wake continuous confidence). See `docs/SELECTOR.md`. Without `--qkw-jsonl`, T2 degrades to T0 and is **not** a speaker experiment.

## Why the ERes sidecar was empty

The wavs were always there:

- raw KWS: `datasetA/{pos,neg}/kws_*.wav` via `kws_rel` (ignore AutoDL `kws_path=/root/...`)
- BSS streams: `pos_neg/{split}/{stage}/wav/{uid}_{tag}.wav` with **`original` → `{uid}_peak.wav`**

T2 only *reads* a jsonl. Nothing embedded tracks until `scripts/build_eres_sidecar.py`.

`cos(track, raw)` is **catastrophe** (collapse vs the original KWS), not purity. CMD cosine is the local quality metric.

## Commands

```bash
# 1) fill sidecar (needs modelscope ERes2NetV2; --backend fft is plumbing only)
python scripts/build_eres_sidecar.py \
  --data-dir d:\media\datasetA --pos-neg d:\media\pos_neg \
  --backend eres2netv2

# 2) T0–T4 picks (T2 needs the sidecar)
python scripts/run_t0_t4.py \
  --cos-jsonl reports/sidecars/cos_to_raw.jsonl \
  --pmusic-jsonl reports/sidecars/p_music.jsonl \
  --strict-cos

# 3) materialize groups
python scripts/export_best_sep_groups.py \
  --out-root d:\media\pos_neg\best_sep_groups

# 4) rank by enroll↔CMD cosine (pos vs neg)
python scripts/eval_cmd_cosine.py \
  --dir t0=d:\media\pos_neg\best_sep_groups\t0 \
  --dir t2=d:\media\pos_neg\best_sep_groups\t2 \
  --dir raw_kws=d:\media\pos_neg\best_sep_groups\raw_kws \
  --baseline t0 --data-dir d:\media\datasetA

# or the whole local pipeline:
python scripts/run_kws_eval.py --backend eres2netv2
python scripts/run_kws_eval.py --backend fft --limit 20   # plumbing smoke
```

Groups written under `--out-root`:

| name | enroll |
|---|---|
| `raw_kws` | always datasetA KWS (no BSS) |
| `t0` | CER oracle, original wins ties (current selector) |
| `e2_qkw` / `t2` | same-CER + `q_kw`; `cos(track, raw)` catastrophe gate only |
| `skip_then_t0` | orig-unique-zero → original; else T0 |
| `skip_then_t2` | orig-unique-zero → original; else T2 |
| `t1_spectral` / `t4_spectral` | optional; `--with-se-groups` (spectral SE, not neural) |

## CMD cosine metrics

For each group, each uid: `score = cos(ERes(enroll), ERes(cmd))`.

- pos scores should be **higher** (same-session speaker)
- neg scores should be **lower** (reject)
- report **mean_gap**, **AUC**, **EER** (+ threshold)
- locked VE τ (zh 0.29305 / en 0.357868) is a **probe** (`locked_tau_probe_not_adopt`), not an adopt rule

Primary rank: locked-τ FAR then FRR **by language**, then pos P10 / neg P90, with Wilson intervals. EER/AUC are secondary. 474 neg ≈ 0.211 pp per error. CER mean ≤ 0.03 is still a constraint.

## Later (not this branch)

```bash
cd /root/extract/ve
python scripts/eval_enroll_swap.py \
  --dir t0=/root/autodl-tmp/pos_neg/best_sep_groups/t0 \
  --dir t2=/root/autodl-tmp/pos_neg/best_sep_groups/t2 \
  --baseline t0 --data-dir /root/autodl-tmp/datasetA
```

Do not `FORCE_CALIB` on locked mix VE.

## Diagnostic, not ranking

| Signal | Use |
|---|---|
| `oracle_cer` / CER=0 rate | Constraint |
| `cos(enroll, raw)` | Catastrophe only |
| `snr_med_db`, `p_music` heuristic | Residual trigger |
| wav fingerprint disagreement | Did groups actually differ? |
| DNSMOS BAK | Residual; denoise ≠ speaker |
| SI-SDR / PESQ | Only with a dry reference |
