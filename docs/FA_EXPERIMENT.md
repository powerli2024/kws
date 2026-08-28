# FA experimental branch (isolated from T0--T4)

This branch tests phonetic routing and registration-text cropping without
changing E1/E2 or locked Presence thresholds.

Arms:

- `F0_t0_full`: T0, full waveform.
- `F1_qkw_full`: Qwen target-NLL/q_kw winner, full waveform.
- `F2_fa_full`: MMS-FA winner by `p10_logp`, then `mean_logp`.
- `F3_agree_full`: switch only when Qwen and MMS-FA agree; otherwise T0.
- `F4_agree_safe_crop`: F3 route plus safe Qwen3-FA/MMS crop; unsafe crop falls back full.

## Alignment sidecar contract

One row per UID; every same-CER stream must exist. MMS route evidence must
include `mean_logp` and `p10_logp`. A crop-only Qwen sidecar may omit them.

```json
{"uid":"pos_1","model":"mms_fa","streams":{"original":{"coverage":1.0,"mean_logp":-0.8,"p10_logp":-1.4,"star_fraction":0.0,"start_sec":0.28,"end_sec":1.22,"duration_sec":1.50,"edge_clipped":false},"spk1":{"coverage":1.0,"mean_logp":-0.3,"p10_logp":-0.5,"star_fraction":0.0,"start_sec":0.31,"end_sec":1.18,"duration_sec":1.50,"edge_clipped":false}}}
```

Run selection first; it does not modify audio:

```bash
# Qwen3-FA produces crop boundaries only (official output has no confidence):
python scripts/score_qwen3_fa.py \
  --model-dir /root/autodl-tmp/Qwen3-ForcedAligner-0.6B \
  --pos-neg /root/autodl-tmp/pos_neg --overwrite

python scripts/run_fa_experiment.py \
  --qkw-jsonl reports/sidecars/q_kw_nll.jsonl \
  --route-fa-jsonl reports/sidecars/mms_fa.jsonl \
  --crop-fa-jsonl reports/sidecars/qwen3_fa.jsonl
```

Generate `mms_fa.jsonl` in a pinned MMS/uroman environment:

```bash
python -m pip install 'torch==2.7.*' 'torchaudio==2.7.*'
python scripts/score_mms_fa.py \
  --uroman-dir /root/autodl-tmp/uroman \
  --pos-neg /root/autodl-tmp/pos_neg --overwrite
```

The output contains real per-token alignment scores. Do not convert Qwen timestamps into fake
`mean_logp`/`p10_logp`: Qwen's official aligner API does not expose confidence.
Torchaudio 2.8 deprecated the forced-alignment API and 2.9 removed it, so record
the torch/torchaudio/model/uroman versions with every MMS sidecar.

Materialize one arm only after inspecting the summary:

```bash
python scripts/export_fa_experiment.py \
  --arm F4_agree_safe_crop \
  --pos-neg d:\media\pos_neg --data-dir d:\media\datasetA
```

Formal runs must not use `--allow-partial`. Acceptance still requires full UID
coverage, frozen ERes2NetV2 CMD/Presence evaluation, language-split FAR/FRR and
paired comparison against E1/E2. The FA score is text-preservation evidence,
not speaker identity evidence.
