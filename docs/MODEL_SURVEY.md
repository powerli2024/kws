# Encoder, p_music, DNSMOS, SE survey

Constraint: **denoise ≠ better speaker embedding**. BAK or p_music may rise while `cos(e_se, e_pre)` and later Presence FRR/FAR get worse.

## Speaker encoder (enroll + CMD cosine)

This is the model T2 sidecar and `eval_cmd_cosine.py` actually run.

| Direction | Candidate | Why | Cost | Use |
|---|---|---|---|---|
| **A = B deploy** | **ERes2NetV2** `iic/speech_eres2netv2_sv_zh-cn_16k-common` | Same family as locked contest Presence; short Chinese SV | GPU ms / clip | **Default.** `scripts/build_eres_sidecar.py --backend eres2netv2` |
| **A quality (alt)** | CAM++ `iic/speech_campplus_sv_zh-cn_16k-common` | Independent SV view; if ERes and CAM++ disagree on dual-zero picks, inspect | similar | `--backend campplus` diagnostic, not default rank |
| **B fallback (plumbing)** | `kws.eres.FftProxyEncoder` | No weights | CPU µs | `--backend fft` smoke only; **must not rank groups** |
| Do not default | WeSpeaker ResNet34-LM / VoxBlink2 | Different embedding space than locked Presence | similar | Only if ERes cannot load; never silent fallback |

**Wired:** `src/kws/eres.py` `load_embedder`. Raw KWS + each BSS stream → `cos_to_raw` sidecar. Enroll vs CMD → EER/AUC/gap.

## p_music

| Direction | Candidate | Why | Cost | Use |
|---|---|---|---|---|
| **A quality** | PANNs CNN14 (AudioSet) or BEATs | Best music/event tagging among open weights; use as quasi-reference to label the 100 listen set and to check the light model | GPU, ~80M (CNN14) | Validation only; not in the enroll hot path |
| **A quality (alt)** | OpenMIC / dedicated music vs speech classifiers | Fewer AudioSet tag collisions (speech clips often have weak music tags) | similar | If CNN14 over-fires on singing-like wake words |
| **B deploy** | YAMNet (TF-Hub / TFLite) | ~3.8M, CPU realtime, AudioSet music cluster | CPU ms | Actual `need_se` / L2 `λ p_music` once calibrated vs A and vs listen-100 |
| **B fallback (wired)** | `kws.residual.p_music_heuristic` (spectral flatness) | No extra weights | CPU µs | `build_eres_sidecar.py` writes `reports/sidecars/p_music.jsonl`. Use for T3/`need_se` plumbing. Do not treat as ground truth until listen-100 |

**Calibration:** 100 listen labels `{0,1}` obvious music/noise residue. Sweep B-threshold so trigger rate is 15–30% and precision vs A/labels is reported. Do not copy 0.40/0.35 from the proposal as frozen.

**Failure mode:** wake singing / tonal noise → high p_music on a clean speaker. Mitigate with `need_se=False` on the listen-clean slice and with the SE cos gate.

## DNSMOS BAK

| Direction | Candidate | Why | Cost | Use |
|---|---|---|---|---|
| **A quality** | DNSMOS P.835 (Microsoft ONNX, SIG/BAK/OVRL) | Trained on DNS P.835; BAK is the right axis for *background*, not overall MOS | small CNN ONNX, CPU | Residual reference; stack-rank SE backends |
| **A alt** | NISQA (MOS + noisiness) | Broader distortions | heavier | Second opinion on SE artifacts |
| **B deploy** | Same DNSMOS P.835 ONNX | Already small enough for offline enroll (KWS is <2 s) | CPU tens of ms | Can enter `need_se` if it beats YAMNet on listen-100 |
| **B lighter** | `kws.residual.snr_med_db` (p80/p20 frame RMS) | No neural net | CPU µs | Co-trigger with p_music; never a speaker score |

Do **not** use PESQ/STOI on real KWS (no reference). SI-SDR only on the synthetic MUSAN overlay diagnostic.

## SE backends

| Direction | Candidate | Why | Use |
|---|---|---|---|
| **B landable (wired)** | `kws.se_backend.spectral_subtract` | Numpy Wiener-like gate; no extra ckpt | `export_best_sep_groups.py --group t1_spectral --se-backend spectral` |
| **A quality** | DeepFilterNet / FRCRN / DTLN | Stronger denoise | Not in this repo until a ckpt is checked in; still must pass `se_safety_ok` |
| **Do not ship** | T4 always-SE | Negative control | `t4_spectral` group exists to lose |

## Recheck: denoise ≠ speaker

On each uid: `raw` / `pre` / `se`.

1. `cos(e_se, e_pre)` must stay ≥ τ (grid 0.90–0.95) or revert.
2. CER(se) ≤ CER(pre)+0.05 or revert.
3. Later: Presence FRR/FAR with frozen CMD and frozen τ on extract@main — contest adopt only. This repo ranks with enroll↔CMD cosine.
4. Report BAK and p_music **and** the speaker numbers side by side. If BAK↑ and FRR↑, keep the row as a known disagreement; do not “fix” by dropping Presence.

Synthetic diagnostic (not for model selection alone): overlay MUSAN at SNR 0–15 dB on CER=0 winners; SE should recover `cos(e_se, e_clean)` more than T4 does on already-clean clips.
