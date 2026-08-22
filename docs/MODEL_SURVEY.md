# p_music and DNSMOS BAK survey

Constraint: **denoise ≠ better speaker embedding**. BAK or p_music may rise while `cos(e_se, e_pre)` and Presence FRR/FAR get worse. Every SE accept must pass the SE safety gate and the Presence veto.

## p_music

| Direction | Candidate | Why | Cost | Use |
|---|---|---|---|---|
| **A quality** | PANNs CNN14 (AudioSet) or BEATs | Best music/event tagging among open weights; use as quasi-reference to label the 100 listen set and to check the light model | GPU, ~80M (CNN14) | Validation only; not in the enroll hot path |
| **A quality (alt)** | OpenMIC / dedicated music vs speech classifiers | Fewer AudioSet tag collisions (speech clips often have weak music tags) | similar | If CNN14 over-fires on singing-like wake words |
| **B deploy** | YAMNet (TF-Hub / TFLite) | ~3.8M, CPU realtime, AudioSet music cluster | CPU ms | Actual `need_se` / L2 `λ p_music` once calibrated vs A and vs listen-100 |
| **B fallback** | `kws.residual.p_music_heuristic` (spectral flatness) | No extra weights | CPU µs | Plumbing and smoke tests only until calibrated |

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

## Recheck: denoise ≠ speaker

On each uid: `raw` / `pre` / `se`.

1. `cos(e_se, e_pre)` must stay ≥ τ (grid 0.90–0.95) or revert.
2. CER(se) ≤ CER(pre)+0.05 or revert.
3. Presence FRR/FAR with frozen CMD and frozen τ — only this can adopt.
4. Report BAK and p_music **and** the speaker numbers side by side. If BAK↑ and FRR↑, keep the row as a known disagreement; do not “fix” by dropping Presence.

Synthetic diagnostic (not for model selection alone): overlay MUSAN at SNR 0–15 dB on CER=0 winners; SE should recover `cos(e_se, e_clean)` more than T4 does on already-clean clips.
