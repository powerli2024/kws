# Code review

`python scripts/review_checklist.py` is **not** a string grep of “T4 exists”. It:

1. Loads `configs/experiment_matrix.yaml` and rejects internal contradictions (`runtime.*` vs grids/defaults, T4 yaml vs arm sets).
2. Compares Python defaults to `runtime:`.
3. **Behavioral probe:** a cosine sidecar that prefers `spk1` must change T2 and must **not** change T4.
4. Sidecar parser must reject empty `scores` and whole-row fallback.
5. MMS-FA symbol scan is only a weak complement.
6. `docs/PIPELINE.md` and `scripts/rerun_sep.sh` must name extract branch `sep`.

It still **cannot** prove DeepFilterNet ran or Presence FRR/FAR was measured. A green checklist is not an adopt decision.
