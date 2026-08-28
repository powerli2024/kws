#!/usr/bin/env python3
"""Review that can fail if the experiment contract is broken.

Substring search alone is not enough (that is the false-security failure mode).
This script imports the arm sets, validates YAML vs code defaults, and runs a
T4-vs-L2 behavioral probe plus sidecar rejection cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.arms import CER_ORACLE_ARMS, GLOBAL_SE_ARMS, L2_ARMS, se_mode, select_mode  # noqa: E402
from kws.config import MatrixError, matrix, validate_code_defaults  # noqa: E402
from kws.sidecar import SidecarError, parse_cos_row  # noqa: E402
from kws.t0_t4 import pick_track  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def ok(name: str, cond: bool, detail: str) -> None:
    CHECKS.append((name, cond, detail))


def main() -> int:
    try:
        matrix()
        yaml_ok, yaml_detail = True, "experiment_matrix.yaml internal invariants hold"
    except MatrixError as e:
        yaml_ok, yaml_detail = False, str(e)
    ok("yaml_invariants", yaml_ok, yaml_detail)

    try:
        validate_code_defaults()
        code_ok, code_detail = True, "Python defaults match runtime:"
    except MatrixError as e:
        code_ok, code_detail = False, str(e)
    ok("code_defaults_match_yaml", code_ok, code_detail)

    ok(
        "t4_is_cer_oracle",
        "T4" in CER_ORACLE_ARMS and select_mode("T4") == "cer_oracle",
        "T4 select_mode is cer_oracle",
    )
    ok(
        "t4_not_l2",
        "T4" not in L2_ARMS,
        "T4 is not in L2_ARMS",
    )
    ok(
        "t4_se_always",
        "T4" in GLOBAL_SE_ARMS and se_mode("T4") == "always",
        "T4 se_mode is always",
    )

    rec = {
        "uid": "probe",
        "oracle_stream": "original",
        "dual_zero": True,
        "streams": {
            "original": {"cer": 0.0},
            "spk1": {"cer": 0.0},
            "spk2": {"cer": 0.4},
        },
    }
    cos = {"probe": {"original": 0.50, "spk1": 0.99, "spk2": 0.10}}
    qkw = {"probe": {"original": 0.40, "spk1": 0.95, "spk2": 0.10}}
    t4 = pick_track("T4", rec, cos_map=cos, pm_map={}, qkw_map=qkw)
    t2 = pick_track("T2", rec, cos_map=cos, pm_map={}, qkw_map=qkw)
    t2_cos_only = pick_track("T2", rec, cos_map=cos, pm_map={}, qkw_map={})
    ok(
        "t4_ignores_qkw_sidecar",
        t4["chosen"] == "original" and t4["reason"] == "t4_cer_oracle",
        "with a q_kw sidecar that prefers spk1, T4 still keeps CER-oracle original",
    )
    ok(
        "t2_uses_qkw_not_cos",
        t2["chosen"] == "spk1" and t2_cos_only["chosen"] == "original",
        "q_kw moves T2 to spk1; cos-only does not (cos is not a purity score)",
    )

    sidecar_ok = True
    sidecar_detail = "empty scores and whole-row fallback raise SidecarError"
    try:
        parse_cos_row({"uid": "a", "scores": {}}, index=0)
        sidecar_ok = False
        sidecar_detail = "empty scores dict was accepted"
    except SidecarError:
        pass
    try:
        parse_cos_row({"uid": "a", "oracle_stream": "original", "spk1": 0.9}, index=0)
        sidecar_ok = False
        sidecar_detail = "whole-row fallback was accepted"
    except SidecarError:
        pass
    ok("sidecar_rejects_ambiguous", sidecar_ok, sidecar_detail)

    partial_ok = True
    partial_detail = "non-empty q_kw map missing uid raises SidecarError"
    try:
        pick_track(
            "T2",
            rec,
            cos_map={},
            pm_map={},
            qkw_map={"other": {"original": 0.9, "spk1": 0.8, "spk2": 0.1}},
        )
        partial_ok = False
        partial_detail = "partial sidecar was accepted"
    except SidecarError:
        pass
    ok("partial_sidecar_is_error", partial_ok, partial_detail)

    # FA is allowed only in its isolated experiment module. Frozen selectors
    # must not import or call it.
    frozen = "\n".join(
        (ROOT / rel).read_text(encoding="utf-8")
        for rel in ("src/kws/t0_t4.py", "src/kws/select_l2.py", "scripts/run_t0_t4.py")
    )
    ok(
        "fa_isolated_from_t0_t4",
        "fa_experiment" not in frozen and "MmsFa" not in frozen and "pick_mms" not in frozen,
        "FA experiment is not imported by frozen T0/T2/T4 selectors",
    )

    pipeline = (ROOT / "docs" / "PIPELINE.md").read_text(encoding="utf-8")
    rerun = (ROOT / "scripts" / "rerun_sep.sh").read_text(encoding="utf-8")
    ok(
        "extract_sep_pipeline",
        "clone -b sep" in pipeline and "run_sep.sh" in pipeline and "EXTRACT_BRANCH:=sep" in rerun,
        "docs/PIPELINE.md and rerun_sep.sh point at extract@sep",
    )

    sidecar_script = (ROOT / "scripts" / "build_eres_sidecar.py").read_text(encoding="utf-8")
    wav_paths = (ROOT / "src" / "kws" / "wav_paths.py").read_text(encoding="utf-8")
    ok(
        "eres_sidecar_builder_exists",
        "resolve_kws_wav" in sidecar_script and '"peak"' in wav_paths and "original" in wav_paths,
        "build_eres_sidecar.py resolves datasetA kws_rel; original stream is peak",
    )
    eval_script = ROOT / "scripts" / "eval_cmd_cosine.py"
    export_script = ROOT / "scripts" / "export_best_sep_groups.py"
    ok(
        "cmd_eval_and_multi_best_sep",
        eval_script.is_file() and export_script.is_file(),
        "export_best_sep_groups.py and eval_cmd_cosine.py exist",
    )

    failed = 0
    for name, cond, detail in CHECKS:
        mark = "PASS" if cond else "FAIL"
        if not cond:
            failed += 1
        print(f"[{mark}] {name}: {detail}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} passed")
    print(
        "NOTE: this review still cannot prove ERes ran on GPU or CMD cosine was measured."
        " Run scripts/run_kws_eval.py for that. Presence is a later extract@main step."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
