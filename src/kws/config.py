"""Load configs/experiment_matrix.yaml and refuse internal contradictions."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .arms import CER_ORACLE_ARMS, CONDITIONAL_SE_ARMS, GLOBAL_SE_ARMS, L2_ARMS, NO_SE_ARMS, se_mode, select_mode


class MatrixError(ValueError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def matrix_path() -> Path:
    return repo_root() / "configs" / "experiment_matrix.yaml"


@lru_cache(maxsize=1)
def matrix() -> dict[str, Any]:
    path = matrix_path()
    if not path.is_file():
        raise MatrixError(f"missing experiment matrix: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise MatrixError("experiment_matrix.yaml is not a mapping")
    validate_matrix(data)
    return data


def runtime() -> dict[str, Any]:
    return dict(matrix()["runtime"])


def validate_matrix(data: dict[str, Any]) -> None:
    errors: list[str] = []
    rt = data.get("runtime")
    if not isinstance(rt, dict):
        raise MatrixError("runtime: block is required")

    def eq(path_a: str, a: Any, path_b: str, b: Any) -> None:
        if a != b:
            errors.append(f"{path_a}={a!r} != {path_b}={b!r}")

    cer = data.get("cer") or {}
    cat = data.get("catastrophe_cos") or {}
    win = data.get("window_min_cos") or {}
    eq("runtime.l1_slack", rt.get("l1_slack"), "cer.l1_slack", cer.get("l1_slack"))
    ns = data.get("need_se") or {}
    eq("runtime.l1_slack", rt.get("l1_slack"), "need_se.cer_slack", ns.get("cer_slack"))
    eq(
        "runtime.catastrophe_cos",
        rt.get("catastrophe_cos"),
        "catastrophe_cos.default_until_calibrated",
        cat.get("default_until_calibrated"),
    )
    eq("runtime.se_cos_thr", rt.get("se_cos_thr"), "runtime.catastrophe_cos", rt.get("catastrophe_cos"))
    eq("runtime.window_min_dur_sec", rt.get("window_min_dur_sec"), "window_min_cos.min_dur_sec", win.get("min_dur_sec"))
    eq("runtime.window_win_sec", rt.get("window_win_sec"), "window_min_cos.win_sec", win.get("win_sec"))
    eq("runtime.window_hop_sec", rt.get("window_hop_sec"), "window_min_cos.hop_sec", win.get("hop_sec"))

    arms = data.get("arms") or {}
    expected = {
        "T0": ("cer_oracle", "none"),
        "T1": ("cer_oracle", "conditional"),
        "T2": ("l2", "none"),
        "T3": ("l2", "conditional"),
        "T4": ("cer_oracle", "always"),
    }
    for arm, (sel, se) in expected.items():
        spec = arms.get(arm) or {}
        got_sel = spec.get("select")
        got_se = spec.get("se")
        if sel == "l2":
            ok_sel = got_sel in ("l2", "l2_cos_to_raw_under_cer_slack")
        else:
            ok_sel = got_sel == sel
        if not ok_sel:
            errors.append(f"arms.{arm}.select={got_sel!r} expected {sel}")
        if got_se != se:
            errors.append(f"arms.{arm}.se={got_se!r} expected {se}")
        if select_mode(arm) == "cer_oracle" and arm not in CER_ORACLE_ARMS:
            errors.append(f"{arm} yaml select is cer_oracle but not in CER_ORACLE_ARMS")
        if select_mode(arm) == "l2" and arm not in L2_ARMS:
            errors.append(f"{arm} yaml select is l2 but not in L2_ARMS")
        if se_mode(arm) != se:
            errors.append(f"{arm} se_mode()={se_mode(arm)!r} != yaml {se!r}")

    if "T4" in L2_ARMS:
        errors.append("T4 must not be in L2_ARMS")
    if "T4" not in CER_ORACLE_ARMS or "T4" not in GLOBAL_SE_ARMS:
        errors.append("T4 must be CER-oracle + always-SE")
    if CONDITIONAL_SE_ARMS & GLOBAL_SE_ARMS:
        errors.append("conditional and global SE arm sets overlap")
    if NO_SE_ARMS & (CONDITIONAL_SE_ARMS | GLOBAL_SE_ARMS):
        errors.append("no-SE arms overlap SE arms")

    if errors:
        raise MatrixError("; ".join(errors))


def validate_code_defaults() -> None:
    """Python constants must match runtime:."""
    from . import need_se as need_se_mod
    from . import select_l2 as select_l2_mod
    from . import skip_sep as skip_mod
    from . import window_mincos as win_mod

    rt = runtime()
    mismatches: list[str] = []

    def chk(name: str, got: Any, key: str) -> None:
        exp = rt[key]
        if isinstance(got, bool) or isinstance(exp, bool):
            if got != exp:
                mismatches.append(f"{name}={got!r} != runtime.{key}={exp!r}")
            return
        if isinstance(got, (int, float)) and isinstance(exp, (int, float)):
            if float(got) != float(exp):
                mismatches.append(f"{name}={got!r} != runtime.{key}={exp!r}")
            return
        if got != exp:
            mismatches.append(f"{name}={got!r} != runtime.{key}={exp!r}")

    chk("select_l2.CER_SLACK_DEFAULT", select_l2_mod.CER_SLACK_DEFAULT, "l1_slack")
    chk("select_l2.DEFAULT_LAMBDA", select_l2_mod.DEFAULT_LAMBDA, "lambda")
    chk("select_l2.DEFAULT_CATASTROPHE_COS", select_l2_mod.DEFAULT_CATASTROPHE_COS, "catastrophe_cos")
    chk("need_se.DEFAULT_P_MUSIC_ORIG", need_se_mod.DEFAULT_P_MUSIC_ORIG, "p_music_orig")
    chk("need_se.DEFAULT_P_MUSIC_SEP", need_se_mod.DEFAULT_P_MUSIC_SEP, "p_music_sep")
    chk("need_se.DEFAULT_SNR_ORIG", need_se_mod.DEFAULT_SNR_ORIG, "snr_orig_db")
    chk("need_se.DEFAULT_SNR_SEP", need_se_mod.DEFAULT_SNR_SEP, "snr_sep_db")
    chk("need_se.DEFAULT_SE_COS_THR", need_se_mod.DEFAULT_SE_COS_THR, "se_cos_thr")
    chk("window_mincos.MIN_DUR_SEC", win_mod.MIN_DUR_SEC, "window_min_dur_sec")
    chk("window_mincos.WIN_SEC", win_mod.WIN_SEC, "window_win_sec")
    chk("window_mincos.HOP_SEC", win_mod.HOP_SEC, "window_hop_sec")
    chk("skip_sep.SKIP_BSS_P_MUSIC_MAX", skip_mod.SKIP_BSS_P_MUSIC_MAX, "skip_bss_p_music_max")
    chk("skip_sep.SKIP_BSS_SNR_MIN_DB", skip_mod.SKIP_BSS_SNR_MIN_DB, "skip_bss_snr_min_db")
    chk("skip_sep.SKIP_BSS_ENABLED_DEFAULT", skip_mod.SKIP_BSS_ENABLED_DEFAULT, "skip_bss_enabled")
    if mismatches:
        raise MatrixError("; ".join(mismatches))
