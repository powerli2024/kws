"""T0–T4 factor sets. YAML arms.* must match these (see config.validate_matrix).

T4 is CER oracle + always SE — never L2. Mixing T4 into L2 confounds the ablation.
"""

from __future__ import annotations

CER_ORACLE_ARMS = frozenset({"T0", "T1", "T4"})
L2_ARMS = frozenset({"T2", "T3"})
NO_SE_ARMS = frozenset({"T0", "T2"})
CONDITIONAL_SE_ARMS = frozenset({"T1", "T3"})
GLOBAL_SE_ARMS = frozenset({"T4"})
ALL_ARMS = frozenset({"T0", "T1", "T2", "T3", "T4"})


def select_mode(arm: str) -> str:
    if arm in CER_ORACLE_ARMS:
        return "cer_oracle"
    if arm in L2_ARMS:
        return "l2"
    raise ValueError(f"unknown arm {arm!r}")


def se_mode(arm: str) -> str:
    if arm in NO_SE_ARMS:
        return "none"
    if arm in CONDITIONAL_SE_ARMS:
        return "conditional"
    if arm in GLOBAL_SE_ARMS:
        return "always"
    raise ValueError(f"unknown arm {arm!r}")
