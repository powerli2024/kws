"""Registration-text CER used by KWS experiments.

Chinese is compared as toneless pinyin; non-Chinese text is compared by
normalized characters.  This intentionally matches extract-sep's KWS metric,
not a competition's strict character CER.
"""

from __future__ import annotations

import re
import string
import unicodedata

_CJK = re.compile(r"[\u4e00-\u9fff]")
_PUNCT = str.maketrans("", "", string.punctuation + "，。！？、；：‘’“”「」『』（）【】《》…·—–")


def normalize_chars(text: str | None) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = "".join(ch for ch in value if not ch.isspace())
    return value.translate(_PUNCT).lower().strip()


def has_cjk(text: str | None) -> bool:
    return bool(_CJK.search(str(text or "")))


def to_pinyin(text: str | None) -> str:
    try:
        from pypinyin import Style, lazy_pinyin
    except ModuleNotFoundError as exc:
        raise RuntimeError("Chinese KWS CER requires pypinyin; install the project requirements") from exc
    value = normalize_chars(text)
    return "".join(lazy_pinyin(value, style=Style.NORMAL, errors=lambda x: list(x.lower())))


def cer_detail(hyp: str | None, ref: str) -> dict[str, float | str]:
    hyp_chars, ref_chars = normalize_chars(hyp), normalize_chars(ref)

    def distance(left: str, right: str) -> float:
        if not right:
            return 0.0 if not left else 1.0
        previous = list(range(len(left) + 1))
        for i, rch in enumerate(right, start=1):
            current = [i]
            for j, lch in enumerate(left, start=1):
                current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (rch != lch)))
            previous = current
        return float(previous[-1]) / len(right)

    cer_char = distance(hyp_chars, ref_chars)
    cer_py = distance(to_pinyin(hyp), to_pinyin(ref))
    use_pinyin = has_cjk(ref)
    return {
        "hyp": hyp_chars,
        "cer": cer_py if use_pinyin else cer_char,
        "cer_char": cer_char,
        "cer_py": cer_py,
        "metric": "pinyin" if use_pinyin else "char",
    }
