# Window min-cos landing

```text
if dur < 0.8s:
    skip metric
else:
    windows = 0.6s, hop 0.3s, plus a tail window so the end is covered
    require >= 2 windows
    e_i = eres(window_i)
    min_cos = min_{i<j} cos(e_i, e_j)
    anomaly if min_cos < percentile_τ(corpus min_cos)
```

| Piece | Value | How to freeze |
|---|---|---|
| min duration | 0.8 s | proposal; shorter clips cannot host two 0.6 s windows stably |
| window | 0.6 s | proposal |
| hop | 0.3 s | 50% overlap; search {0.2, 0.3, 0.4} only if listen-100 F1 is flat |
| embedding | eres2netv2 | same as Presence; FFT proxy in the script is not a gate |
| threshold | P5/P10/P15/P20 of **this** corpus | pick by listen-100, not 0.x cosine from another domain |

Same-file two-segment check is the same code path with exactly two windows (head + tail) when `0.8s ≤ dur < 1.2s`.
