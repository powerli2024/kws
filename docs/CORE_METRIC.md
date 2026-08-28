# 注册音频选择的核心指标与计算方法

本文是指标口径的唯一说明。必须区分两个问题：

1. **单条 KWS 的多路候选音频怎么选**：使用“CER 硬约束 + `q_kw` 核心排序”。
2. **一整组 `best_sep` 是否更好**：使用 enroll↔CMD cosine 做本仓初筛，再用 extract `main` 的冻结 Presence/竞赛分作最终否决。

CMD 音频和标签只允许用于离线评估，不能反过来参与单条注册音频选择。

## 1. 单条注册音频选择

对一个 UID，候选集合为 `original`、`spk1`、`spk2` 及现有级联分离流。记候选音频为
`x_i`，已知唤醒文本为 `y=(y_1,...,y_T)`。

### 1.1 第一层：CER 是硬约束，不是优化目标

先计算每路候选相对已知唤醒文本的 CER：

```text
CER_i = EditDistance(normalize(hyp_i), normalize(y)) / len(normalize(y))
CER_min = min_i CER_i
S = {i | CER_i <= CER_min + delta}
```

当前正式配置 `delta=0`，即只有与最小 CER 完全相同的候选进入第二层。中文沿用项目的拼音
CER，英文沿用字符 CER。不要把 CER 与声纹分数加权求和，也不要用全组 mean CER 给注册集排名。

如果没有真实 `q_kw`/NLL，E2 必须停止或降级为 E1；降级结果不能解释为“L2 无效”。E1 的确定性
回退规则是 CER 最小、并在并列时优先 `original`。

### 1.2 第二层：核心排序指标是 `q_kw`

在冻结 ASR 上，对每个 `i in S` 用已知唤醒文本做 teacher-forced/forced-decode，计算长度归一化
token NLL：

```text
NLL_i = - sum_t [w_t * log p_theta(y_t | y_<t, x_i)] / sum_t w_t
s_text_i = -NLL_i
```

- `theta` 必须冻结，所有候选使用同一模型、tokenizer、音频前处理和提示词。
- `w_t=1` 仅覆盖真实唤醒文本 token；BOS、EOS、PAD、提示词和非目标 token 不计入分母。
- 必须按有效目标 token 数归一化，不能用总 NLL，否则长 tokenization 会被系统性惩罚。
- NLL 越低越好；sidecar 写 `nll` 时，加载器取负数后按越大越好排序。

仓库实现为 `scripts/score_qkw_nll.py`。它固定空 context，只把已知唤醒文本放入 target labels，
屏蔽 prefix/PAD/EOS，逐 UID 输出所有候选流的长度归一化 NLL。20 UID 冒烟应使用独立的
`q_kw_nll_smoke.jsonl`；全量输出为 `reports/sidecars/q_kw_nll.jsonl`，可直接传给
`run_kws_eval.py --qkw-jsonl`。中断后只有模型、输入哈希和参数签名完全一致时才允许 `--resume`。

推荐在独立校准集上按语言拟合单调校准器：

```text
q_kw_i = Calibrate_lang(s_text_i),  0 <= q_kw_i <= 1
i* = argmax_{i in S} q_kw_i
```

`Calibrate_lang` 可以是预先冻结的 logistic 或 isotonic calibration；拟合目标是“该路完整、正确地保留
已知唤醒词”的二值标签。校准集不能与最终评估集重合。未校准的 `-NLL` 可以用于同一 UID 内排序，
但不能使用 `q_kw>=0.80` 这类绝对阈值。

完全同分时，当前实现先优先 `original`；若 `original` 不在候选集合中，再使用确定性的流名称顺序。

### 1.3 两个安全门不参与核心排序

**分离灾难回退：** 若 `i*` 是分离流，计算

```text
c_raw = cosine(embed(x_i*), embed(x_raw))
```

当 `c_raw < 0.92` 且 `original in S` 时回退到 `original`。`0.92` 目前是待校准占位值；
`cos(candidate,raw)` 只能发现严重失真，不能衡量纯净度，也不能参与候选排名。

**疑似两个说话人时拒绝注册：** 只有 `q_kw` 已校准时才启用。若至少两路分离流满足
`q_kw>=0.80`，且这些高分流中最小 pairwise speaker cosine `<0.35`，输出 `reject`。正式运行必须为
所有相关流提供完整 `pair_cos`；缺失 pair cosine 时不能声称该拒绝门已经验证。

## 2. 整组注册音频的离线核心评估

单条选择完成后，为每个候选组和 UID 计算：

```text
z_uid = cosine(ERes2NetV2(enroll_uid), ERes2NetV2(cmd_uid))
```

其中 pos 的 `z` 应更高，neg 的 `z` 应更低。使用冻结的语言阈值 `tau_lang`：

```text
FRR_lang = count(pos: z < tau_lang) / N_pos_lang
FAR_lang = count(neg: z >= tau_lang) / N_neg_lang
```

本仓对注册组的主排序为：

```text
FAR（越低越好） -> FRR（越低越好） -> pos P10（越高越好） -> EER（越低越好）
```

同时报告 AUC、mean gap、neg P90、Wilson 区间和相对 E1 的逐 UID paired delta，但它们不是第一排序键。
冻结阈值只对同一 ERes2NetV2 分数空间有效；FFT backend 只能检查管线，不能产生排名。

所有组必须覆盖 datasetA 的同一完整 UID 集，而不只是彼此交集。当前口径是 1,838 条：
1,364 pos + 474 neg。共同漏样、缺失音频或静默丢弃困难样本都应阻断排名。

## 3. 最终采用规则

CMD cosine 只是 KWS-local 初筛，不是最终上线指标。候选组必须分别送到 extract `main`，冻结
ERes2NetV2、语言阈值、CMD 集和下游 ASR，只替换 enroll，计算真实 FRR、FAR、字符加权 CER：

```text
RR = 1 - FAR
contest = 0.5 * RR + 0.5 * (1 - CER_micro)
```

正样本被拒、提取失败或 ASR 失败时该条 CER 记为 1。最终采用还必须满足：全量 coverage、
`CER_mean<=0.03`、CER=0 rate 相对基线下降不超过 2 个百分点、语言切片不出现不可接受退化，且相对
冻结基线的配对 bootstrap 分差下界通过预设 Go/No-Go。

## 4. 明确禁止的替代指标

- 不用 `cos(candidate,raw)` 给候选排序；它天然偏向 original。
- 不用 mean oracle CER 给注册组排名；CER 在大量 dual-zero 样本上没有区分度。
- 不用 `p_music`、SNR、DNSMOS BAK、SI-SDR 作为说话人纯净度核心指标。
- 不用 CMD cosine 为同一条样本选择 enroll；这会使用评估期 CMD 信息，造成标签泄漏。
- 不使用未经校准的 NLL 激活 `q_kw` 绝对阈值或双说话人拒绝门。
