# KWS 注册音频全流程：s1–s8 同 UID 选路、SE 与最终验收

## 1. 目标与结论

本项目的目标不是简单寻找“听起来最干净”的音频，而是从 extract-sep 产生的 s1–s8 同 UID
候选中，选择最适合注册的音频，并证明它在已知唤醒文本、说话人保持和下游 KWS 上都优于冻结基线。

正式思路应当是：

1. 先对 s1–s8 的 raw 音频做完整、独立、可复现的评价。
2. 探索阶段对所有候选流生成 MossFormer2_SE_48K 视图，同时永久保留 raw；不应先用未经校准的
   噪声检测器拦截一部分流。
3. 低 CER 是最终注册候选的第一层硬约束。中文使用无调拼音 CER；英文同时保存严格字符 CER、
   冻结别名 CER 和关键词覆盖，处理 `hi colmo` 被识别成 `hey colmo` 这类可接受 ASR 变体。
4. 每条 raw/SE 的 CER 变化全部记录，但不按“SE 后 CER 上升”一刀切拒绝。某些流本来就是干扰人声、
   音乐或噪声，必须结合 ASR 文本类别解释；最终被选作注册音频的候选仍必须满足低 CER 硬约束。
5. 在精确最小 CER 候选内，再用校准 `q_kw` 排序；没有校准 `q_kw` 时，只能使用 target NLL
   做同 UID、同 CER 候选的相对排序，不能使用绝对阈值。
6. 噪声分数、`cos(SE, raw)`、DNSMOS、SNR 等只做诊断、异常门或后续降算力触发器，不能取代
   CER、`q_kw` 和冻结下游评价。
7. 先得到每个固定 s1–s8 阶段的评价和“对 s1 的互补增益”，再决定 s1 应和 s7、其他阶段或 SE
   如何组成级联。不能先在最终测试集自动挑最佳阶段。

这里的“s1 与 SE 结合”和“s1 与 s7/其他阶段结合”是两个独立因素，必须先分别实验，再测试组合，
否则无法判断提升来自分离阶段还是增强模型。

## 2. 数据与目录契约

`POS_NEG` 指向 extract-sep 的完整输出根目录，例如：

```text
/root/autodl-tmp/kws_sep_fullaudio_v1/
├── pos/
│   ├── s1_onnx_full/index.jsonl
│   ├── s1_onnx_full/wav/
│   ├── s2_cv_full/
│   ├── ...
│   └── s7_cv_then_onnx_gate/thr_a/index.jsonl
├── neg/
│   └── 同样结构
└── reports/
```

每条记录至少需要：

```text
uid, split, wake_text, streams.{original,spk1,spk2...}, 每流 CER, WAV 路径
```

旧 index 缺少 `lang` 时，可以由 `wake_text` 稳定推导 `zh/en`；如果连注册文本也缺失，必须从
DatasetA 的 `pos.jsonl/neg.jsonl` 回填，不能猜测 CER 参考文本。

正式运行必须满足：

- 1,838 UID 全覆盖，当前口径为 1,364 pos + 474 neg。
- 不静默丢弃失败样本。
- 所有音频按完整长度处理
- 同时保存容器文件 `file_sha256` 和统一解码后 PCM 的 `pcm_sha256`；以 `pcm_sha256` 建立 canonical
  audio registry，使相同波形即使 WAV 头或路径不同也只核对一次，所有阶段别名只引用同一结果。
- ASR 缓存键至少包含 `(pcm_sha256, wake_text, lang, model_signature)`。
- NLL、embedding、VAD、噪声、音乐和其他确定性特征也只对 canonical audio 计算一次，不按阶段别名重复检查。
- SE 缓存键至少包含 `(raw_pcm_sha256, se_backend, model_signature, inference_signature)`；相同 SE 输出
  再按输出 PCM SHA256 合并，其 ASR/embedding/噪声指标同样只计算一次。
- 不同 SE 后端、模型权重或参数使用不同 `WORK_DIR`；缓存签名不一致时拒绝复用。

## 3. 候选的统一表示

同一 UID 的一个候选应表示为：

```text
(stage, threshold_arm, stream, view)
```

其中：

- `stage`：s1–s8 的固定阶段族。
- `threshold_arm`：如 `thr_a/thr_b/thr_c`，必须独立评价，不能因名字不同就当成不同有效实验。
- `stream`：`original/spk1/spk2/...`。
- `view`：`raw`、`moss_se48k`、`spectral` 或其他对照。

所有引用相同波形的候选必须先按 `pcm_sha256` 合并。对一份 canonical audio，ASR、NLL、embedding、VAD、
噪声/音乐指标和其他特征都只计算一次；各阶段、阈值和 stream 仅保存 alias 引用。相同音频因重复推理
出现不同 CER 时，不能把最低值当真实提升，正式重算应从源头做到“一份音频一次评分、一次质检”。

## 4. 第一阶段：s1–s8 raw 基线评价

### 4.1 固定阶段评价

分别评价每个固定阶段和阈值臂，不能直接使用跨阶段 `best_sep` 神谕结果冒充可部署路由。

每个 UID 在一个固定阶段内：

```text
CER_min = min(CER_i)
S_cer = {i | CER_i == CER_min}
```

`delta=0`，只有精确最小 CER 候选进入下一层。完全同分且没有其他可靠证据时，确定性回退顺序为：

```text
raw 优先 → original 优先 → 固定流名称顺序
```

每个阶段至少报告：

- coverage、错误数和缺失 WAV 数；
- mean CER、CER0 数量和比例；
- 各 stream 被选次数；
- 与父阶段的逐 UID 改善、恶化、相同数量；
- 阈值臂之间是否拥有相同 cohort、相同 WAV 或相同语义评分；
- 计算量和推理失败率，作为次要指标。

### 4.2 同 UID 全阶段评价

对每个 UID 汇总 s1–s8 全部独立音频，生成：

- 所有候选的 dense rank；
- 最低 CER 并列集合；
- 每个阶段的独占最佳数、并列最佳数和 tie credit；
- “全阶段神谕上界”，只用于判断候选空间还有多少潜力，不能直接部署。

当前仓库已实现：

```bash
python scripts/compare_all_stages.py \
  --pos-neg /root/autodl-tmp/kws_sep_fullaudio_v1 \
  --expected-uids 1838

python scripts/rank_same_uid_audio.py \
  --pos-neg /root/autodl-tmp/kws_sep_fullaudio_v1 \
  --expected-uids 1838 --top-k 0
```

第二条正式分析使用 `--top-k 0`，避免截断后无法精确计算阶段互补性。

### 4.3 英文唤醒词的额外低 CER 口径

英文必须同时保存严格指标和业务容错指标，不能用一个宽松指标覆盖原始错误。对参考文本 `r`、ASR 文本
`h`，先做冻结规范化 `norm_en()`：小写、统一空白和标点，但不自动改写单词。计算：

```text
CER_char(r,h) = EditDistance(chars(norm_en(r)), chars(norm_en(h)))
                / max(1, len(chars(norm_en(r))))
```

对于 `hicolmo` 被识别成 `heycolmo`（以及 `hi colmo`/`hey colmo`）这类业务上允许、声学上合理但
严格字符 CER 不为零的结果，需在最终测试前冻结每个唤醒词的可接受别名集合 `A(r)`，例如：

```text
A("hicolmo") = {"hicolmo", "heycolmo", "hi colmo", "hey colmo"}
CER_alias(r,h) = min[a in A(r)] CER_char(a,h)
```

正式英文低 CER 可使用 `CER_alias`，但必须满足以下约束：

- 报告始终同时保留 `CER_char`、命中的 `alias` 和 `CER_alias`，不能只展示更好看的值；
- 别名由产品语义和独立开发集预先确定，禁止看最终测试输出后追加别名；
- 品牌核心词必须命中。以该示例为例，`colmo` 缺失或变成其他品牌词不能靠 `hi/hey` 容错通过；
- 记录 `wake_coverage` 与 `extra_ratio`，避免“包含关键词但还有大量其他语音”被视为低 CER；
- 音素/PER 可以作为诊断，但不能自动把任意近音词都判为正确。

推荐的文本辅助量为：

```text
wake_coverage = 已按顺序匹配的目标 token 或字符数 / 目标 token 或字符数
extra_ratio   = 对齐后非目标插入 token 或字符数 / max(1, ASR 输出长度)
core_hit      = 1[核心品牌词或冻结核心别名被完整保留]
```

最终英文候选的 L1 仍是低 `CER_alias`，并要求 `core_hit=1`；`wake_coverage`、`extra_ratio` 和严格
`CER_char` 用于并列解释、风险审计及验收切片，而不是临时放宽测试结果。

## 5. 第二阶段：对所有候选流做 SE

### 5.1 为什么探索阶段应全流运行

如果先用 SNR、`p_music` 或其他噪声检测决定“是否运行 SE”，会产生选择盲区：检测器可能认为音频
不需要增强，但该音频实际上会从 SE 获益。效果优先阶段应对 s1–s8 的每个 unique raw WAV 都生成
SE 视图，之后再根据真实收益判断哪些流需要 SE。

因此正式探索矩阵是：

| 实验臂 | 作用 |
|---|---|
| raw | 冻结基线 |
| 16k→48k→16k resample-only | 隔离重采样影响 |
| spectral SE | 低成本频谱处理对照，不代表神经 SE |
| MossFormer2_SE_48K | 主 SE 候选 |
| 其他冻结 SE | 可选，如 DeepFilterNet/FRCRN；必须独立目录和签名 |
| always-SE | 负对照，验证“无条件替换 raw”是否有害 |

每个 SE 后端都必须保留 raw，不能覆盖源文件。

### 5.2 MossFormer2_SE_48K 音频链

extract-main 的模型调用链必须显式记录：

```text
raw 16 kHz
  → poly resample 48 kHz
  → MossFormer2_SE_48K 完整波形推理
  → poly resample 16 kHz
  → 冻结 ASR / q_kw / 声纹评价
```

默认优先完整波形。只有完整推理发生 GPU OOM 时，才允许采用 8/4/2 秒带重叠分块重试，并在结果中
记录 `full` 或 `chunkXs`，因为分块可能影响增强效果。

KWS 已提供一次加载模型的 manifest 适配器：

```text
scripts/extract_main_se48k_manifest.py
```

### 5.3 SE 前后 CER 的记录与解释

对同一 raw 候选 `x` 和对应 SE 视图 `x_se`，完整记录：

```text
delta_CER = CER(x_se) - CER(x)
```

但不能把 `delta_CER > 0` 直接等同于“SE 有害并硬拒绝”。s1–s8 的某些流可能主要包含竞争人声、
音乐、噪声或混合内容；其 raw ASR 恰好接近唤醒词也可能是幻觉。SE 去掉干扰后，ASR 文本变化甚至
CER 上升，不必然表示目标说话人被破坏。每个 raw/SE 对需要结合 ASR 文本划分：

| 文本/内容类别 | 判定证据 | SE 前后解释 |
|---|---|---|
| 目标唤醒流 | 低 CER、高 `wake_coverage`、`core_hit=1` | CER 上升是风险信号，但先记录并与 q_kw、声纹及下游结果联合判断 |
| 竞争人声流 | ASR 为连贯非目标文本、目标覆盖低、speech ratio 高 | 抑制后变为空或非目标不是失败；该流通常不应成为注册候选 |
| 噪声/音乐流 | 低 speech ratio、音乐/噪声证据高、ASR 为空或低置信幻觉 | CER 变化主要反映 ASR 幻觉，不作为 SE 成败的单一依据 |
| 目标+干扰混合流 | 命中目标但 `extra_ratio` 高，或检测到重叠语音 | 同时看目标保留和额外语音是否下降 |
| 不确定流 | 各证据冲突或置信度不足 | 保留全部字段并进入人工误差分析，不据单项自动下结论 |

这里“不硬拒绝”指不因单个 raw/SE 配对的 CER 变化提前删除实验记录。最终同 UID 选路仍按所有
raw/SE 候选的 L1 低 CER 排序，因此高 CER 的 SE 视图自然不会战胜低 CER raw；若 SE 的低 CER 来自
疑似幻觉，还必须由 `q_kw`、文本覆盖和下游声纹评价排除。

最终导出的整个路由仍必须满足：

```text
CER_selected(uid) <= CER_frozen_baseline(uid)
```

或者将任何恶化样本明确回退到冻结基线。报告必须给出 paired improved/worsened/same，不能只看均值。

## 6. 第三阶段：CER 后的 `q_kw` 与其他证据

### 6.1 正式排序顺序

单 UID 候选使用字典序决策，不做随意加权求和：

```text
CER_route = CER_pinyin（中文）或 CER_alias（英文，别名集合已冻结）
L1：最小 CER_route，硬约束
L2：最大 calibrated q_kw
L3：可靠的独立说话人证据或确定性回退
```

英文同时保留严格 `CER_char`；任何 `core_hit=0` 的英文结果不进入 L1 候选集。`CER_route` 只统一选路
接口，不表示中英文指标可以混在一起计算未分语言的宏平均。

`q_kw` 来自冻结 ASR 对已知注册文本的 teacher-forced target NLL：

```text
NLL_i = -mean_t log P(y_t | y_<t, audio_i)
q_kw_i = Calibrate_lang(-NLL_i)
```

校准器应按中文/英文分别在独立校准集上拟合 logistic 或 isotonic calibration，标签是“该候选是否
完整、正确地保留注册唤醒词”。校准集不能与最终测试集重合。

没有校准器时：

- raw NLL 只能在同一 UID 的精确最小 CER 集内排序；
- 不能使用 `q_kw >= 0.8` 等绝对阈值；
- 不能据此启动双说话人绝对拒绝门；
- 报告必须写 `score_kind=nll`，不能写成 `q_kw`。

### 6.2 `cos(SE, raw)` 的正确位置

`cos(embed(SE), embed(raw))` 只描述增强前后 embedding 的变化幅度，不描述变化方向。raw 本身可能
含噪、混响或第二说话人，因此 SE 真正去除干扰时，这个 cosine 也可能下降。

当前 `0.92` 来自待校准网格 `0.90–0.95` 的临时值，不是当前数据上统计得到的声纹增强阈值；同理，
不能因为阈值写成 `0.90` 就变得可靠。因此可以计算，但默认不作为硬门：

- 不用 `cos(SE, raw)` 给候选排序；
- 不把 `<0.90` 或 `<0.92` 直接解释为“说话人错误”；
- 探索阶段计算 `c_change=cos(e_se,e_raw)`，报告 p01/p05/p50/p95、直方图和异常样本；
- 计算 `c_change` 与 `delta_CER`、`delta_q_kw`、噪声/音乐变化的相关性，确认它实际描述了什么；
- 若要作为硬灾难门，必须用同说话人/异说话人数据及固定编码器重新校准；
- 如果存在独立、同说话人的 enrollment reference，更有意义的增益是比较
  `gain_ref=cos(e_se,e_ref)-cos(e_raw,e_ref)`；不能把同一份受污染 raw 当成绝对真值；
- 可分别用冻结 ERes2NetV2、CAM++ 计算 `gain_ref`，报告两个编码器的方向是否一致；不一致时标记
  `speaker_uncertain`，而不是择优挑一个分数；
- 没有独立 reference 时直接跳过 `gain_ref`，最终由 CMD/Presence 的 FRR/FAR 验证。

### 6.3 噪声检测的正确位置

“噪声”不能只用一个 SNR 表示。数据中可能包含稳态噪声、瞬态冲击、音乐、混响、竞争人声、重叠
人声、削波或大段静音；其中音乐和竞争人声尤其不能简单并入背景噪声。所有阶段、所有 raw/SE 的
canonical audio 都要执行同一套冻结评估，再通过 alias 回填到 s1–s8，不能只检查最终被选中的音频。

每份 canonical audio 至少计算以下指标；`eps` 为固定小常数，所有阈值、模型版本及采样率写入签名：

| 类别 | 指标及计算方法 | 解释与限制 |
|---|---|---|
| 电平/损坏 | `duration=N/sr`；`rms_dbfs=20*log10(sqrt(mean(x^2))+eps)`；`peak=max(abs(x))`；`clip_rate=count(abs(x)>=0.999)/N`；`dc=abs(mean(x))`；`zero_ratio=count(abs(x)<eps)/N` | 发现过短、过静、削波、直流偏置或断流；不是语义纯净度 |
| 语音占比 | 冻结 VAD 帧标签；`speech_ratio=sum(speech_frame_duration)/duration`；同时报告最长连续语音和静音比例 | 音乐可能误触发 VAD，需与音乐分类联合解释 |
| SNR 代理 | 令 `P_active` 为 VAD 语音帧功率、`P_noise` 为非语音帧功率，计算 `snr_vad=10*log10(max(P_active-P_noise,eps)/(P_noise+eps))` | 仅在两类帧都足够且噪声近似平稳时有效；含竞争人声或持续音乐时标记无效，不强行填高分 |
| 频谱噪声 | `flatness=exp(mean(log(P+eps)))/(mean(P)+eps)`；`flux=mean_t(norm(M_t-M_(t-1)))`；低/中/高频带能量各除以总能量 | flatness 高常见于噪声，flux 高可能是瞬态或音乐；均只作描述 |
| 音乐 | 冻结 PANNs/YAMNet/MusiCNN 等分类器的 `p_music`，同时保存 top-k 类别及概率 | 单独报告音乐、乐器、歌声；不能把歌声自动当普通背景噪声 |
| 竞争/重叠人声 | diarization/overlap 模型的 `p_speech`、`p_overlap`、估计说话人数及各自时长 | 用于区分目标声、第二说话人和纯噪声；不是说话人身份结论 |
| 感知质量 | 冻结 DNSMOS `SIG/BAK/OVRL`；可选 SRMR、DRR 或 T60 proxy | 无参考质量诊断可能偏向过度平滑，不可独立决定注册音频 |
| ASR 语义 | `CER_char/CER_pinyin/CER_alias`、`wake_coverage`、`extra_ratio`、`core_hit`、空文本、语言错配、target NLL/q_kw | 用于判断目标、干扰人声、音乐/噪声 ASR 幻觉和混合流 |

每个 SE 视图除绝对值外，还要以相同 raw 为基准计算：

```text
delta_metric = metric(SE) - metric(raw)
```

并按 `stage × split × lang × stream × view × 内容类别` 报告均值、中位数、p05/p95、改善/恶化/相同
数量。不同含义的指标不合成一个任意“noise score”；如确需训练条件 SE 路由，应使用开发集标签重新
校准，并在独立验证集测漏召回。

“一次核对”不等于少报阶段：ASR、VAD、噪声、音乐、embedding 等推理只在 canonical audio 上运行
一次，之后可在任意阶段、stream 和阈值臂重复引用同一特征记录。不得为了估计“稳定性”对同一模型和
同一音频无意义地重复推理；需要多模型一致性时，每个冻结模型各运行一次，并把模型签名作为不同特征键。

上述 SNR、`p_music`、噪声类别、DNSMOS 等可以用于：

- 分析什么类型的音频从 SE 获益；
- 训练或校准“是否值得运行 SE”的计算量路由；
- 解释 SE 失败类型。

它们不能直接作为注册音频纯净度排名，也不能替代说话人评价。推荐先用全流 SE 得到分类型标签：

```text
目标唤醒流 utility =
  最终仍满足低 CER/别名 CER 和核心词覆盖
  且 q_kw/NLL、额外语音、声纹或下游 KWS 至少一项改善
  且其余冻结硬指标不发生不可接受退化

干扰人声/噪声/音乐流 utility =
  目标 ASR 幻觉、竞争人声或噪声证据下降
  且没有伪造出高 q_kw 的目标唤醒词
```

干扰流的 SE 可以在分析上“有效”，却不等于可作为注册音频。最终候选仍需通过低 CER、文本语义、
q_kw 和下游声纹门。再评估噪声检测器能否高召回预测分类型 utility；只有漏掉有效 SE 的比例足够低
时，才把检测器用于线上节省计算，效果优先版本仍可选择所有流运行 SE。

## 7. 第四阶段：形成 s1–s8 的阶段评价

每个固定阶段 `sK` 都应生成下列成对结果：

```text
sK_raw
sK_raw_plus_moss_se48k
sK_raw_plus_spectral
sK_always_moss_se48k（负对照）
```

阶段排行榜不能只看 mean CER。建议字段为：

| 维度 | 指标 |
|---|---|
| 覆盖 | n_uid、missing、errors、hash reuse |
| 文本 | 中英文各自 CER、英文严格/别名 CER、CER0、coverage、extra ratio、core hit、ASR 内容类别、paired improved/worsened/same |
| 基础声学 | duration、RMS/peak、clip/DC/zero ratio、VAD speech/silence ratio |
| 噪声/音乐 | VAD-SNR proxy、flatness、flux、分频带能量、p_music/top-k 类、p_overlap/说话人数、DNSMOS/SRMR |
| SE | 采用率、CER 改善率/退化率及语义类别解释、所有声学指标相对 raw 的 delta、full/chunk 数量 |
| q_kw | mean/分位数、相对 raw 的 paired delta、校准方式 |
| 声纹诊断 | `cos(SE,raw)` 分位数、与其他 delta 的相关性、`gain_ref`、多编码器方向一致性 |
| 整组 KWS | CMD FRR/FAR、pos P10、neg P90、EER、AUC |
| 最终 | extract-main Presence、真实 ASR CER、contest score |

上述声学、噪声、音乐和文本指标必须覆盖 s1–s8 每一阶段的每个 unique raw 及所有 SE 视图，而不是
先选路再检查。去重后的一份音频只核对一次，阶段报告通过 canonical audio id 关联同一结果；报告同时
给出 `n_candidate_refs`、`n_unique_audio`、`feature_cache_hit` 和 `feature_cache_miss`，证明没有重复推理。

还要专门计算每个阶段相对 s1 的互补性：

```text
n_s1_error
n_stage_available_on_s1_error
n_stage_strictly_improves_s1
n_stage_worsens_if_forced
mean_delta_on_s1_error
incremental_CER0
```

这组结果才能回答“s1 应该和哪一个阶段组合”，而不是预先认定一定是 s7。

## 8. 第五阶段：s1、SE 与其他阶段如何组合

### 8.1 先做三个单因素实验

```text
A：s1 raw
B：s1 raw + SE views
C：s1 → 某固定阶段 raw
```

B 回答 SE 是否帮助 s1；C 回答其他分离阶段是否补足 s1。两者通过后再做：

```text
D：s1 raw/SE → 某固定阶段 raw/SE
```

### 8.2 s1 级联规则

推荐的可部署级联为：

1. 在 s1 的全部 raw/SE 记录中，先排除不满足最终低 CER/核心词语义的候选，再按
   `CER → q_kw/NLL → deterministic fallback` 选择；不能仅用配对 CER 变化或 cosine 提前删除记录。
2. 若 s1 最终 CER=0，保持 s1，不触发其他阶段。
3. 若 s1 CER>0 或 `q_kw` 低/不确定，才打开已冻结的补充阶段。
4. 补充阶段只有严格改善 CER，或在 CER 完全相同时显著改善冻结 `q_kw`，才允许切换。
5. 强制切换禁止；没有充分证据时保持 s1。
6. 对每个候选阶段分别做 C/D，再按开发集证据锁定一个阶段和阈值；测试集禁止 `auto` 重选。

当前 spectral 摸底自动选出了：

```text
s7_cv_then_onnx_gate/thr_a
```

它只能作为开发候选。正式实验需要显式设置该标签，并与 s2/s3/s4/s5/s6/s8 的互补增益比较后，才能
决定 s7 是否真的是最佳 s1 fallback。

## 9. 当前实现状态与缺口

### 9.1 已实现

- extract-sep 输入严格审计与 1,838 UID 覆盖检查。
- s1–s8 固定阶段、阈值臂比较。
- 同 UID 全阶段 WAV SHA256 去重排行。
- 中文拼音 CER、英文字符 CER。
- s1→s7 raw/SE 重算、Qwen3-ASR 哈希缓存和 target NLL。
- spectral SE、外部单文件 SE、manifest 批量 SE、预计算 SE。
- extract-main MossFormer2_SE_48K 的一次加载批量适配器。
- raw 路由与 safe-SE 路由导出，以及 CMD cosine 后续入口。

### 9.2 尚需落实

- 将当前 s1+s7 SE 脚本泛化为 s1–s8 任意阶段列表，全流生成 raw/SE 配对报告。
- 增加 resample-only 控制臂，隔离 16k↔48k 重采样影响。
- 将 `CER_se <= CER_raw + 0.05` 从默认硬门改为配对统计；增加 ASR 内容分类、英文冻结别名 CER、
  核心词覆盖和 extra ratio，最终注册候选仍执行低 CER 硬约束。
- `cos(SE,raw)>=0.92` 降级为诊断/待校准灾难门，补齐分位数、相关性、相对 enrollment 的
  `gain_ref` 及多编码器一致性。
- 对每个 canonical raw/SE 一次性计算完整声学、噪声、音乐、竞争人声和感知质量特征，并在所有阶段复用。
- 在独立校准集把 NLL 校准为真正 `q_kw`。
- 为每个 s1–s8 阶段生成互补 s1 的增益表，并据此锁定 fallback。
- 对 finalist 运行冻结 CMD FRR/FAR 和 extract-main Presence/contest。

## 10. 当前 spectral 证据及边界

`D:\gpt\report.json` 是 spectral SE 控制实验，不是 MossFormer2_SE_48K 结果：

| 路由 | mean CER | CER0 |
|---|---:|---:|
| s1 raw | 0.09220103 | 0.84657236 |
| s1 + safe spectral | 0.08489538 | 0.85473341 |
| s1→s7 raw | 0.03563202 | 0.92655060 |
| s1→s7 + safe spectral | 0.03115840 | 0.93471164 |
| always spectral s1 | 0.09009709 | 0.85038085 |

相对 s1→s7 raw，条件 spectral 改善 23 条、恶化 0 条、相同 1,815 条，mean CER 改善
0.00447362；但 `0.03115840 > 0.03`，因此严格结论仍是 `NO_GO`。

该结果说明：

- 保留 raw、让 SE 作为候选视图的方向有效；
- 无条件替换成 SE 不合理；
- s7 对 s1 错误集有明显互补；
- 不能据此判断 MossFormer2_SE_48K 的效果；
- 不能把大量 `<0.92` 的 spectral 样本直接判为说话人错误。

## 11. Go/No-Go

### 11.1 KWS 本地门槛

- coverage=1,838，缺失和错误均为 0。
- selected mean CER ≤0.03。
- CER0 相对冻结基线下降不超过 2 个百分点。
- 最终选路相对冻结基线逐 UID CER 恶化数为 0；若有恶化必须逐条回退。
- 所有 ASR、NLL、SE 和 embedding 输出来自匹配的模型/输入签名。
- 未校准 NLL 不得冒充 `q_kw`。
- 自动挑阶段/阈值仅限开发集；最终测试使用冻结标签。

### 11.2 整组声纹门槛

将候选 best_sep 目录与冻结 raw/s1 基线做同 UID 配对：

- 固定 ERes2NetV2 和语言阈值；
- FAR 优先不升，FRR 不升；或者一项显著改善且另一项不恶化；
- 同时报 pos P10、neg P90、EER、AUC 和置信区间；
- 不能用评估期 CMD 音频反向选择单 UID enroll。

### 11.3 最终采用门槛

把 finalist 送到 extract-main，只替换 enroll，冻结 Presence、CMD、ASR 和阈值，计算：

```text
RR = 1 - FAR
contest = 0.5 * RR + 0.5 * (1 - CER_micro)
```

只有全量覆盖、语言切片无不可接受退化、Presence/FRR/FAR 通过、最终 contest 不降，才允许设置为
默认注册音频。KWS 本地 `LOCAL_PASS` 仍不等于生产批准。

## 12. 推荐执行顺序

```text
P0  审计 s1–s8 输入、时长、index、WAV 和哈希
P1  对每份 unique raw 一次性重算 ASR/CER/NLL、声学、噪声、音乐和竞争人声指标
P2  对全部 unique raw 生成 resample-only、spectral、MossFormer2_SE_48K 视图
P3  对每份 unique SE 输出一次性重算同套指标及声纹诊断，保留全部配对变化并划分 ASR 内容类别
P4  形成每个 s1–s8 阶段 raw vs SE 的配对排行榜；最终候选执行低 CER 和核心词语义硬约束
P5  计算所有阶段相对 s1 的互补增益，锁定一个 fallback 和阈值
P6  运行 s1、s1+SE、s1→fallback、s1+SE→fallback+SE 四臂消融
P7  对 finalist 做冻结 CMD FRR/FAR
P8  到 extract-main 做 Presence、真实 ASR 和 contest 最终验收
P9  效果确认后，再用噪声检测器学习条件 SE，优化算力
```

这个顺序可以同时回答三个问题：哪一个分离阶段最有价值、SE 是否真正改善注册音频、以及噪声检测器
是否值得用于节省计算，同时避免把开发集神谕、未经校准的 cosine 或听感当成生产证据。
