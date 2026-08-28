# s1 → s7 → SE 技术方案

## 结论

SE 可以帮助选轨，但不能把“听起来更干净”直接当成更好的注册音频。本方案把每条候选拆成
`raw` 和 `se` 两个视图，重新计算相同冻结 ASR 的注册文本 CER，并计算
`cos(embed(se), embed(raw))`。只有文本没有明显回退、声纹也没有坍塌时，SE 视图才有资格
参与选择。原始音频始终保留，因此 SE 不会被强制采用。

该实验是新增分支，不改变现有 T0–T4 冻结基线，也不在 `kws` 内重新训练分离模型。

## 独立选路

对每个 UID 分别读取指定的 s1 和 s7 index，不能用 best_sep 的跨阶段神谕结果冒充在线选路：

1. s1 的所有流先分别比较；中文用无调拼音 CER，英文用规范化字符 CER。
2. 若 s1 最优 CER 为 0，保持 s1，不运行 s7 切换。
3. 若 s1 最优 CER 大于 0，才查看该 UID 的 s7 候选。
4. s7 只有严格改善 CER 才切换；CER 完全相同时，可用已知注册文本的 target NLL 破同分。
5. raw 与通过安全门的 SE 视图一起参与同一规则；相同 CER/NLL 时保守选择 raw、s1、original。

`--s7-arm auto` 只用于开发集摸底：它按覆盖优先、原有 CER 次优自动锁定一个 s7 arm。
冻结复验必须把报告中选出的准确标签写入 `S7_ARM`，再从空工作目录重跑，不能在测试集重新挑 arm。

## SE 接入

仓库提供三种方式：

- `spectral`：无需模型，用于验证整个闭环和建立负/控制组，不作为最终质量方案。
- `command`：推荐的神经 SE 接口。命令模板必须含 `{input}` 和 `{output}`，例如
  `python /root/cmd_se/infer.py --input {input} --output {output}`。命令按 argv 执行，不经过 shell 展开。
- `command` 的全量推荐形式是 `SE_BATCH_COMMAND='python infer_manifest.py --manifest {manifest}'`。
  manifest 每行包含 `input`、`output`、`audio_sha256`、`sample_rate=16000` 和
  `length_policy=full_waveform`；包装器只加载一次模型，处理完全部行后退出。
- `precomputed`：复用已离线生成的 SE 文件，便于比较 CMD-SE、DeepFilterNet、FRCRN 等模型。

所有长度音频都处理，不裁成 3 秒或 6 秒；SE 输出与输入长度允许最多 2% 或 0.1 秒差异，超出立即失败。
源音频永不覆盖，输出按原音频 SHA256 去重。

## 重算、缓存和评价

主入口是 `scripts/run_se_recompute.sh`。它会先严格审查 extract-sep 输入，再执行：

1. 对 s1/s7 的每条 raw 音频生成 SE 视图；字节相同的音频只增强一次。
2. 用 Qwen3-ASR 对 `(audio_sha256, wake_text, lang)` 只转写一次，消除同一音频重复 ASR 得到不同 CER 的问题。
3. 可选重算 target NLL；它只在精确最小 CER 内排序，未校准前不是概率 `q_kw`，不能使用绝对阈值。
4. 用 ERes2NetV2 计算 raw/SE 声纹余弦；默认安全门为余弦不低于 0.92，且
   `CER_se <= CER_raw + 0.05`。
5. 同时报告 s1 raw、s1+SE、s1→s7 raw、s1→s7+SE、always-SE 负对照及逐 UID 配对差值。

AutoDL 全量运行：

```bash
cd /root/kws
POS_NEG=/root/autodl-tmp/kws_sep \
ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B \
WORK_DIR=/root/autodl-tmp/kws_se_route \
S7_ARM=你的冻结s7标签 \
SE_BACKEND=command \
SE_BATCH_COMMAND='python /root/kws/scripts/extract_main_se48k_manifest.py --manifest {manifest} --extract-main /root/extract-main --clearvoice-root /root/autodl-tmp/ClearerVoice-Studio' \
WITH_NLL=1 RESUME=1 \
bash scripts/run_se_recompute.sh
```

第一次摸底可省略 `S7_ARM`，也可用 `SE_BACKEND=spectral` 检查整个管线。中断后保持相同模型、输入和参数，
设置 `RESUME=1` 即可复用哈希评分；签名不一致会拒绝混用旧结果。

设置 `DATA_DIR=/root/datasetA` 后，shell 会使用同次运行导出的 `best_sep_s1_to_s7_raw` 作为基线，
继续生成 CMD 配对评价；也可用 `BASELINE_DIR` 覆盖该路径。不同 SE 模型必须使用不同 `WORK_DIR`，
缓存签名会拒绝把 spectral、CMD-SE 等输出混在一起。

## 验收和 Go/No-Go

KWS 本地通过必须同时满足：1838 UID 全覆盖、相对 s1→s7 raw 的逐条 CER 恶化数为 0、平均 CER
不高于 0.03、CER0 下降不超过 2 个百分点、所有被采用的 SE 视图通过文本和声纹双门。

本地通过只会输出 `LOCAL_PASS_NEEDS_CMD_PRESENCE`，不会输出 production approved。随后将导出的
`best_sep_s1_s7_safe_se` 送入冻结的 CMD FRR/FAR 评价，并送到 extract-main 做 Presence、真实混合 ASR
和竞赛分验证；只有 FRR 或 FAR 改善且另一项不恶化、最终竞赛分不降，才允许成为默认注册音频。
