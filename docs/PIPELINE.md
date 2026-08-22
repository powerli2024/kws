# Pipeline: extract@sep then kws

kws does **not** run MossFormer. Redo all BSS first in
[powerli2024/extract](https://github.com/powerli2024/extract) branch **`sep`**.

```text
datasetA/{pos,neg}.jsonl + kws_*.wav
        │
        ▼
extract@sep   ./run_sep.sh
        │     s1–s8 Qwen CER oracle, no MMS-FA, no duration skip-sep
        ▼
$VM_OUT/kws_handoff.json
$VM_OUT/{pos,neg}/s*/index.jsonl
$VM_OUT/best_sep/index.jsonl + {pos,neg}/{uid}.wav
        │
        ▼
kws   rebuild_best_sep → analyze_dual_zero → T0–T4 → Presence veto
```

Contest Presence / mix ASR is a **separate clone**: `/root/extract` on **`main`**. This BSS clone is `/root/extract-sep`. Do not `git checkout` between them. They share `/root/autodl-tmp` and conda env `ve`.

## AutoDL

```bash
cd /root
git clone -b sep https://github.com/powerli2024/extract.git extract-sep
cd /root/extract-sep && chmod +x *.sh run_sep.sh pick_python.sh
export DATA_DIR=/root/autodl-tmp/datasetA
export VM_OUT=/root/autodl-tmp/kws_sep
export MOSS_CKPT_DIR=/root/autodl-tmp/checkpoints
export ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B
./setup_env.sh          # 装进 conda env ve（与 /root/extract 同一 PYTHON_BIN）
source ./env.sh         # 或: conda activate ve && source ve/.env_ve
./download_models.sh
./check_env.sh
./run_sep.sh --limit 20
./run_sep.sh
# enroll for main VE:
mkdir -p /root/autodl-tmp/pos_neg
ln -sfn /root/autodl-tmp/kws_sep/best_sep /root/autodl-tmp/pos_neg/best_sep
```

Then here:

```bash
python scripts/rebuild_best_sep.py --pos-neg /root/autodl-tmp/kws_sep
python scripts/analyze_dual_zero.py --pos-neg /root/autodl-tmp/kws_sep
python scripts/run_t0_t4.py --pos-neg /root/autodl-tmp/kws_sep
```

Handoff contract: extract `KWS_HANDOFF.md`. `mms_fa` must be false.
Selector: within-stage min CER, original wins ties; across-stage min CER, prefer non-original on ties.
