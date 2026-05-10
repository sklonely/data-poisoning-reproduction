# OSU HPC (SLURM) Skill

## Overview
Oregon State University College of Engineering HPC cluster access skill.
Handles the full chain: Local → Gateway → Submit Node → Compute Node.

## Infrastructure

| Node | Hostname | Alias |
|------|----------|-------|
| Gateway | `access.engr.oregonstate.edu` (flip1-4) | `osu-engr` in `~/.ssh/config` |
| Submit | `submit.hpc.engr.oregonstate.edu` (submit-a/b/c) | — |
| Compute | `cn-*.hpc.engr.oregonstate.edu` | Assigned by SLURM |

- **User**: `chanc7`
- **Group**: `eecs` / `upg147111`
- **Home (NFS shared)**: `/nfs/stak/users/chanc7` — same path on ALL nodes
- **SSH key**: `~/.ssh/id_ed25519`

## SSH Config (`~/.ssh/config`)

```
Host osu-engr
  HostName access.engr.oregonstate.edu
  User chanc7
  IdentityFile ~/.ssh/id_ed25519
  AddKeysToAgent yes
  UseKeychain yes
  IdentitiesOnly yes
```

For ProxyJump to submit node, add:
```
Host osu-submit
  HostName submit.hpc.engr.oregonstate.edu
  User chanc7
  ProxyJump osu-engr
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

## Connection Patterns

### 1. Connect to Gateway
```bash
ssh osu-engr
```

### 2. Connect to Submit Node (via Gateway)
```bash
ssh -J osu-engr chanc7@submit.hpc.engr.oregonstate.edu
# or if osu-submit alias configured:
ssh osu-submit
```

### 3. Interactive Job on Compute Node
```bash
# From submit node — eecs partition (user has access)
srun --partition=eecs --account=eecs --nodes=1 --ntasks=1 --time=60 --pty bash

# share partition (open to all, default)
srun --partition=share --nodes=1 --ntasks=1 --time=60 --pty bash
```

### 4. Run Command on Compute Node (non-interactive)
```bash
ssh -J osu-engr chanc7@submit.hpc.engr.oregonstate.edu \
  "srun --partition=eecs --account=eecs --nodes=1 --ntasks=1 --time=5 bash -c 'hostname; <your_command>'"
```

### 5. File Transfer (Local ↔ HPC)
```bash
# Local → HPC
scp -o "ProxyJump osu-engr" myfile.txt chanc7@submit.hpc.engr.oregonstate.edu:~/

# HPC → Local
scp -o "ProxyJump osu-engr" chanc7@submit.hpc.engr.oregonstate.edu:~/scratch/file.txt /local/path/

# rsync (recommended for directories)
rsync -avz -e "ssh -J osu-engr" ./local_dir/ chanc7@submit.hpc.engr.oregonstate.edu:~/remote_dir/
```

## SLURM Commands

```bash
# Check partitions and available nodes
sinfo --state=idle --format='%P %a %D %C %l'

# Submit batch job
sbatch myjob.sh

# Submit interactive job
srun --partition=eecs --account=eecs --nodes=1 --ntasks=1 --time=60 --pty bash

# Check queue
squeue -u chanc7
# or
sq

# Cancel job
scancel <jobid>
scancel -u chanc7   # cancel all my jobs

# Node status
nodestat eecs
showjob <jobid>
```

## Batch Script Template

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --account=eecs
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ── Environment ──
export HPC_SHARE=/nfs/hpc/share/chanc7
export HF_HOME=${HPC_SHARE}/hf_cache
export HF_HUB_CACHE=${HPC_SHARE}/hf_cache
export TORCH_HOME=${HPC_SHARE}/cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate project venv (uv-managed, Python 3.12)
source /path/to/your/venv/bin/activate

echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# your commands here
python train.py
```

Submit with: `sbatch myjob.sh`

### Array Jobs（平行多實驗）

```bash
#SBATCH --array=0-3          # 4 parallel tasks (indices 0,1,2,3)

case $SLURM_ARRAY_TASK_ID in
  0) EXPERIMENT="exp_a" ;;
  1) EXPERIMENT="exp_b" ;;
  2) EXPERIMENT="exp_c" ;;
  3) EXPERIMENT="exp_d" ;;
esac

python train.py --experiment $EXPERIMENT
```

## Available Partitions (re-verified 2026-05-04, chanc7 實測 — submit + read GPU model on actual node)

| Partition | GPU 型號 | Compute Cap | VRAM | Time Limit | GPU/節點 | 可用？ | 備註 |
|-----------|---------|-------------|------|-----------|---------|--------|------|
| `dgxh` | H100 40G(×16) / 80G(×8) | 9.0 | 40–80 GB | 2 天 | 8–16 | ✅ submit OK | 最頂級；queue 排很久；TF 2.10/2.15 對 H100 支援邊緣 |
| `dgx2` | V100-SXM3 | 7.0 | 32 GB | 7 天 | 16 | ✅ submit OK | 我們的主力；TF 2.15 + CUDA 12.2 已驗證 |
| `ampere` | A40 | 8.6 | 46 GB | 2 天 | 2 | ✅ submit OK | 同硬體 = share, preempt, classgpu, nacse 共享 11 個 cn-r/cn-s 節點 |
| `share` | A40（GPU node 部分）+ 多種 CPU node | 8.6 | 46 GB | 14 天 | 2（GPU node） | ✅ submit OK | 同 ampere 的 A40 硬體但 time limit 拉到 14 天 |
| `gpu` | Quadro RTX 8000 | 7.5 | 46 GB | 7 天 | 8 | ✅ submit OK | eecs group，性能不錯 VRAM 大 |
| `eecs` | RTX 2080 Ti | 7.5 | 11 GB | 7 天 | 8 | ✅ submit OK | VRAM 小，不夠跑大 batch；可用做 victim |
| `preempt` | RTX 2080 Ti / 混合 | varies | varies | 7 天 | varies | ✅ submit OK | **可被高優先 job 砍掉** — 不適合長 job |
| `eecs3` | — | — | — | 7 天 | — | ❌ Invalid account | — |
| `classgpu` | gpu:14/16/8 等大節點 | — | — | 1 天 | varies | ❌ Invalid account | 課程用 |
| `nacse` | A40 等 | — | — | — | — | ❌ Invalid account | — |

**節點共享拓撲**（重要！）：
- `cn-r-1..6`、`cn-s-1..5` 這 11 個 A40 節點 (Gres=gpu:2) **同時屬於 share、classgpu、ampere、nacse、preempt** 五個 partition
- 從不同 partition submit 等於用不同優先序 / time limit / preemption 規則競爭同樣的硬體
- ampere 跟 share 是我們唯二用得到的 path：ampere = 短期 (≤2 天)，share = 長期 (≤14 天)

**dgxh-1**: 16 × H100-40GB, 224 CPU, 2 TB RAM (Sapphire Rapids)
**dgxh-2/3/4**: 8 × H100-80GB, 224 CPU, 2 TB RAM (Sapphire Rapids)
**dgx2-2..5**: 16 × V100-SXM3-32GB, 96 CPU, 1.5 TB RAM
**ampere/share A40 節點**: 2 × A40, 48 CPU, 252 GB RAM

### 使用 H100 指令

```bash
# 互動式（1 張 H100，任意）
srun --partition=dgxh --gres=gpu:1 --nodes=1 --ntasks=1 --time=120 --pty bash

# 指定 80GB 版本（dgxh-2/3/4）
srun --partition=dgxh --gres=gpu:1 --constraint=vram80g --nodes=1 --ntasks=1 --time=120 --pty bash

# 16 張 H100-40G 節點（dgxh-1）
srun --partition=dgxh --gres=gpu:1 --constraint=h100-40g --nodes=1 --ntasks=1 --time=120 --pty bash
```

## Storage

### 儲存階層

| 路徑 | 用途 | 持久性 | 備註 |
|------|------|--------|------|
| `/nfs/stak/users/chanc7` (Home) | SSH config、dotfiles、小型設定檔 | ✅ 持久 | NFS 共享，所有節點可見。**不放大型資料或模型** |
| `/nfs/hpc/share/chanc7/` | 主要工作目錄：repo、venv、outputs、模型快取 | ⚠️ 定期清除 | 1.5 TB，NFS 共享。**管理員會定期清理，資料不可長期依賴** |
| 節點本機 `/scratch` | 訓練期間的臨時 I/O 密集操作 | ❌ Job 結束即清除 | 不跨節點共享 |

> **⚠️ share 空間會被定期清除**：網路管理人員可能在不通知的情況下清理 `/nfs/hpc/share/` 的資料。因此：
> 1. **腳本必須具備環境自我修復能力**（偵測 venv/repo/cache 不存在時自動重建）
> 2. **不可將 share 當作長期儲存**（重要結果應及時拉回本機）
> 3. **模型快取視為可重建**（腳本應能自動重新下載）

### 建議的 share 目錄結構

```
/nfs/hpc/share/chanc7/
├── <project>/
│   ├── repo/          # git clone（可重建）
│   ├── venv/          # uv venv, Python 3.12（可重建）
│   └── outputs/       # training outputs — 完成後盡快拉回本機
├── hf_cache/          # HuggingFace models & tokenizers（可重新下載）
└── cache/
    ├── torch/         # PyTorch hub cache
    ├── uv/            # uv package cache
    └── pip/           # pip wheel cache
```

### 環境變數（在 batch script 中設定）

```bash
export HPC_SHARE=/nfs/hpc/share/chanc7
export HF_HOME=${HPC_SHARE}/hf_cache
export HF_HUB_CACHE=${HPC_SHARE}/hf_cache
export TORCH_HOME=${HPC_SHARE}/cache/torch
```

> **原則**：Home 只放 config，所有大型檔案（模型、資料、輸出）放 `/nfs/hpc/share/`，但視為暫存——完成的結果必須及時拉回本機保存。

## Environment Setup

不使用 `module load`，改用 `uv` 管理 Python 環境。

### 初次建立 venv

```bash
# 在 submit node 或 compute node 上執行
export HPC_SHARE=/nfs/hpc/share/chanc7
PROJECT_DIR=${HPC_SHARE}/<project_name>

mkdir -p ${PROJECT_DIR}/{repo,venv,outputs}
cd ${PROJECT_DIR}/repo
git clone <repo_url> . 

# 建立 venv（uv 已安裝於 ~/.local/bin/uv，NFS 共享）
uv venv --python 3.12 ${PROJECT_DIR}/venv
source ${PROJECT_DIR}/venv/bin/activate
uv sync                    # 從 pyproject.toml 安裝依賴
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### Batch Script 中啟用環境

```bash
source /nfs/hpc/share/chanc7/<project>/venv/bin/activate
```

> **注意**：share 空間可能被清除，腳本應在啟動時檢查 venv 是否存在，不存在則自動重建。
> 典型做法是寫一個 `env.sh` 提供 `ensure_env()` 函數，在 batch script 開頭 source。

### 預下載模型（submit node 有網路）

```bash
# 在 submit node 上執行（有外網）
export HF_HOME=/nfs/hpc/share/chanc7/hf_cache
python -c "from transformers import AutoModel; AutoModel.from_pretrained('model_name')"
```

Compute node 可以下載資源，但建議在 submit node 預先快取，確保訓練不會因網路問題中斷。

## First-Time Setup

If host key verification fails for submit node:
```bash
# From local: scan submit node keys through gateway, add to known_hosts
ssh osu-engr "ssh-keyscan -H submit.hpc.engr.oregonstate.edu 2>/dev/null" >> ~/.ssh/known_hosts
```

## Verified Workflow (2026-04-14)

Full chain test was successful:
1. Local → `access.engr.oregonstate.edu` (flip4) via `ssh osu-engr`
2. Gateway → `submit.hpc.engr.oregonstate.edu` (submit-a) via ProxyJump
3. Submit → `cn-gpu3.hpc.engr.oregonstate.edu` via `srun --partition=eecs`
4. Created `~/scratch/hello.md` on compute node
5. File visible at gateway (NFS shared home)
6. SCP'd back to local via ProxyJump

## 合規規定（Compliance）

### ✅ 允許
- 學術研究與教學用途
- 自行安裝軟體到 home 目錄或 container（不需 sudo）
- 使用 Conda/pip/Singularity 管理環境
- 開源或具合法授權的商業軟體

### ❌ 禁止 / 需注意

**資源使用**
- 不能壟斷資源（fair share 原則）— 單用戶上限 1000 工作 / 400 同時執行
- 不能在 gateway/submit node 直接跑 CPU 密集計算（必須透過 SLURM）
- 資料密集的 I/O 應用本機 `/scratch`，避免 NFS 壅塞

**網路與安全**
- 禁止網路掃描、IP/MAC 欺騙、未授權連線
- 禁止架設個人 web server、IRC server、file-sharing 服務
- 禁止參與非 OSU 的分散式計算（如 SETI@home、folding@home）
- **Compute node 網路限制**：可以**下載**外部資源（pip install、git clone、model download），但**不可主動向外傳輸資料**。要取回 HPC 上的結果，必須從本機端主動拉取（scp/rsync via ProxyJump），HPC 端無法推送到外部

**軟體**
- 禁止安裝出口管制（ITAR/EAR）軟體，需先通過 COE IT 審查（14 天）
- 禁止將學生授權軟體安裝於共享系統
- 禁止破解、逆向工程授權保護

**資料**
- FERPA（學生資料）/ HIPAA（健康資料）/ 出口管制資料：**不得存放在 HPC**
- 敏感資料完整政策需登入查閱：https://uit.oregonstate.edu/infosec

**商業用途**
- 隱性禁止非學術商業用途（instruction/research 優先）

### ⚠️ 監控聲明
登入時出現的 banner 具法律效力：
> "Use Constitutes a Consent to Monitoring. Users have No Expectation of Privacy."

OSU 在 server network 安裝監控工具，所有活動可被記錄。

### 違規後果
- 帳號立即停用
- 必須與 assistant dean 面談
- 嚴重者移交學生事務處

### 尚未確認（需 OSU 帳號才能查）
- 詳細資料分類等級（Restricted / Sensitive / Public）
- HIPAA/FERPA 的 HPC 具體規範
- 出口管制軟體完整審核流程

聯絡：COE IT `coe.support@oregonstate.edu` 或 OSU OIS https://uit.oregonstate.edu/infosec

---

## Usage Instructions (for Claude)

When the user asks to run something on OSU HPC:

1. **Check connectivity**: `ssh -o BatchMode=yes osu-engr "hostname"`
2. **Choose partition**: `dgxh`（H100，預設首選）for GPU jobs, `eecs` for lightweight jobs, `share` for general use
3. **For quick tasks**: Use `srun` via ProxyJump SSH one-liner
4. **For batch jobs**: Write `myjob.sh`, scp to HPC, then `sbatch myjob.sh`
5. **File transfer**: 從本機主動拉取 — SCP/rsync with `-o "ProxyJump osu-engr"` or `-J osu-engr`（HPC 無法推送到外部）
6. **Monitor**: `ssh -J osu-engr chanc7@submit.hpc.engr.oregonstate.edu "squeue -u chanc7"`
7. **Environment**: 使用 `uv` + Python 3.12 venv，不使用 `module load`。腳本應能偵測並自動重建被清除的環境
8. **Results**: 訓練完成後盡快拉回本機，share 空間不保證長期保留
