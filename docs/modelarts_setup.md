# 华为云 ModelArts 环境配置与迁移记录

本文记录 DCASE 2026 Task 6 基线项目在华为云 ModelArts Notebook 上的可复现配置流程。命令以 2026-10-09 的实际部署为准。

## 1. 已验证的云端规格

- GPU：NVIDIA Tesla T4，16 GB 显存（系统显示 15,360 MiB）
- CPU：8 vCPU
- 内存：28 GiB
- 系统：Ubuntu 22.04，x86_64
- ModelArts 镜像：PyTorch 2.1.0
- 镜像内 Python：3.9.11
- 镜像内 PyTorch：2.1.0+cu121
- torchvision：0.16.0+cu121
- torchaudio：2.1.0+cu121
- NVIDIA 驱动：550.144.03

该配置可以直接训练本项目的 QD-DETR。CLAP 特征已经预提取，因此训练阶段不需要加载或训练 CLAP 模型。

## 2. 存储规划

ModelArts 中不同挂载点的用途不同：

- `/home/ma-user/work`：EVS 持久化磁盘。代码、环境、数据和实验结果都应放在这里。
- `/cache`：宿主机高速临时盘，容量大，但实例停止、迁移后可能清空。
- `/data`：在本次实例中是 tmpfs，会占用内存，不能用于存放数据集。
- `/` 和 `/modelarts`：系统空间，不用于项目数据。

Clotho-Moment 两个压缩包约 22.7 GB，解压后约 22.9 GiB。如果同时保留压缩包和解压结果，建议将 EVS 扩容到至少 80 GB，推荐 100 GB。

检查磁盘：

```bash
df -h /home/ma-user/work /cache
```

## 3. 代码迁移

本地仓库使用两个 remote：

```bash
git remote rename origin upstream
git remote add origin https://github.com/<YOUR_USER>/dcase2026_task6_baseline.git
git push -u origin main
```

其中：

- `origin`：自己的 GitHub fork，用于 push。
- `upstream`：原作者仓库，用于跟踪官方更新。

云端首次部署：

```bash
cd /home/ma-user/work
git clone --depth 1 https://github.com/<YOUR_USER>/dcase2026_task6_baseline.git
cd dcase2026_task6_baseline
```

使用 `--depth 1` 是因为上游当前版本追踪了约 85 MB 的官方 checkpoint，完整历史克隆会明显更慢。后续更新执行：

```bash
git pull --ff-only
```

## 4. 创建持久化 Python 环境

不要直接修改 ModelArts 自带的 PyTorch 环境，也不建议使用 `conda create --clone`。实际部署时，Conda 克隆会访问华为内部 Conda 镜像，并曾因 HTTP 502 失败、留下约 5 GB 的不完整目录。

推荐创建复用系统 PyTorch/CUDA 的持久化 venv：

```bash
mkdir -p /home/ma-user/work/venvs

/home/ma-user/anaconda3/envs/PyTorch-2.1.0/bin/python \
  -m venv --system-site-packages \
  /home/ma-user/work/venvs/dcase-task6

source /home/ma-user/work/venvs/dcase-task6/bin/activate
```

安装项目依赖。使用清华 PyPI 镜像可获得较稳定的国内下载速度：

```bash
python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --upgrade pip

python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  easydict==1.13 \
  numpy==1.26.4 \
  packaging==26.1 \
  pandas==2.3.3 \
  PyYAML==6.0.3 \
  scikit-learn==1.6.1 \
  tqdm==4.67.3
```

这里不重新安装 `torch`、`torchvision` 和 `torchaudio`，而是复用镜像内已经验证可用的 CUDA 12.1 版本。

验证环境：

```bash
python -m torch.utils.collect_env
```

输出中应包含：

```text
PyTorch version: 2.1.0+cu121
Is CUDA available: True
GPU 0: Tesla T4
```

## 5. 下载 Clotho-Moment 特征

安装 Hugging Face CLI：

```bash
source /home/ma-user/work/venvs/dcase-task6/bin/activate

python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  huggingface_hub
```

从 HF Mirror 下载：

```bash
cd /home/ma-user/work/dcase2026_task6_baseline

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/ma-user/work/hf-cache
export HF_HUB_CACHE=/home/ma-user/work/hf-cache/hub
export HF_XET_CACHE=/home/ma-user/work/hf-cache/xet
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600

mkdir -p downloads/clotho-moment

hf download \
  lighthouse-emnlp2024/Clotho-Moment_CLAP_features \
  clap.tar.gz clap_text.tar.gz \
  --repo-type dataset \
  --local-dir downloads/clotho-moment
```

命令中断后可原样重跑，Hugging Face 本地元数据会用于断点恢复。

如果 ModelArts 共享代理上的单连接下载过慢，可以改用仓库提供的并行分片脚本。它会下载、拼接、解压并校验文件数量；中断后原样重跑即可从已有分片继续：

```bash
cd /home/ma-user/work/dcase2026_task6_baseline
bash scripts/download_clotho_parallel.sh
```

## 6. 解压与校验

```bash
cd /home/ma-user/work/dcase2026_task6_baseline
mkdir -p features/clotho-moment

tar -xzf downloads/clotho-moment/clap.tar.gz \
  -C features/clotho-moment

tar -xzf downloads/clotho-moment/clap_text.tar.gz \
  -C features/clotho-moment
```

校验文件数量：

```bash
find features/clotho-moment/clap -name '*.npz' | wc -l
find features/clotho-moment/clap_text -name '*.npz' | wc -l
```

正确结果应分别为：

```text
51240
44261
```

项目配置 `config_pretraining.yml` 期望的路径正是：

```text
features/clotho-moment/clap
features/clotho-moment/clap_text
```

## 7. 训练前检查与启动

进入环境和项目目录：

```bash
source /home/ma-user/work/venvs/dcase-task6/bin/activate
cd /home/ma-user/work/dcase2026_task6_baseline
```

第一次不要直接训练 200 轮。复制配置，将 `n_epoch` 改为 1，并将结果写入独立目录，先测量单轮时间、显存和验证指标。

正式长时间训练建议使用 `nohup` 或 `tmux`，同时关闭或延长 ModelArts 自动停止时间。例如：

```bash
nohup python src/train.py --config config_pretraining.yml \
  > pretraining.out 2>&1 &
```

监控：

```bash
tail -f pretraining.out
nvidia-smi
```

训练输出和 checkpoint 必须保存在 `/home/ma-user/work` 下。浏览器关闭不会终止 `nohup` 任务，但 Notebook 实例自动关机后任务仍会停止。

本项目每轮保存 `latest_checkpoint.pth`。实例重启后使用下面的命令精确恢复模型、优化器、学习率调度器、轮次和最佳验证分数：

```bash
python src/train.py \
  --config config_pretraining.yml \
  --resume-training results_pretraining/latest_checkpoint.pth
```

`--resume-training` 用于恢复同一个被中断的训练实验；原有 `--resume` 参数只加载模型权重，保留用于从预训练模型开始新的微调实验。两者不能同时使用。

## 8. SSH 注意事项

SSH 使用 PEM 私钥认证，不使用密码。`SHA256:...` 通常是公钥或主机密钥指纹，不是登录密码。

Windows OpenSSH 会拒绝权限过宽的私钥。如果出现 `UNPROTECTED PRIVATE KEY FILE`，需要在文件属性中移除 `Everyone` 的访问权限，仅保留当前用户、SYSTEM 和 Administrators。不要提交或上传 PEM 私钥到 GitHub。

## 9. 本次部署遇到的问题

1. ModelArts 内置 Conda 镜像返回 HTTP 502，导致 `conda create --clone` 失败。
   - 解决：改用 `venv --system-site-packages`，复用已验证的 PyTorch/CUDA。
2. GitHub 完整克隆速度慢。
   - 原因：仓库历史较大，且共享公网代理速度波动。
   - 解决：云端首次使用 `git clone --depth 1`。
3. Notebook 默认持久化盘只有 5 GB。
   - 解决：将 EVS 在线扩容到约 100 GB，并只在 `/home/ma-user/work` 保存重要文件。
4. `pip` 会提示 ModelArts SDK 与新版 pandas/tqdm 的依赖冲突。
   - 本项目不调用 ModelArts Python SDK；依赖安装在独立 venv 中，不修改系统 PyTorch。训练前仍应通过 1 epoch 冒烟实验验证项目运行。
