🚀 LeRobot Setup on Lambda GH200 (CUDA 12.8, ARM64)
This guide walks through setting up the LeRobot repository on a Lambda Cloud GH200 machine (ARM64, CUDA 12.8).

Component	Spec
GPU	NVIDIA GH200 (Hopper, SM120)
CPU	ARM64
RAM	 ≈ 96 GB
OS	 Linux (ARM64)

📦 Installation
1  Install Miniforge (ARM64)
bash
Copy
Edit
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh
source ~/.bashrc   # or ~/.zshrc if you use Zsh
conda --version    # sanity‑check
### 2  Clone LeRobot

bash
Copy
Edit
git clone https://github.com/lerobot/lerobot.git
cd lerobot
### 3  Create & Activate Environment

bash
Copy
Edit
conda create -y -n lerobot python=3.10
conda activate lerobot
### 4  Install Core Deps

bash
Copy
Edit
conda install -y ffmpeg -c conda-forge
pip install -e .
### 5  Remove Any Existing Torch

bash
Copy
Edit
pip uninstall -y torch torchvision torchaudio torchcodec 2>/dev/null || true
conda remove   -y pytorch torchvision torchaudio pytorch-cuda 2>/dev/null || true
### 6  Install PyTorch Nightly (CUDA 12.8)

bash
Copy
Edit
pip install --pre --upgrade \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  torch torchvision
✅ Sanity Check
bash
Copy
Edit
python - <<'PY'
import torch, os, sys
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
x = torch.randn(4096, 4096, device='cuda')
torch.cuda.synchronize()
print("Max abs:", (x @ x.T).abs().max())
PY
You should see the correct Torch nightly version (≥ 2.9.0.dev*) and CUDA: 12.8.

🏁 Training Configuration
Typical train.py snippet:

yaml
Copy
Edit
num_workers: 16
batch_size: 48
steps: 100_000
eval_freq: 20_000
log_freq: 200
These defaults assume plenty of RAM/VRAM; feel free to tune.

🚀 Launch Training
bash
Copy
Edit
python -m ts.train --config-path=configs/train/<your_config>.yaml
Replace <your_config>.yaml with the actual config file for your task.

🧠 Pro Tips
Use watch -n1 nvidia-smi to monitor GPU memory and temps.

Save checkpoints every 10‑20 k steps to guard against interruptions.

WANDB logging at log_freq=200 is a good balance between detail and overhead.

🔗 Resources
LeRobot repo: https://github.com/lerobot/lerobot

PyTorch nightly wheels: https://download.pytorch.org/whl/nightly/

Lambda Cloud docs: https://lambdalabs.com/service/cloud-gpu
