## 🖥️ Instance Specs

- **GPU**: Lambda Cloud GH200 (aarch64)
- **RAM**: ~96 GB
- **CUDA Toolkit**: 12.8+

---

## ⚙️ Installation

```bash
# Go to home directory
cd ~

# Download Miniforge installer
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh

# Run the installer
bash Miniforge3-Linux-aarch64.sh

# Apply changes (or use ~/.zshrc if you use zsh)
source ~/.bashrc

# Verify Conda is installed
conda --version

git clone https://github.com/adityanagachandra/lerobot.git
cd lerobot

# Create a new Conda env with Python 3.10
conda create -y -n lerobot python=3.10

# Activate it
conda activate lerobot

# Install FFmpeg from conda-forge
conda install -y ffmpeg -c conda-forge

# Install LeRobot in editable mode
pip install -e .

# Uninstall any existing Torch builds
pip uninstall -y torch torchvision torchaudio torchcodec 2>/dev/null || true
conda remove -y pytorch torchvision torchaudio pytorch-cuda 2>/dev/null || true

# Install nightly Torch built for CUDA 12.8
pip install --pre --upgrade \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  torch torchvision
```

## 🚀 Training

### train.py configs
| Parameter     | Type & Default  | Description               |
| ------------- | --------------- | ------------------------- |
| `num_workers` | `int = 12`      | Data loading processes    |
| `batch_size`  | `int = 24`      | Samples per batch         |
| `steps`       | `int = 100_000` | Total training steps      |
| `eval_freq`   | `int = 20_000`  | Evaluate every N steps    |
| `log_freq`    | `int = 200`     | Log metrics every N steps |


