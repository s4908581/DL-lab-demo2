#!/bin/bash
#SBATCH --job-name=dawnbench_test
#SBATCH --partition=a100-test
#SBATCH --gres=shard:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# 1. Use Python 3.11
echo "=== Using system Python 3.11 ==="
PYTHON_CMD=python3
$PYTHON_CMD --version || { echo "Python3 command not available"; exit 1; }

# 2. Create Virtual env
echo "=== Creating virtual environment ==="
$PYTHON_CMD -m venv cifar_venv || { echo "Failed to create virtual environment"; exit 1; }

# 3. activate venv
echo "=== Activating virtual environment ==="
source cifar_venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# 4. upgrade pip
echo "=== Upgrading pip and setting mirror ==="
pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 5. install dependencies
echo "=== Installing dependencies ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || { echo "Failed to install PyTorch"; exit 1; }
pip install numpy matplotlib || { echo "Failed to install other dependencies"; exit 1; }

# 6. set training env
echo "=== Setting environment variables ==="
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=1
export LOCAL_RANK=0
export RANK=0

# 7. run training script
echo "=== Starting training ==="
python -u train_cifar.py || { echo "Training script execution failed"; exit 1; }

# 8. cleaning
echo "=== Cleaning up environment ==="
deactivate
echo "=== Job completed ==="
