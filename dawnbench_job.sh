#!/bin/bash
#SBATCH --job-name=dawnbench_test
#SBATCH --partition=a100-test
#SBATCH --gres=shard:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# 1. Load necessary modules
echo "=== Loading necessary modules ==="
module purge
module load cuda/11.8
module load python/3.10  # Explicitly load Python module

# 2. Check if Python is available
echo "=== Checking Python version ==="
python --version || { echo "Python command not available"; exit 1; }

# 3. Create virtual environment
echo "=== Creating virtual environment ==="
python -m venv cifar_venv || { echo "Failed to create virtual environment"; exit 1; }

# 4. Activate virtual environment
echo "=== Activating virtual environment ==="
source cifar_venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# 5. Check if pip is available
echo "=== Checking pip version ==="
pip --version || { echo "pip command not available"; exit 1; }

# 6. Install dependencies
echo "=== Installing dependencies ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || { echo "Failed to install PyTorch"; exit 1; }
pip install numpy matplotlib || { echo "Failed to install other dependencies"; exit 1; }

# 7. Set up distributed training environment variables
echo "=== Setting environment variables ==="
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=1
export LOCAL_RANK=0
export RANK=0

# 8. Run training script
echo "=== Starting training ==="
python -u train_cifar.py || { echo "Training script execution failed"; exit 1; }

# 9. Clean up
echo "=== Cleaning up environment ==="
deactivate
echo "=== Job completed ==="
