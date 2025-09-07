#!/bin/bash
#SBATCH --job-name=dawnbench_test
#SBATCH --partition=a100-test
#SBATCH --nodes=1
#SBATCH --gres=shard:2            # 请求2个GPU shard（A100的一半资源）
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         # 使用4个vCPU核心
#SBATCH --time=00:20:00           # 20分钟时间限制
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# 加载必要模块
module load cuda/11.8
module load python/3.10

# 创建虚拟环境
python -m venv cifar_venv
source cifar_venv/bin/activate

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib

# 设置分布式训练环境变量
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=1
export LOCAL_RANK=0
export RANK=0

# 运行训练脚本
python -u train_cifar.py

# 清理
deactivate
