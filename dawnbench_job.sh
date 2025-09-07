#!/bin/bash
#SBATCH --job-name=dawnbench_cifar10
#SBATCH --partition=a100          # 使用A100分区
#SBATCH --nodes=1                 # 使用1个节点
#SBATCH --gres=gpu:1               # 每节点1个GPU
#SBATCH --ntasks-per-node=1        # 每节点1个任务
#SBATCH --cpus-per-task=4          # 每个任务4个CPU核心
#SBATCH --time=00:10:00            # 最大运行时间10分钟（实际会更短）
#SBATCH --output=%x_%j.out        # 输出日志
#SBATCH --error=%x_%j.err         # 错误日志
#SBATCH --mem=16G                  # 内存请求

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