#!/bin/bash
#SBATCH --job-name=max_pool_cifar
#SBATCH --output=runs/max_pool_cifar.txt
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH -n 1

module load Anaconda3
module load Singularity
singularity exec --nv /opt/software/Singularity/pytorch-19.09-py3.sif python train_cifar.py --pooling_type max --lr 0.01 --tensorboard


