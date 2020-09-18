#!/bin/bash
#SBATCH --job-name=apool
#SBATCH --output=runs/apool.txt
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1 
#SBATCH -n 1

module load Anaconda3
module load Singularity
singularity exec --nv /opt/software/Singularity/pytorch-19.09-py3.sif python ../train_cifar.py --pooling_type avg --lr 0.1 --tensorboard


