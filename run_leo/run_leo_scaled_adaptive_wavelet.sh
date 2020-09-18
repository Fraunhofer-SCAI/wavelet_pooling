#!/bin/bash
#SBATCH --job-name=sawpool
#SBATCH --output=runs/sawpool.txt
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH -n 1

module load Anaconda3
module load Singularity
singularity exec --nv /opt/software/Singularity/pytorch-19.09-py3.sif python ../train_cifar.py --pooling_type scaled_adaptive_wavelet --lr 0.1 --tensorboard --momentum 0.0
