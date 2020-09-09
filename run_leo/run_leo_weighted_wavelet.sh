#!/bin/bash
#SBATCH --job-name=awpool2
#SBATCH --output=runs/s_wavelet2.txt
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH -n 1

module load Anaconda3
module load Singularity
singularity exec --nv /opt/software/Singularity/pytorch-19.09-py3.sif python ../train_cifar.py --pooling_type scaled_wavelet --lr 0.01 --tensorboard
