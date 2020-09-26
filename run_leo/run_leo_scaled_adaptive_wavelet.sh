#!/bin/bash
#SBATCH --job-name=sawpool
#SBATCH --output=runs/sawpool.txt
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH -n 1

module load Anaconda3
module load Singularity
singularity exec --nv /opt/software/Singularity/pytorch-19.09-py3.sif python ../train_cifar.py --lr 0.01 --momentum 0.9 --b 96 --pooling_type adaptive_wavelet --tensorboard
