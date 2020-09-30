#!/bin/bash
#SBATCH --job-name=multi_cifar
#SBATCH --output=runs/multi_cifar.txt
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1 
#SBATCH -n 1

module load Anaconda3
module load Singularity
singularity exec --nv /opt/software/Singularity/pytorch-19.09-py3.sif python cifar_loop.py
