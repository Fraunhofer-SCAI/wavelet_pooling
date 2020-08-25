#!/bin/bash
python cifar_pool.py --pool adaptive_wavelet --lr 0.1
python cifar_pool.py --pool adaptive_wavelet --lr 0.01 --resume
python cifar_pool.py --pool adaptive_wavelet --lr 0.001 --resume
