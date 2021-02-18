# Adaptive wavelet pooling for CNN

This repository implements and evaluates adaptive wavelet pooling.
Download it on Ubuntu by typing:
``` bash
$ git clone git@github.com:Fraunhofer-SCAI/wavelet_pooling.git
```
This repository ships code to compute the fast wavelet transformation
in PyTorch. To test this part of the the code run
``` bash
$ pytest
```
To manually verify the wavelet loss values for various standard wavelets type
``` bash
$ ipython tests/wavelet_test.py
```
This code allows repetition of the experiments in the paper.
Assuming you want to work with GPU 0, to repeat the experiments from table 1 run
``` bash
$ cd experiments
$ CUDA_VISIBLE_DEVICES=0 python experiments/run_mnist.py
```

TODO: Finish until the AISTATS camera ready deadline.
