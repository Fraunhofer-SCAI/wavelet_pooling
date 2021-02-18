# Adaptive wavelet pooling for convolutional neural networks

This repository implements and evaluates adaptive wavelet pooling.
Download it on Ubuntu by typing:
``` bash
$ git clone git@github.com:Fraunhofer-SCAI/wavelet_pooling.git
```
This repository ships code to compute the fast wavelet transformation
in PyTorch. To test the code run
``` bash
$ pytest
```
To manually verify the wavelet loss values for various standard wavelets type
``` bash
$ ipython tests/wavelet_test.py
```
This code allows reproduction of the resits in the corresponding paper.
To reproduce the results from table 1 run
``` bash
$ ipython experiments/
```