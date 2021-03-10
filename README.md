# Adaptive wavelet pooling for CNN

This repository implements and evaluates adaptive wavelet pooling.
Download it on Ubuntu by typing:
``` bash
$ git clone git@github.com:Fraunhofer-SCAI/wavelet_pooling.git
```
###### Dependencies
Code in the repository depends on PyTorch.
To install it run:
``` bash
$ pip install torch torchvision
```

###### Experiments
This code allows repetition of the experiments in the paper.
Assuming you want to work with GPU 0, to repeat the experiments from table 1 run
``` bash
$ cd experiments
$ CUDA_VISIBLE_DEVICES=0 python experiments/run_mnist.py
```
To repeat our experiments on street view house numbers or CIFAR-10 run ```train_cifar_SVHN.py``` with the parameters described in the paper.

###### Unit-Testing
This repository ships code to compute the fast wavelet transformation
in PyTorch. To test sparse matrix multiplication and convolution based
implementations run
``` bash
$ pip install PyWavelets
$ pytest
```
To manually verify the wavelet loss values for various standard wavelets type
``` bash
$ ipython tests/wavelet_test.py
```
###### Citation:
If you find this work useful please consider citing the paper:
```
@inproceedings{wolter2021adaptive,
  author={Wolter, Moritz and Garcke, Jochen},
  title={Adaptive wavelet pooling for convolutional neural networks},
  booktitle={The 24th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year = {2021}
}
```

###### Funding:
This work was developed in the Fraunhofer Cluster of Excellence Cognitive Internet Technologies.

