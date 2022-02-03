# README #

This package is a PyTorch port of the original **Spiking** Neural Networks with **GoogLe**Net-Like Inception Module(**SpikeGoogle**) framework for improved backpropagation based spiking neural networks (SNNs) learning with Inception modules. The original implementation is in C++ with CUDA and CUDNN. It is available at https://github.com/changjunyuan/slayerPytorch.



## Citation

Xuan Wang , Minghong Zhong , Hoiyuen Cheng, Junjie Xie, Yingchu Zhou, Jun Ren, Mengyuan Liu. "SpikeGoogle: Spiking Neural Networks with GoogLeNet-Like Inception Module." official paper version will be released soon.



## What is this repository for?

- For learning synaptic weight and axonal delay parameters of a multilayer spiking neural network.

- Natively handles multiple spikes in each layer and error backpropagation through the layers.

- Implant inception inside the SNNs, where does [SLYAER](https://bitbucket.org/bamsumit/slayer ) provide.

- Version 1.0

  

- 

## Requirements

Python 3 with the following packages installed:

- matplotlib==3.4.1
- numpy==1.20.2
- pickleshare==0.7.5
- PyYAML==5.4.1
- h5py==3.2.1
- torch==1.7.1
- torchvision==0.8.2

A **CUDA** enabled **GPU** is required for training any model. No plans on CPU only implementation yet. The software has been tested with CUDA libraries version 9.2 and GCC 7.3.0 on Ubuntu 18.04



## Installation

The repository includes C++ and CUDA code that has to be  compiled and installed before it can be used from Python, download the  repository and run the following command to do so:

```
python setup.py install
```

To test the installation:

```
cd test
python -m unittest
```



## Code

Any implementations can be found inside Examples folder.

###  Examples

- Run example CNN implementation

  ```
  cd  01_NMNIST_CNN
  tar -xvf NMNISTsmall.zip
  python nmnistCNN.py
  ```

### Our main work

* Run  SpikeGoogle implementation

  ```
  cd 03_NMNIST_3G
  tar -xvf NMNISTsmall.zip
  python nmnist3G.py
  ```



## Contribution

* By Xuan Wang , Minghong Zhong , Hoiyuen Cheng, Junjie Xie, Yingchu Zhou, Jun Ren, Mengyuan Liu.

* This work builds on initial implementation by [Hoiyuen Cheng](mailto:1094404954@qq.com).

  

## Contact

For queries contact [Hoiyuen Cheng](mailto:1094404954@qq.com)

### License & Copyright

Copyright 2022 SpikeGoogle-PyTorch is free software: you can redistribute it and/or modify it under the terms of GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

SpikeGoogle-PyTorch is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License SpikeGoogle. If not, see http://www.gnu.org/licenses/.

