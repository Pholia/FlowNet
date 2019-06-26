## FlowNet2 (TensorFlow)

Most part of this repo and this README file is from https://github.com/sampepose/flownet2-tf.

This repo contains FlowNet2[1] for TensorFlow. It includes FlowNetC, S, CS, CSS, CSS-ft-sd, SD, and 2.

### Installation
```
pip install enum
pip install pypng
pip install matplotlib
pip install image
pip install scipy
pip install numpy
pip install tensorflow_gpu
pip install opencv-python
```

Linux:
`sudo apt-get install python-tk`

You must have CUDA 9.0 and Cudnn 7.0 installed:
`export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}$`
`make all`
