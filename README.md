Deep learning based MRI segmentation toolkit with patch-based method.

Please refer to different branches for different applications/examples.

- master: whole brain segmentation

The main programming language used is Python 3.5.


### Convolutional Neural Networks
This package implements Convolutional Neural Networks (CNN) on top of Keras.
Keras supports Theano and TensorFlow, and I decided to use Theano for data parallelism.

You can switch to TensorFlow backend.

### Setting environment

In order to reproduce the results, it is recommended to run in exactly the same environment.
In this document I use Anaconda.

### CUDA and cuDNN

This package is tested with CUDA 8.0 and cuDnn 5105

Consult the system manager to install them

[CUDA](https://developer.nvidia.com/cuda-downloads)

The tested CUDA environment

    $ cat /proc/driver/nvidia/version
```
NVRM version: NVIDIA UNIX x86_64 Kernel Module  375.26  Thu Dec  8 18:36:43 PST 2016
GCC version:  gcc version 4.8.5 20150623 (Red Hat 4.8.5-11) (GCC)
```

[cuDNN](https://developer.nvidia.com/cudnn)

The tested environment

    $ python -c "import theano; from theano.sandbox.cuda.dnn import version; print(version())"
```
Using gpu device 1: GeForce GTX TITAN X (CNMeM is disabled, cuDNN 5105)
(5105, 5105)
```

### Anaconda environment
A distribution management tool that helps you to install packages without root permission

[Download and install Anaconda Python 3.x version] (https://www.continuum.io/downloads)

Update conda
Create a new environment running Python 3.5

    $ conda update conda
    $ conda update anaconda
    $ conda install cmake
    $ conda install openblas
    $ conda create -n py35env python=3.5

Activate the environment

    $ source activate py35env

Install Python packages into the environment

    (py35env)$ pip install -r requirements.txt


### Bleeding-edge of Theano

This package uses the dev version of Theano (0.9.dev4), which means you may not to able to install it via the requirements.txt.
You need to fetch it by

    $ pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git@ecfc65ec8de80ebfee4c63d1bed48c3cd105a805

You may use updated/newest version at your own risk not reproducing exact results.


Set default Keras backend as Theano and configure Theano

    $ cp keras.json ~/.keras/keras.json
    # edit the blas path
    $ cp .theanorc ~/.theanorc

### Environment and alias

I provide my favorite alias and environment variable setting, some of are optional.

Consider adding the configuration in ```.bashrc``` to your shell.


See directory ```cnn_segmentation``` for more details.
