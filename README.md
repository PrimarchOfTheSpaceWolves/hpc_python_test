# HPC Python Test Files

## Setup
Certain packages should be installed in the system:
```
sudo apt install gcc g++
```

Needless to say, some distribution of Python 3 should be installed as well (ideally either Anaconda or Miniconda).

Environment instructions can be found in the script ```createEnv.sh```.

If run directly, this script will create a conda environment ```HPC``` and install the necessary packages.

However, it should be possible to install the packages directly, since ```pip3``` is used throughout.

If you do use conda, you will need to activate the environment before running any of the Python scripts:

```
conda activate HPC
```

## Simple CNN
This is a very simple CNN to predict classes on CIFAR10.  This will download the CIFAR data to a folder ```data```.  On completion, it will also save the final model to ```models/model.pth```.  The script should otherwise run in a self-contained fashion.

```
python simpleCNN.py
```

On a GeForce GTX 4060, this took less than 2 minutes to train.
Final training and testing accuracy should be somewhere in the 70s, maybe 80s.

## Image Generation with SDXL
This loads up Stable Diffusion XL and its refiner and generates some images.
Effectively derived from [here](https://huggingface.co/docs/diffusers/using-diffusers/sdxl).

The script will download a bunch of model files to ```~/.cache/huggingface/hub``` directory.

When complete, it will save ten images to the ```out_images``` directory in the repo.





