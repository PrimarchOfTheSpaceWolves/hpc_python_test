# HPC Python Test Files

## Setup
Environment instructions can be found in the script ```createEnv.sh```.

If run directly, this script will create a conda environment ```HPC``` and install the necessary packages.

However, it should be possible to install the packages directly, since ```pip3``` is used throughout.

If you do use conda, you will need to activate the environment before running any of the Python scripts:

```
conda activate HPC
```

## Simple CNN
Very simple CNN to predict classes on CIFAR10.  This will download the CIFAR data to a folder ```data```.  On completion, it will also save the final model to ```models/model.pth```.  The script should otherwise run in a self-contained fashion.

```
python simpleCNN.py
```

On a GeForce GTX 4060, this took less than 2 minutes to train.
Final training and testing accuracy should be somewhere in the 70s, maybe 80s.




