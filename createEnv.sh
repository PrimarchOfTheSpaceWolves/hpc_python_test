#!/bin/bash

## HCP ###########
conda create -y -n HPC python=3.11
conda activate HPC

# Installing PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Verifying: should see random array and True
python -c "import torch;x=torch.rand(5, 3);print(x);print(torch.cuda.is_available())"

# Installing other packages
pip3 install pandas scikit-learn scikit-image matplotlib pylint gradio jupyter opencv-python
pip3 install diffusers["torch"] transformers 
pip3 install peft

# UP TO HERE

# Using 3.10 in the original example
pip3 install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


pip3 install --upgrade huggingface_hub
pip3 install huggingface_hub[cli,torch]
pip3 install peft
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip3 install --upgrade datasets
pip3 install tensorboard
pip3 install bitsandbytes

pip3 install gym pygame

pip3 install trl[diffusers]
pip3 install wandb

pip3 install lightning
pip3 install seaborn

# pip3 install deepspeed

cd external/diffusers
pip3 install .
cd -

cd external/trl
pip3 install .
cd -



# Should see: [{'label': 'POSITIVE', 'score': 0.9998704791069031}]
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
