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
