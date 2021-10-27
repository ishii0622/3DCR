# 3DCR
3D Cell Reconstruction
## Requirements
Python 3  
CUDA 11.3  
PyTorch 1.7 or newer.  
I recommend the latest stable release, obtainable from https://pytorch.org/.  
This project is tested on Windows 10 with python 3.9.7, PyTorch 1.10.0
## Installation instructions
I recomend using Anaconda to create a new environment.
1. Create a fresh conda environment, and install all dependencies.
```bash 
conda create -n 3DCR python=3.9
conda activate 3DCR
git clone https://github.com/ishii0622/3DCR
cd 3DCR
pip install -r requirements.txt
```
2. install pytorch  
I recommend the latest stable release, obtainable from https://pytorch.org/.  
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
## Usage
```bash
python reconstruct.py
```