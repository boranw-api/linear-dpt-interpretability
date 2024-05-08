# Decision-Pretrained Transformers

## Instructions for Setting Up the Environment


To create a new conda environment, open your terminal and run the following command:

```bash
conda create --name dpt python=3.9.15
```

Install PyTorch by following the [official instructions here](https://pytorch.org/get-started/locally/) appropriately for your system. The recommended versions for the related packages are as follows with CUDA 11.7:

```bash
torch==1.13.0
torchvision==0.14.0
```
For example, you might run:

```bash
conda install pytorch=1.13.0 torchvision=0.14.0 cudatoolkit=11.7 -c pytorch -c nvidia
```

The remaining requirements are fairly standard and are listed in the `requirements.txt`. These can be installed by running

```bash
pip install -r requirements.txt
```

for executing the program

```bash
sh run_bandit.sh
```