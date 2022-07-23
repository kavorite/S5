# S5: Simple State Space Layers for Sequence Modeling

This repository provides the official implementation and experiments for the 
paper: Simple State Space Layers for Sequence Modeling (https://arxiv.org/TBDNEEDTOUPDATE TRY TO INCLUDE Picture like S4 does!!!). 
The core contribution is the S5 layer which is meant to simplify the prior
S4 approach (https://arxiv.org/abs/2111.00396) while retaining its performance and efficiency.

While it has departed a fair amount, this repository originally started off with much of the JAX implementation of S4 from the
Annotated S4 blog post by Sasha Rush (https://github.com/srush/annotated-s4). 

## Experiments
The Long Range Arena and 
Speech Commands experiments in the paper were performed using the dataloaders from the Official S4 repository (https://github.com/HazyResearch/state-spaces). 
We are currently in the process of adding simplified dataloaders that reduce the number of packages required to run the experiments for 
our JAX implementation.

We currently provide the ability to run the IMDB classification experiment easily in a Google Colab notebook (<a href="https://githubtocolab.com/jimmysmith1919/S5_release/S5_IMDB_Classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>). 
The rest of the experiments will be added soon!


## Requirements
All of the requirements are already installed in the Google Colab notebook provided.

To run the code on your own machine, you will need to install JAX (https://github.com/google/jax#installation).
A more formal requirements installation process will be added soon!

## Repository Structure
```
data/            default location of data files
src/             source code for models, datasets, etc.
    dataloading.py   dataloading functions
    layers.py        Defines the S5 layer which wraps the S5 SSM with nonlinearity, norms, dropout, etc.
    seq_model.py     Defines deep sequence models that consist of stacks of S5 layers
    ssm.py           S5 SSM implementation
    ssm_init.py      Helper functions for initializing the S5 SSM 
    train.py         training loop entrypoint
    train_helpers.py functions for optimization, training and evaluation steps
```