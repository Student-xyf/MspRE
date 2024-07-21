**MspRE**

This repository contains the code to reproduce for the paper: Joint Entity Relation Extraction Based on Multi-Scale Feature Fusion and Non-parametric Attention Mechanism.

**Installing**

To reproduce all the code, you first need to install the required modules.

You should probably do this in a Python 3 virtual environment.

`conda create -n MspRE python==3.7`

`conda activate MspRE`

`conda install pip`

`pip install -r requirements.txt`

**data**

We used BB, ChemPort, SciERC, and CMeIE datasets. These public datasets can be found on the corresponding websites.

For example：https://sites.google.com/view/bb-2019/dataset

**Pre-trained language model**

The pre-trained language model mainly uses huggingfaced's scibert_scivocab_cased and bert_base_chinese models
