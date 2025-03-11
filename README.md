# AptGAN
de novo generation of RNA aptamers for protein targets using generative adversarial network-based deep learning approach
![title](title.png)

## Requirements
* Python 3.8
* torch
* tensorflow
* numpy
* pandas
* xgboost
* propy
* RNA
* repDNA
* sklearn
* pickle
* random
* optuna
* matplotlib

You can install the dependencies with the versioins specified in requirements.txt. 

## Usage
You can use AptGAN to generate aptamers:
```
$ python main.py --type 0 --seq_num 10 --seq_min 20 --seq_max 120 --path './dataset/'
```

AptGAN can generate aptamers for a specific protein:
```
$ python main.py --type 1 --seq_num 10 --seq_min 20 --seq_max 120 --threshold 0.6 --path './dataset/' --pro_file './dataset/sota/CREB3/CREB3.fasta' --pro_ss './dataset/sota/CREB3/CREB3_SS.fas'
```

But you have to generate the secondary structural file of the protein sequence first (in FASTA format), e.g.:
```
$ python ./s4pred/run_model.py --outfmt fas ./dataset/sota/CREB3/CREB3.fasta >./dataset/sota/CREB3/CREB3.fas
```



## SOTA
The source code and generated sequences used in SOTA comparisons are provided [Zenodo](https://zenodo.org/records/14862169).

