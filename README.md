# Release Graph Pre-Trained Model via Privacy-Preserving Data Augmentation

This project is the implementatoin of the paper "Release Graph Pre-Trained Model via Privacy-Preserving Data Augmentation". This paper proposes to use **P**rivacy-preserving data **Aug**mentation (**PAug**) when pre-training the GNN, to protect the sensitive link in the pre-training dataset from leakage.

This repo contains the codes and data reported in the paper.

## Dependencies

This script has been tested runing under Python 3.7.12, with the following packages installed (along with their dependencies).

- `torch==1.11.0`
- `torch-geometric==2.0.4`
- `dgl==0.4.3`
- `numpy==1.21.6`
- `scipy==1.4.1`

Some Python module dependencies are listed in `requirements.txt`, which can be easily intalled with pip:

```bash
pip install -r requirements.txt
```

## Pretraining and Donstream Datasets

The raw graph datasets (H-index, Us-Airport, Actor, Chameleon and DD242) have been saved in "data/raw_data".

The pretraining graph $G_{train}$ and $G_{downstream}$ are also saved in "data/". $G_{train}$ and $G_{downstream}$ can also be derived by running generate_pretraining_and_downstream_dataset.py.

Generate the pretraining and downstream dataset of usa-airports. 

```bash
python generate_pretraining_and_downstream_dataset.py --dataset usa-airports
```

## Pretrain the GNN with PAug

Pretrain the GNN on usa-airports, with $\beta$=5, $m$=0.3 and $\gamma$=0.5.

```bash
bash scripts/pretrain_paug.sh <gpu> 0.1 0.3 0.5 usa-airports_mst_twin_domain1 usa-airports_mst_domain1
```


## Test the Performance of the Pre-trained GNN with PAug

### Generate the Node Embedding

```bash
bash scripts/generate.sh <gpu> <load_path> <dataset>
```

For example:

```bash
bash scripts/generate.sh 1 saved/.../current.pth usa-airports_mst_twin_domain2
```

### Test the Privacy-preserving Performance of the Pre-trained GNN with PAug

```bash
bash scripts/link_prediction/predict_links_of_pretraining_dataset.sh <gpu> <load_path> <hidden_size> <dowstream_dataset> <pretraining_dataset>
```

For example:

```bash
bash scripts/privacy_performance/link_prediction.sh 0 saved/...... usa-airports_mst_twin_domain2 usa-airports_mst_twin_domain1
```

### Test the Generalizability of the Pre-trained GNN with PAug

```bash
bash scripts/node_classification/classify_node_of_downstream_dataset.sh <gpu> <load_path> <hidden_size> <downstream_dataset>
```

For example:

```bash
bash scripts/node_classification/classify_node_of_downstream_dataset.sh 1 save/...... 64 usa-airports_mst_twin_domain2
```
