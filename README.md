# Release Graph Pre-Trained Model via Privacy-Preserving Data Augmentation

This project is the implementatoin of the paper "Release Graph Pre-Trained Model via Privacy-Preserving Data Augmentation". This paper proposes to use **P**rivacy-preserving data **Aug**mentation (**PAug**) when pre-training the GNN, to protect the sensitive link in the pre-training dataset from leakage.

This repo contains the codes and data reported in the paper.

## Dependencies

- Python ≥ 3.7
- torch==1.11.0
- dgl==0.4.3
- torch_geometric==2.0.4
- pip install -r requirements.txt

## Pretraining and Donstream Datasets

The raw graph datasets (H-index, Us-Airport, Actor, Chameleon and DD242) have been saved in "data/raw_data".

The pretraining graph $G_{train}$ and the downstream graphs $G_{downstream}$ are also saved in "data/". $G_{train}$ and $G_{downstream}$ can also be derived by running generate_pretraining_and_downstream_dataset.py.

For example: Generate the pretraining and downstream dataset of usa-airports.

```bash
python generate_pretraining_and_downstream_dataset.py --dataset usa-airports
```

## Pretrain the GNN with PAug

```bash
bash scripts/pretrain_paug.sh <gpu> <beta> <m> <gamma> <pretraining_dataset> <pretraining_dataset> 
```

For example: Pretrain the GNN on usa-airports, with $\beta$=5, $m$=0.3 and $\gamma$=0.5.

```bash
bash scripts/pretrain_paug.sh <gpu> 0.1 0.3 0.5 usa-airports_mst_twin_domain1 usa-airports_mst_twin_domain1
```


## Test the Performance of the Pre-trained GNN with PAug

### Generate the Node Embedding

```bash
bash scripts/generate.sh <gpu> <load_path> <dataset>
```

For example:

```bash
bash scripts/generate.sh 1 saved/<file_name>/current.pth usa-airports_mst_twin_domain2
```

### Test the Privacy-preserving Performance of the Pre-trained GNN with PAug

```bash
bash scripts/link_prediction/predict_links_of_pretraining_dataset.sh <gpu> <load_path> <hidden_size> <dowstream_dataset> <pretraining_dataset>
```

For example:

```bash
bash scripts/link_prediction/predict_links_of_pretraining_dataset.sh 0 saved/<file_name> 64 usa-airports_mst_twin_domain2 usa-airports_mst_twin_domain1
```

### Test the Generalizability of the Pre-trained GNN with PAug

```bash
bash scripts/node_classification/classify_node_of_downstream_dataset.sh <gpu> <load_path> <hidden_size> <downstream_dataset>
```

For example:

```bash
bash scripts/node_classification/classify_node_of_downstream_dataset.sh 1 saved/<file_name> 64 usa-airports_mst_twin_domain2
```

## Acknowledgements

Part of this code is inspired by Jiezhong Qiu et al.'s [GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training](https://github.com/THUDM/GCC).
