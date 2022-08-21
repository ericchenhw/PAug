#!/bin/bash
gpu=$1
load_path=$2
hidden_size=$3
dataset=$4

python gcc/tasks/node_classification.py --dataset $dataset --hidden-size $hidden_size --model from_numpy --emb-path "$load_path/$dataset.npy"

