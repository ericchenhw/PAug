#!/bin/bash
gpu=$1
perturbation_strength=$2
perturbation_threshold=$3
semantic_strength=$4
pretraining_dataset=$5
adversarial_dataset=$6
# idle_interval=$7
# max_patience_discriminator=$8
# max_patience_aug=$9
# max_patience_contrastiv=$10
ARGS=${@:7}


python pretrain_paug.py \
    --exp Final_Test \
    --batch-size 128\
    --model-path saved \
    --tb-path tensorboard \
    --gpu $gpu \
    --tb-freq 10 \
    --perturbation_strength ${perturbation_strength} \
    --perturbation_threshold ${perturbation_threshold} \
    --semantic_strength ${semantic_strength} \
    --pretraining_dataset ${pretraining_dataset} \
    --adversarial_dataset ${adversarial_dataset} \
    --epochs 34 \
    --idle_interval 30 \
    --max_patience_discriminator 5 \
    --max_patience_aug 8 \
    --max_patience_contrastive 30 \
    $ARGS  >logs/Final_Test_${iterative_contrastive_interval}_${iterative_loss_interval}_reg_strength_${reg_strength}_reg_threshold_${reg_threshold}_pretraining_dataset_${pretraining_dataset}_adversarial_dataset_${adversarial_dataset}.out 2>&1