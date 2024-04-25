#!/bin/bash
margin=0.15
num_layers=4
weight_decay=5e-4
lr=0.002
alpha=25
python run.py \
  --dataset yelp \
  --margin ${margin} \
  --num-layers ${num_layers} \
  --weight-decay ${weight_decay} \
  --log 0 \
  --lr ${lr} \
  --alpha ${alpha}