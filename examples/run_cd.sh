#!/bin/bash
margin=0.15
num_layers=8
weight_decay=5e-3
lr=0.0015
alpha=25
python run.py \
  --dataset Amazon-CD \
  --margin ${margin} \
  --num-layers ${num_layers} \
  --weight-decay ${weight_decay} \
  --log 0 \
  --lr ${lr} \
  --alpha ${alpha}