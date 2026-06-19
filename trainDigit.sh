#!/bin/bash


conda activate reconhecimento_placa

python mainDigit.py \
    --epocas 16 \
    --lr 0.02 \
    --batch 16