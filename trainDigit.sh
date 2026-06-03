#!/bin/bash


conda activate reconhecimento_placa

python mainDigit.py \
    --epocas 16 \
    --lr 0.002 \
    --batch 16