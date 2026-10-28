#!/bin/bash

conda init bash
conda activate placa

python mainDigit.py      \
    --epocas 2     \
    --lr 0.0000001  \
    --batch 16      \
    --dataset "/media/vitor/Dados/NewLettersDataset" \
    --fine_tuning True