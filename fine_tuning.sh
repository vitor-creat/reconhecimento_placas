#!/bin/bash

conda init bash
conda activate reconhecimento_placa

python main.py      \
    --epocas 8     \
    --lr 0.0000001  \
    --batch 16      \
    --fine_tuning True