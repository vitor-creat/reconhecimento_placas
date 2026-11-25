import os
import glob
import pandas as pd
import string
import collections

from tqdm import tqdm

import datetime

from PIL import Image

from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import PlacaDataset
from engine import Engine
from model import CRNN

ds_path = '/media/vitor/Dados/tbFcZE-RodoSol-ALPR/images/cars-br'

detect_model = YOLO(Path("best.pt"))
data = sorted(glob.glob(os.path.join(ds_path, '*.jpg')))
dataTxt = []
for d in data:
    dataTxt.append(d.replace('jpg', 'txt'))

path = ds_path
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print(data[20:30])
# print(dataTxt[20:30])
# input()

all_letters = string.ascii_uppercase + string.digits

mapping = {}
mapping_inv = {}
i = 1
for x in all_letters:
    mapping[x] = i
    mapping_inv[i] = x
    i += 1

num_class = len(mapping) + 1


images = []
labels = []
datas = collections.defaultdict(list)
for d, arquivo_txt in zip(data, dataTxt):
    x = d.split('/')[-1]


    placa = []
    with open(arquivo_txt, "r") as f:
        placa = f.readlines()
    plate_str = placa[1].split(":")[1].strip()
    if len(plate_str) != 7:
        print(f"Ignorando arquivo com tamanho diferente de 7 ({len(plate_str)}): {x}")
        continue
    datas['image'].append(x)
    datas['label'].append([mapping[i] for i in plate_str])
df = pd.DataFrame(datas)
# print(df.head())

df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)
print(f"Train: {len(df_train)}")
print(f"Test: {len(df_test)}")

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

transform_train = T2.Compose([
    T2.Resize((256, 512)),
    T2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T2.GaussianBlur(kernel_size=3, sigma=1),
    T2.RandomPerspective(distortion_scale=0.2, p=0.5),
    T2.ToTensor(),
])

transform_val = T2.Compose([
    T2.Resize((256, 512)),
    T2.ToTensor(),
])

train_data = PlacaDataset(path, df_train, detect_model,transform_train)
test_data = PlacaDataset(path, df_test, detect_model,transform_val)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=8, num_workers=0, pin_memory=True, drop_last=True)



model = CRNN(in_channels=1, output=num_class).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CTCLoss()

engine = Engine(model, optimizer, criterion,device=DEVICE, epochs=50, batch=16, mapping_inv=mapping_inv)
engine.fit(train_loader, test_loader)
outs, loss = engine.evaluate(test_loader)


print('Loss de validação: ', loss)
# print(outs)

os.makedirs("logs", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
session_name = f"{timestamp}"
log_file = f"logs/{session_name}.log"

import matplotlib.pyplot as plt
import numpy as np

ids = np.random.randint(len(data))

image = data[ids]
out = engine.predict(image)[0]

# print(len(out))

def show_prediction(out, image):
    gt = image.split('/')[-1].split('.')[0]
    imagePIL = Image.open(image).convert('L')

    pred = ''
    then = 0
    for x in out:
        if then != x:
            if x > 0 :
                pred += mapping_inv[x]
        then = x

    plt.figure(figsize=(15, 12))
    img_array = np.asarray(imagePIL)
    # img_array = transform_val(imagePIL).unsqueeze(0)
    plt.title(f'Ground Truth - {gt} || Prediction - {pred}')
    plt.axis('off')
    plt.imshow(img_array)
    plt.show()

show_prediction(out, image)