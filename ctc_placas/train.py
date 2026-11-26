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
import argparse

from utils import save_model, load_model


parser = argparse.ArgumentParser(description='Treinar modelo CTC para reconhecimento de placas')
parser.add_argument('--epochs', type=int, default=10, help='Número de épocas')
parser.add_argument('--batch-size', type=int, default=16, help='Tamanho do batch')
parser.add_argument('--learning-rate', type=float, default=1e-4, help='Taxa de aprendizado')
parser.add_argument('--data-path', type=str, default='/media/vitor/Dados/tbFcZE-RodoSol-ALPR/images/cars-br', help='Caminho para os dados')
parser.add_argument('--yolo-weights', type=str, default='best.pt', help='Caminho para os pesos do modelo YOLO')
parser.add_argument('--experiment-name', type=str, default='results', help='Nome do experimento')


args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
data_path = args.data_path
yolo_weights = args.yolo_weights


ds_path = data_path

os.makedirs(f'{args.experiment_name}/models', exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
session_name = f"{timestamp}"
log_file = f"{args.experiment_name}/{session_name}.log"

detect_model = YOLO(Path(yolo_weights))
data = sorted(glob.glob(os.path.join(ds_path, '*.jpg')))
dataTxt = []
for d in data:
    dataTxt.append(d.replace('jpg', 'txt'))

path = ds_path


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
    T2.Resize((64, 256)),
    T2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T2.GaussianBlur(kernel_size=3, sigma=1),
    T2.RandomPerspective(distortion_scale=0.2, p=0.5),
    T2.ToTensor(),
])

transform_val = T2.Compose([
    T2.Resize((64, 256)),
    T2.ToTensor(),
])

train_data = PlacaDataset(path, df_train, detect_model,transform_train)
test_data = PlacaDataset(path, df_test, detect_model,transform_val)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=1, num_workers=0, pin_memory=True, drop_last=True)



model = CRNN(in_channels=1, output=num_class).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CTCLoss()

engine = Engine(model, optimizer, criterion,device=DEVICE, epochs=epochs, mapping_inv=mapping_inv, experiment_name=args.experiment_name)
engine.fit(train_loader, test_loader)
outs, loss = engine.evaluate(test_loader)


print('Loss de validação: ', loss)
# print(outs)



import matplotlib.pyplot as plt
import numpy as np

ids = np.random.randint(len(data))

image = data[ids]
out = engine.predict(image)[0]

# print(len(out))

def show_prediction(out, image):
    gt = image.split('/')[-1].split('.')[0]
    imagePIL = Image.open(image)

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