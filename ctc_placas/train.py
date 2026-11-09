import os
import glob
import pandas as pd
import string
import collections

from tqdm import tqdm


from PIL import Image

from sklearn.model_selection import train_test_split

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

ds_path = 'caminho/para/o/dataset'

data = glob.glob(os.path.join(ds_path, '*.jpg'))
path = ds_path
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
for d in data:
    x = d.split('/')[-1]
    plate_str = x.split('.')[0]
    if len(plate_str) != 7:
        print(f"Ignorando arquivo com tamanho diferente de 7 ({len(plate_str)}): {x}")
        continue
    datas['image'].append(x)
    datas['label'].append([mapping[i] for i in plate_str])
df = pd.DataFrame(datas)
print(df.head())

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

train_data = PlacaDataset(path, df_train, transform_train)
test_data = PlacaDataset(path, df_test, transform_val)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=8, num_workers=4, pin_memory=True, drop_last=True)



model = CRNN(in_channels=1, output=num_class).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CTCLoss()

engine = Engine(model, optimizer, criterion, device=DEVICE, epochs=50)
engine.fit(train_loader)
outs, loss = engine.evaluate(test_loader)


print('Loss de validação: ', loss)
# print(outs)

import matplotlib.pyplot as plt
import numpy as np

ids = np.random.randint(len(data))

image = data[ids]
out = engine.predict(image)[0]

print(len(out))

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