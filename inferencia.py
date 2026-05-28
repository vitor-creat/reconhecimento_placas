from dataset import DatasetCatAndDog
import os
import torch
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model import CnnModel
from torch.nn import CrossEntropyLoss
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt


CLS2LABEL = ['Gato', 'Cachorro']

datasetPath = "/media/vitor/data/dogs-vs-cats/" 

if os.path.exists("/media/vitor/data/dogs-vs-cats/"):
    datasetPath = "/media/vitor/data/dogs-vs-cats/"
else:
    datasetPath = "/home/vitor/Documents/dogs-cats/dogs-vs-cats"


def load1Image(modelo,args):
        image = Image.open(args.imagePath)
        copyImage = image.copy()
        transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((224,224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transforms(image).unsqueeze(0)
        inferencia = modelo(image)
        maior = torch.argmax(inferencia, dim=1)
        classe_predita = CLS2LABEL[maior]
        plt.imshow(copyImage)

        plt.title(f"Classe predita {classe_predita}")
        plt.show()


def loadBatch(modelo):
        listImage = []
        device = torch.device("cuda")
        modelo.to(device)

        files_val = []
        with open('val_files.txt', 'r') as f:
            for line in f:
                files_val.append(line.strip())

        transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((224,224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        X_val = listImage
        """/home/vitor/Documents/data/dogs-vs-cats/"""
        """/media/vitor/data/dogs-vs-cats/"""
        datasetVal = DatasetCatAndDog(datasetPath,files_val, transforms)

        dataLoaderVal = DataLoader(datasetVal, 1, False)
        criterio = CrossEntropyLoss().cuda()
        
        accuracyFinal = 0.0
        totalLoss = 0.0
        with torch.no_grad():
            for iteracao, (image, label) in enumerate(dataLoaderVal):
                #print(type(image))
                image = image.cuda()
                label = label.cuda()
                inferencia = modelo(image)
                loss = criterio(inferencia, label)
                totalLoss += loss
                maior = torch.argmax(inferencia, dim=1)
                tensorCompare = maior == label
                accuracy = torch.sum(tensorCompare)
                accuracyFinal += accuracy
        accuracyFinal = accuracyFinal / len(dataLoaderVal)
                
        print(f"acurácia final: {accuracyFinal.item()}")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", required=False, type=str, help="camiho da imagem que será carregada", default="vazio")
    args = parser.parse_args()
    
    LoadModel = CnnModel()
    LoadModel.load_state_dict(torch.load('Model.pth'))
    LoadModel.eval() 
    if args.imagePath != "vazio":
        load1Image(LoadModel,args)
    else:
         loadBatch(LoadModel)

    

    
    



if __name__  =="__main__":
    main()