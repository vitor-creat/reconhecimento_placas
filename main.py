from dataset import DatasetCatAndDog
import os
import torch
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model import CnnModel
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import numpy as np
import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", default=16, required=False, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch", default=16 ,type=int)
    parser.add_argument("--fine_tuning", default=False, type=bool)
    args = parser.parse_args()

    listImage = []
    """/home/vitor/Documents/data/dogs-vs-cats/train"""
    """/media/vitor/data/dogs-vs-cats/train"""
    for i in os.listdir("/media/vitor/data/dogs-vs-cats/train"):
        listImage.append(i)
    transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    X_train, X_val = train_test_split(
    listImage, test_size=0.20, shuffle=False
)
    """/home/vitor/Documents/data/dogs-vs-cats/"""
    """/media/vitor/data/dogs-vs-cats/"""
    datasetTrain = DatasetCatAndDog('/media/vitor/data/dogs-vs-cats/',X_train, transforms)
    datasetVal = DatasetCatAndDog('/media/vitor/data/dogs-vs-cats/',X_val, transforms)

    dataLoaderTrain = DataLoader(datasetTrain, 16, True)
    dataLoaderVal = DataLoader(datasetVal, 1, False)

    Cnn = CnnModel()
    image, label = next(iter(dataLoaderTrain))
    resultado = Cnn(image)

    print(args)
    print(Cnn)
    # print(resultado.shape, maior.shape)
    # print(resultado[0], maior[0])
    # print(maior)
    # print(label)
    epocas = args.epocas
    learning_rate = 0.002
    minLoss = 5.0
    Cnn = Cnn.cuda()
    criterio = CrossEntropyLoss().cuda()
    otimizador = SGD(Cnn.parameters(),learning_rate)
    print(len(dataLoaderTrain))
    for epoca in range(epocas):
        perdaTotal = 0.0
        perdaTotalVal = 0.0
        accuracyTotal = 0.0
        Cnn.train(True)
        for iteracao, (image, label) in enumerate(dataLoaderTrain):
            otimizador.zero_grad()
            image = image.cuda()
            label = label.cuda()
            inferencia = Cnn(image)
            perda = criterio(inferencia, label)
            perda.backward()
            otimizador.step()
            perdaTotal += perda
            
            if iteracao % 100 == 0:
                print(f"epoca {epoca}, iteração: {iteracao}, perda {perda.item()}")

        perdaTotal /= len(dataLoaderTrain)
        
        # print(f"a perda por epoca no treino é: {perdaTotal}")

        Cnn.eval()
        with torch.no_grad():
            for iteracao, (image, label) in enumerate(dataLoaderVal):

                image = image.cuda()
                label = label.cuda()
                # b = image.size(0)
                inferencia = Cnn(image)
                perda = criterio(inferencia, label)
                perdaTotalVal += perda
                maior = torch.argmax(inferencia, dim=1)
                tensorComparacao = maior == label
                accuracy = torch.sum(tensorComparacao)
                accuracyTotal += accuracy
                if iteracao % 100 == 0:
                    print(perda.item())
        accuracyTotal = accuracyTotal/ len(dataLoaderVal)
        perdaTotalVal = perdaTotalVal / len(dataLoaderVal)
        if minLoss > perdaTotalVal:
            minLoss = perdaTotalVal
            torch.save(Cnn.state_dict(), 'Model.pth')
            print(f"Melhor modelo encontrado!!!")
        print(f"acurácia validação: {accuracyTotal.item()}, Perda Total na validação {perdaTotalVal.item()}")
    print(f"Fim da epoca {epoca}")






        
            
            
if __name__ == "__main__":
    main()