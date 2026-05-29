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
import matplotlib.pyplot as plt

def main():
    #Isso vai me poupar um tempo do caralho
    datasetPath = "/media/vitor/data/dogs-vs-cats/" 
    if os.path.exists("/media/vitor/data/dogs-vs-cats/"):
        datasetPath = "/media/vitor/data/dogs-vs-cats/"
    else:
        datasetPath = "/home/vitor/Documents/dogs-cats/dogs-vs-cats/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", default=16, required=False, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch", default=16 ,type=int)
    parser.add_argument("--fine_tuning", default=False, type=bool)
    args = parser.parse_args()

    #Parametros
    epocas = args.epocas
    learning_rate = args.lr
    batch_size = args.batch
    fine_tuning = args.fine_tuning
    minLoss = 5.0
    
    file_list = os.listdir(datasetPath + "train")
    random.shuffle(file_list)
    
    files_train = file_list[:int(len(file_list)*0.8)]
    files_val = file_list[int(len(file_list)*0.8):]

    if not os.path.exists("val_files.txt"):
        with open ('val_files.txt', "w") as f:
            for file in files_val:
                f.write(file + '\n')

    if fine_tuning == False:

        transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

        datasetTrain = DatasetCatAndDog(datasetPath,files_train, transforms)
        datasetVal = DatasetCatAndDog(datasetPath,files_val, transforms)

        dataLoaderTrain = DataLoader(datasetTrain, batch_size, True)
        dataLoaderVal = DataLoader(datasetVal, 1, False)

        Cnn = CnnModel()
        image, label = next(iter(dataLoaderTrain))

        Cnn = Cnn.cuda()
        criterio = CrossEntropyLoss().cuda()
        otimizador = SGD(Cnn.parameters(),learning_rate)

        for epoca in range(epocas):

            totalLoss = 0.0
            totalLossVal = 0.0
            accuracyTotal = 0.0

            Cnn.train(True)

            for iteration, (image, label) in enumerate(dataLoaderTrain):
                otimizador.zero_grad()
                image = image.cuda()
                label = label.cuda()
                inferencia = Cnn(image)
                loss = criterio(inferencia, label)
                loss.backward()
                otimizador.step()
                totalLoss += loss
                
                if iteration % 100 == 0:
                    print(f"epoca {epoca}, iteração: {iteration}, perda {loss.item():.4f}")

            totalLoss /= len(dataLoaderTrain)

            fig, ax = plt.subplots()
            ax.plot(totalLoss.item(), epocas)
            plt.show()
            # print(f"a perda por epoca no treino é: {totalLoss}")

            Cnn.eval()
            with torch.no_grad():
                for iteration, (image, label) in enumerate(dataLoaderVal):

                    image = image.cuda()
                    label = label.cuda()
                    # b = image.size(0)
                    inferencia = Cnn(image)
                    loss = criterio(inferencia, label)
                    totalLossVal += loss
                    predictions = torch.argmax(inferencia, dim=1)
                    tensorComparacao = predictions == label
                    accuracy = torch.sum(tensorComparacao)
                    accuracyTotal += accuracy
                    # if iteration % 100 == 0:
                    #     print(loss.item())
                    fig, ax = plt.subplot()
                    ax.plot(epocas,loss.item())
                    plt.show()

            accuracyTotal = accuracyTotal/ len(dataLoaderVal)
            totalLossVal = totalLossVal / len(dataLoaderVal)

            if minLoss > totalLossVal:
                minLoss = totalLossVal
                torch.save(Cnn.state_dict(), 'best_model/Model.pth')
                print(f"Melhor modelo encontrado!!!")
            print(f"acurácia validação: {accuracyTotal.item():.4f}, Perda Total na validação {totalLossVal.item():.4f}")
        #print(f"Fim da epoca {epoca}")
    
    else:
         LoadModel = CnnModel()
         LoadModel.load_state_dict(torch.load('best_model/Model.pth'))
         LoadModel.cuda()
         criterio = CrossEntropyLoss().cuda()
         otimizador = SGD(LoadModel.parameters(),learning_rate)

         transforms = v2.Compose([
            v2.ToImage(),
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
         
         datasetTrain = DatasetCatAndDog(datasetPath,files_train, transforms)
         datasetVal = DatasetCatAndDog(datasetPath,files_val, transforms)

         dataLoaderTrain = DataLoader(datasetTrain, batch_size, True)
         dataLoaderVal = DataLoader(datasetVal, 1, False)

         for epoca in range(epocas):

            totalLoss = 0.0
            totalLossVal = 0.0
            accuracyTotal = 0.0

            LoadModel.train(True)

            for iteration, (image, label) in enumerate(dataLoaderTrain):
                otimizador.zero_grad()
                image = image.cuda()
                label = label.cuda()
                inferencia = LoadModel(image)
                loss = criterio(inferencia, label)
                loss.backward()
                otimizador.step()
                totalLoss += loss
                
                if iteration % 100 == 0:
                    print(f"epoca {epoca}, iteração: {iteration}, perda {loss.item():.4f}")

            totalLoss /= len(dataLoaderTrain)



            LoadModel.eval()
            with torch.no_grad():
                for iteration, (image, label) in enumerate(dataLoaderVal):

                    image = image.cuda()
                    label = label.cuda()
                    inferencia = LoadModel(image)
                    loss = criterio(inferencia, label)
                    totalLossVal += loss
                    predictions = torch.argmax(inferencia, dim=1)
                    tensorComparacao = predictions == label
                    accuracy = torch.sum(tensorComparacao)
                    accuracyTotal += accuracy
                    # if iteration % 100 == 0:
                    #     print(loss.item())
            accuracyTotal = accuracyTotal/ len(dataLoaderVal)
            totalLossVal = totalLossVal / len(dataLoaderVal)

            if minLoss > totalLossVal:
                minLoss = totalLossVal
                torch.save(LoadModel.state_dict(), 'best_model/Model.pth')
                print(f"Melhor modelo encontrado!!!")
            print(f"acurácia validação: {accuracyTotal.item():.4f}, Perda Total na validação {totalLossVal.item():.4f}")
            #print(f"Fim da epoca {epoca}")
            
            
if __name__ == "__main__":
    main()