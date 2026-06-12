import os
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from model import DigitModel
from dataset import DatasetDigits
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():

    # datasetPath = "/media/vitor/data/CNN_letters_custom"
    datasetPath = "/media/vitor/data/CNN_letters_digit_merge"

    # if os.path.exists("/media/vitor/data/CNN_letters_custom"):
    #     datasetPath = "/media/vitor/data/CNN_letters_custom"
    # else:
    #     datasetPath = "/media/vitor/data/CNN_letters_custom"

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
    
    PATH_GRAFICOS = "grafico"

    if not os.path.exists(PATH_GRAFICOS):
        os.makedirs(os.path.join(PATH_GRAFICOS, "treino"), exist_ok=True)
        os.makedirs(os.path.join(PATH_GRAFICOS, "fine_tunning"), exist_ok=True)


    digitsClass = []
    file_list = []
    for pastas in os.listdir(datasetPath):
        for imgs in os.listdir(os.path.join(datasetPath, pastas)):
            digitsClass.append(pastas)
            file_list.append(os.path.join(datasetPath, pastas, imgs))

            

    file_list_tupla = list(zip(digitsClass, file_list))

    # print(len(file_list), len(digitsClass))

    # for cls, img in file_list_tupla[:5]:
    #     print(cls, img)
    # print('\n\n\n')
    # classes = list(set(digitsClass))
    # print(classes, len(classes))
    # input()

    random.shuffle(file_list_tupla)

    # for cls, img in file_list_tupla[:5]:
    #     print(cls, img)
    
    files_train = file_list_tupla[:int(len(file_list_tupla)*0.8)]
    files_val = file_list_tupla[int(len(file_list_tupla)*0.8):]

    if not os.path.exists("val_files.txt"):
        with open ('val_files_digits.txt', "w") as f:
            for file in files_val:
                f.write(file + '\n')

    if fine_tuning == False:

        transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((128,128)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        datasetTrain = DatasetDigits(files_train, transforms)
        datasetVal = DatasetDigits(files_val, transforms)

        dataLoaderTrain = DataLoader(datasetTrain, batch_size, True)
        dataLoaderVal = DataLoader(datasetVal, 1, False)

        Cnn = DigitModel()
        image, label = next(iter(dataLoaderTrain))

        Cnn = Cnn.cuda()
        criterio = CrossEntropyLoss().cuda()
        otimizador = SGD(Cnn.parameters(),learning_rate)

        listLossTrain = []
        listLossVal = []
        listAccTrain = []
        listAccVal = []

        for epoca in range(epocas):
            
            totalLossTrain = 0.0
            accuracyTotalTrain = 0.0
            totalLossVal = 0.0
            accuracyTotalVal = 0.0

            Cnn.train(True)
            iteration = 0
            pbar = tqdm(dataLoaderTrain, desc=f"epoca {epoca}")
            for (image, label) in pbar:
                otimizador.zero_grad()
                image = image.cuda()
                label = label.cuda()
                inferencia = Cnn(image)
                loss = criterio(inferencia, label)
                loss.backward()
                otimizador.step()
                totalLossTrain += loss
                predictions = torch.argmax(inferencia, dim=1)

                tensorComparacao = predictions == label
                accuracy = torch.sum(tensorComparacao) / batch_size
                accuracyTotalTrain += accuracy
                iteration += 1
                pbar.set_postfix(loss = f"{loss.item():.4}", acc = accuracy.item())
            

            totalLossTrain /= len(dataLoaderTrain)

            accuracyTotalTrain /= len(dataLoaderTrain)

            listLossTrain.append(totalLossTrain.item())
            listAccTrain.append(accuracyTotalTrain.item())


            Cnn.eval()
            with torch.no_grad():
                for iteration, (image, label) in enumerate(dataLoaderVal):

                    image = image.cuda()
                    label = label.cuda()

                    inferencia = Cnn(image)
                    loss = criterio(inferencia, label)
                    totalLossVal += loss
                    predictions = torch.argmax(inferencia, dim=1)
                    tensorComparacao = predictions == label
                    accuracy = torch.sum(tensorComparacao)
                    accuracyTotalVal += accuracy

            accuracyTotalVal = accuracyTotalVal/ len(dataLoaderVal)
            totalLossVal = totalLossVal / len(dataLoaderVal)

            listAccVal.append(accuracyTotalVal.item())
            listLossVal.append(totalLossVal.item())

            if minLoss > totalLossVal:
                minLoss = totalLossVal
                torch.save(Cnn.state_dict(), 'best_model/ModelDigit.pth')
                print(f"Melhor modelo encontrado!!!")
            print(f"acurácia validação: {accuracyTotalTrain.item():.4f}, Perda Total na validação {totalLossVal.item():.4f}")
        #print(f"Fim da epoca {epoca}")
    
        x = range(epocas)
        x = list(x)
        y = list(listLossTrain)
        plt.plot(x, y)
        plt.title("Loss de treino")
        # plt.ylim(0,5)
        plt.savefig(os.path.join(PATH_GRAFICOS, "treino",  "loss_de_treino.png"), dpi=300)
        plt.show()

        y_acc = list(listAccTrain)
        plt.plot(x,y_acc)
        plt.title("Acurácia de treino")
        
        plt.savefig(os.path.join(PATH_GRAFICOS, "treino", "acc_de_treino.png"), dpi=300)
        plt.show()

        y_LossTrain = list(listLossVal)
        plt.plot(x,y_LossTrain)
        plt.title("loss na validação")
        # plt.ylim(0,5)
        plt.savefig(os.path.join(PATH_GRAFICOS, "treino", "loss_de_val.png"), dpi=300)
        plt.show()

        y_accVal = list(listAccVal)
        plt.plot(x,y_accVal)
        plt.title("Acurácia na validação")
        plt.savefig(os.path.join(PATH_GRAFICOS, "treino", "acc_de_val.png"), dpi=300)
        plt.show()

    else:
         LoadModel = DigitModel()
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
         
         datasetTrain = DatasetDigits(files_train, transforms)
         datasetVal = DatasetDigits(files_val, transforms)

         dataLoaderTrain = DataLoader(datasetTrain, batch_size, True)
         dataLoaderVal = DataLoader(datasetVal, 1, False)

         for epoca in range(epocas):

            totalLossTrain = 0.0
            accuracyTotalTrain = 0.0
            totalLossVal = 0.0
            accuracyTotalVal = 0.0

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

            totalLossTrain /= len(dataLoaderTrain)
            accuracyTotalTrain /= len(dataLoaderTrain)

            listLossTrain.append(totalLossTrain.item())
            listAccTrain.append(accuracyTotalTrain.item())
            # print(f"a perda por epoca no treino é: {totalLoss}")


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
            accuracyTotalVal = accuracyTotalVal/ len(dataLoaderVal)
            totalLossVal = totalLossVal / len(dataLoaderVal)

            listAccVal.append(accuracyTotal.item())
            listLossVal.append(totalLossVal.item())

            if minLoss > totalLossVal:
                minLoss = totalLossVal
                torch.save(LoadModel.state_dict(), 'best_model/ModelDigit.pth')
                print(f"Melhor modelo encontrado!!!")
            print(f"acurácia validação: {accuracyTotalTrain.item():.4f}, Perda Total na validação {totalLossVal.item():.4f}")
            #print(f"Fim da epoca {epoca}")
            x = range(epocas)
            x = list(x)
            y = list(listLossTrain)
            plt.plot(x, y)
            plt.title("Loss de treino pós ajuste fino")
            # plt.ylim(0,5)
            plt.savefig(os.path.join(PATH_GRAFICOS, "fine_tunning", "loss_de_treino_fineTunning.png"), dpi=300)
            plt.show()

            y_acc = list(listAccTrain)
            plt.plot(x,y_acc)
            plt.title("Acurácia de treino")
            plt.savefig(os.path.join(PATH_GRAFICOS, "fine_tunning", "acc_de_treino_fineTunning"), dpi=300)
            plt.show()
            y_LossTrain = list(listLossVal)
            plt.plot(x,y_LossTrain)
            plt.title("loss na validação")
            # plt.ylim(0,5)
            plt.savefig(os.path.join(PATH_GRAFICOS, "fine_tunning", "loss_de_val_fineTunning"), dpi=300)
            plt.show()
            y_accVal = list(listAccVal)
            plt.plot(x,y_accVal)
            plt.title("Acurácia na validação")
            plt.savefig(os.path.join(PATH_GRAFICOS, "fine_tunning", "acc_de_val_fineTunning"), dpi=300)
            plt.show()


            
            
if __name__ == "__main__":
    main()