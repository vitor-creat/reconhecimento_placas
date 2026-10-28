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
import datetime
import sys

class Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, menssage):
        self.terminal.write(menssage)
        self.log.write(menssage)
        self.log.flush()

    def flush(self):
        pass

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", required=False, type=int, default=16, help='Number of epochs for training (default: 16)')
    parser.add_argument("--lr", type=float, default=0.001, help='Learning rate for training (default: 0.001)')
    parser.add_argument("--batch",type=int, default=16, help='Batch size for training (default: 16)')
    parser.add_argument("--fine_tuning", type=bool, default=False, help='Whether to perform fine-tuning on the model (default: False)')
    parser.add_argument("--dataset-path", type=str, default="/media/vitor/data/CNN_letters_custom", help='Path to the dataset directory')
    parser.add_argument("--session", type=str, default='', help='Session name for logging (default: timestamp)')
    args = parser.parse_args()

    datasetPath = args.dataset_path
    
    #Parametros
    epocas = args.epocas
    learning_rate = args.lr
    batch_size = args.batch
    fine_tuning = args.fine_tuning
    minLoss = 5.0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    PATH_GRAFICOS = "grafico"

    if not os.path.exists(PATH_GRAFICOS):
        os.makedirs(os.path.join(PATH_GRAFICOS, "treino"), exist_ok=True)
        os.makedirs(os.path.join(PATH_GRAFICOS, "fine_tunning"), exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_name = f"{args.session}_{timestamp}" if args.session else timestamp
    log_file = f"logs/{session_name}.log"

    sys.stdout = Logger(log_file, sys.stdout)
    sys.stderr = Logger(log_file, sys.stderr)
    print(f"Starting training session: {session_name}")
    print(f"Logging to: {log_file}")
    print('='* 50)

    digitsClass = []
    file_list = []
    str2cls = {}
    for i, pastas in enumerate(os.listdir(datasetPath)):
        for imgs in os.listdir(os.path.join(datasetPath, pastas)):
            digitsClass.append(pastas)
            file_list.append(os.path.join(datasetPath, pastas, imgs))
            str2cls[pastas] = i
            
    file_list_tupla = list(zip(digitsClass, file_list))
    random.shuffle(file_list_tupla)

    
    files_train = file_list_tupla[:int(len(file_list_tupla)*0.8)]
    files_val = file_list_tupla[int(len(file_list_tupla)*0.8):]

    if not os.path.exists("train_files.txt"):
        with open ('train_files_digits.txt', "w") as f:
            for file in files_train:
                # print(file)
                f.write(file[1] + '\n')

    if not os.path.exists("val_files.txt"):
        with open ('val_files_digits.txt', "w") as f:
            for file in files_val:
                f.write(file[1] + '\n')
    
    if not os.path.exists('str2cls.txt'):
        with open('str2cls.txt', 'w') as f:
            for key, value in str2cls.items():
                f.write(f"{key}:{value}\n")

    if fine_tuning == False:

        transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((128,128)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        datasetTrain = DatasetDigits(files_train, transforms, str2cls)
        datasetVal = DatasetDigits(files_val, transforms, str2cls)

        dataLoaderTrain = DataLoader(datasetTrain, batch_size, True)
        dataLoaderVal = DataLoader(datasetVal, 1, False)

        Cnn = DigitModel(n_classes=len(str2cls))
        image, label = next(iter(dataLoaderTrain))

        Cnn = Cnn.to(device)
        criterio = CrossEntropyLoss().to(device)
        otimizador = SGD(Cnn.parameters(),learning_rate)

        listLossTrain = []
        listLossVal = []
        listAccTrain = []
        listAccVal = []

        print(f"Treinamento iniciando com {len(files_train)} imagens de treino e {len(files_val)} imagens de validação.")
        print(f"Número de classes: {len(str2cls)}")
        print(f"Treinamento com {epocas} épocas.")

        for epoca in range(epocas):
            
            totalLossTrain = 0.0
            accuracyTotalTrain = 0.0
            totalLossVal = 0.0
            accuracyTotalVal = []

            Cnn.train(True)
            iteration = 0
            pbar = tqdm(dataLoaderTrain, desc=f"epoca {epoca}")
            for (image, label) in pbar:
                otimizador.zero_grad()
                image = image.to(device)
                label = label.to(device)
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

                    image = image.to(device)
                    label = label.to(device)

                    inferencia = Cnn(image)
                    loss = criterio(inferencia, label)
                    totalLossVal += loss
                    predictions = torch.argmax(inferencia, dim=1)
                    tensorComparacao = predictions == label
                    accuracy = torch.sum(tensorComparacao)
                    accuracyTotalVal.append(accuracy.item()) 

            accuracyTotalVal = np.array(accuracyTotalVal).mean()
            totalLossVal = totalLossVal / len(dataLoaderVal)

            listAccVal.append(accuracyTotalVal)
            listLossVal.append(totalLossVal.item())

            if minLoss > totalLossVal:
                minLoss = totalLossVal
                if not os.path.exists('best_model'):
                    os.makedirs('best_model')
                torch.save(Cnn.state_dict(), 'best_model/ModelDigit.pth')
                print(f"Melhor modelo encontrado!!!")
            print(f"acurácia validação: {accuracyTotalTrain.item():.4f}, Perda Total na validação {totalLossVal.item():.4f}")
    
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
        print(listAccVal)

        with open('loss_treino.txt', 'w') as file:
            for l in listLossTrain:
                file.writelines(l)

        with open('acc_treino.txt', 'w') as file:
            for l in listAccTrain:
                file.writelines(l)

        with open('loss_val.txt', 'w') as file:
            for l in listLossVal:
                file.writelines(l)

        with open('acc_val.txt', 'w') as file:
            for l in listAccVal:
                file.writelines(l)

    else:
        import torch.nn as nn

        LoadModel = DigitModel()
        LoadModel.load_state_dict(torch.load('best_model/ModelDigit.pth'))

        topclf = nn.Linear(128, len(str2cls))
        LoadModel.cls[4] = topclf

        LoadModel.to(device)
        criterio = CrossEntropyLoss().to(device)
        otimizador = SGD(LoadModel.parameters(),learning_rate)

        transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size=(128, 128), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        datasetTrain = DatasetDigits(files_train, transforms, str2cls)
        datasetVal = DatasetDigits(files_val, transforms, str2cls)

        dataLoaderTrain = DataLoader(datasetTrain, batch_size, True)
        dataLoaderVal = DataLoader(datasetVal, 1, False)

        print(f"Fine tuning iniciando com {len(files_train)} imagens de treino e {len(files_val)} imagens de validação.")
        print(f"Número de classes: {len(str2cls)}")
        print(f"Treinamento com {epocas} épocas.")


        for epoca in range(epocas):

            totalLossTrain = 0.0
            accuracyTotalTrain = 0.0
            totalLossVal = 0.0
            accuracyTotalVal = 0.0

            LoadModel.train(True)

            for iteration, (image, label) in enumerate(dataLoaderTrain):
                otimizador.zero_grad()
                image = image.to(device)
                label = label.to(device)
                inferencia = LoadModel(image)
                loss = criterio(inferencia, label)
                loss.backward()
                otimizador.step()
                totalLossTrain += loss
                
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

                    image = image.to(device)
                    label = label.to(device)
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