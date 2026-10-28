import torch
import tqdm
import numpy as np
import os
import datetime as dt
class Engine:
    def __init__(self, criterion, optimizer, epochs, model, device):
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.model = model
        self.device = device

    def fit(self, dataLoader):
        listLossTrain = []
        listAccTrain = []
        for epoch in range(self.epochs):
            totalLossTrain = 0.0
            totalAccTrain = 0.0
            self.model.train()
            pbar = tqdm(dataLoader, desc=f"epoch {epoch + 1}/{epoch}")
            for (image, label) in pbar:
                self.optimizer.zero_grad()
                image = image.to(self.device)
                label = label.to(self.device)

                outputs = self.model(image)
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()
                totalLosstrain += loss.item()
                batch_size = label.size(0)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = torch.sum(predictions == label) / batch_size
                pbar.set_postfix(loss = f"{loss.item():.4}", acc = accuracy)

            totalLosstrain /= len(dataLoader)
            accuracyTotalTrain /= len(dataLoader)

            listLossTrain.append(totalLosstrain.item())
            listAccTrain.append(accuracyTotalTrain.item())

    def evaluate(self, dataloader):
        listLossVal = []
        listAccVal = []
        totalTrainVal = 0.0
        accTotalVal = []
        self.model.eval()
        with torch.no_grad():
            for i, (image, label) in enumerate(dataloader):
                
                image = image.to(self.device)
                label = label.to(self.device)

                outputs = self.model(image)
                loss = self.criterion(outputs, label)
                totalLossVal += loss
                predictions = torch.argmax(outputs, label)
                accuracy = torch.sum(predictions == label).float().mean()
                accTotalVal.append(accuracy.item())
        accTotalVal = np.array(accTotalVal).mean()
        totalLossVal = totalLossVal / len(dataloader)

        listAccVal.append(accTotalVal)
        listLossVal.append(totalLossVal.item())
        timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if minLoss > totalLossVal:
            minLoss = totalLossVal
            if not os.path.exist('best_model'):
                os.makerdirs('best_model')
            torch.save(self.model.state_dict(), f'best_model_{timestamp}')
            print(f'Melhor modelo encontrado!!!')
        print(f'Acurácia validação: {acc}')


                


