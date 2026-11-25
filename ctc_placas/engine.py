import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import collections
import numpy as np
import matplotlib.pyplot as plt
import os

class Engine:
    def __init__(self, model, optimizer, criterion, batch,epochs=8,early_stop=False, device='cpu', mapping_inv=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch = batch
        self.epochs = epochs
        self.early_stop = early_stop
        self.device = device
        self.mapping_inv = mapping_inv
        self.val_vis = 'saved_images'

        if not os.path.exists(self.val_vis):
            os.makedirs(self.val_vis)

    def _decode_ctc(self, out, target):
        preds = out.permute(1, 0, 2).argmax(dim=2)
        max_len = target.size(1)
        decoded_preds = torch.zeros((preds.size(0), max_len), dtype=target.dtype, device=self.device)

        for i, p in enumerate(preds):
            p = torch.unique_consecutive(p)
            p = p[p > 0] 
            length = min(len(p), max_len)
            decoded_preds[i, :length] = p[:length]

        return decoded_preds

    def fit(self, dataloader, val_dataloader):
        hist_loss = []
        for epoch in range(self.epochs):
            self.model.train()
            tk = tqdm(dataloader, total=len(dataloader))
            loss_epoch = []
            acc_epoch = []
            for data, target in tk:
                data = data.to(device=self.device)
                target = target.to(device=self.device)

                self.optimizer.zero_grad()

                out, loss = self.model(data, target, criterion=self.criterion)

                loss_epoch.append(loss.item())

                loss.backward()

                self.optimizer.step()

                self.prediction = self._decode_ctc(out, target)

                # print(self.prediction)
                # print(target)
                # input()

                acc = torch.sum(self.prediction == target) / target.numel()
                acc_epoch.append(acc.item())

                tk.set_postfix({'Epoch': epoch + 1, 'Loss': loss.item(), 'acc': acc.item()})

            totalLosstrain = np.array(loss_epoch).mean()
            accuracyTotalTrain = np.array(acc_epoch).mean()
            print(f'epoch loss: {totalLosstrain:.4f} | epoch acc: {accuracyTotalTrain:.4f}')

            totalLossVal = 0.0
            totalAccVal = 0.0
            self.model.eval()
            pbar = tqdm(val_dataloader, total=len(val_dataloader))
            with torch.no_grad():
                for (image, label) in pbar:
                    image = image.to(self.device)
                    label = label.to(self.device)
                    
                    outputs, loss = self.model(image, label, criterion=self.criterion)
                    
                    totalLossVal += loss.item()
                    predictions = self._decode_ctc(outputs, label)

                    accuracy = torch.sum(predictions == label) / label.numel()
                    totalAccVal += accuracy.item()
                    pbar.set_postfix(loss=f"{loss.item():.4f}", acc=accuracy.item())
                
                self.visualize_sample(image[0], predictions[0], epoch)

            totalLossVal /= len(val_dataloader)
            totalAccVal /= len(val_dataloader)
            print(f"Epoch {epoch + 1}: Loss train: {totalLosstrain:.4f}, Acc train: {accuracyTotalTrain:.4f}, Loss val: {totalLossVal:.4f}, Acc val: {totalAccVal:.4f}")


    def visualize_sample(self, tensor, pred, epoch):
        image = T.ToPILImage()(tensor[0].cpu())
        pred = pred[pred > 0]
        pred_name = ''.join([self.mapping_inv[int(i)] for i in pred.cpu().numpy()])
        image.save(os.path.join(self.val_vis, f'epoch_{epoch}_{pred_name}.png'))

    def evaluate(self, dataloader):
        self.model.eval()
        loss = 0
        hist_loss = []
        outs = collections.defaultdict(list)
        tk = tqdm(dataloader, total=len(dataloader))
        with torch.no_grad():
            for data, target in tk:
                data = data.to(device=self.device)
                target = target.to(device=self.device)

                out, loss = self.model(data, target, criterion=self.criterion)

                outs['pred'].append(out)
                outs['target'].append(target)


                hist_loss.append(loss)

                tk.set_postfix({'Loss':loss.item()})

        return outs, hist_loss

    def predict(self, image):
        image = Image.open(image).convert('L')
        transform_val = T.Compose([
            T.Resize((256, 512)),
            T.ToTensor(),
        ])
        image_tensor = transform_val(image)
        image_tensor = image_tensor.unsqueeze(0)
        out, _ = self.model(image_tensor.to(device=self.device))
        out = out.permute(1, 0, 2)
        out = out.log_softmax(2)
        out = out.argmax(2)
        out = out.cpu().detach().numpy()

        return out




