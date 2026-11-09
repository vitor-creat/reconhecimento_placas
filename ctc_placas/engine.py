import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import collections
import numpy as np

class Engine:
    def __init__(self, model, optimizer, criterion, epochs=50, early_stop=False, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.early_stop = early_stop
        self.device = device

    def fit(self, dataloader):
        hist_loss = []
        for epoch in range(self.epochs):
            self.model.train()
            tk = tqdm(dataloader, total=len(dataloader))
            loss_epoch = [];
            for data, target in tk:
                data = data.to(device=self.device)
                target = target.to(device=self.device)

                self.optimizer.zero_grad()

                out, loss = self.model(data, target, criterion=self.criterion)

                loss_epoch.append(loss.item())

                loss.backward()

                self.optimizer.step()


                tk.set_postfix({'Epoch':epoch+1, 'Loss' : loss.item()})
            loss_epoch = np.array(loss_epoch).mean()
            print(f'epoch loss: {loss_epoch}')

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




