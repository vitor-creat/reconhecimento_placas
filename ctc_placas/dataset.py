import os
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

class PlacaDataset:
    def __init__(self, path, df, transform=None):
        self.df = df
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        image = Image.open(os.path.join(self.path, self.df['image'][idx])).convert('L')
        label = torch.tensor(data['label'], dtype=torch.int32)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


