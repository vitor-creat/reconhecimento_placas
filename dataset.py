import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
class DatasetCatAndDog(Dataset):
    def __init__(self, root, imageList,transforms = None):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.imageList = imageList
        self.str2Class = {"cat":0, "dog":1}
        
    
    def __getitem__(self, idx):
        imageName = self.imageList[idx]
        #criar uma varivael que diga o label, se é cat(0) ou dog(1)
        label = self.str2Class[imageName[0:3]]
        img_path = os.path.join(self.root, "train",imageName)
        loadImage = Image.open(img_path)
        npImage = np.array(loadImage)
        if self.transforms:
            npImage = self.transforms(npImage)
        return npImage, label
    
    def __len__(self):
        return len(self.imageList)