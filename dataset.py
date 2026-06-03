import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
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
    
class DatasetDigits(Dataset):
    def __init__(self, imageList,transforms = None):
        super().__init__()
        # print(len(imageList), type(imageList))
        self.transforms = transforms
        self.imageList = imageList
        self.str2Class = {'X':0, '9':1, 'C':2, 'H':3, 'P':4, 'R':5, 'U':6, 'Z':7, 'E':8, '2':9, 'L':10, '5':11, 'T':12, 'F':13, 'A':14, 'M':15, 'W':16, '0':17, 'Y':18, 'D':19, 'S':20, '1':21, 'I':22, 'G':23, 'V':24, 'N':25, '3':26, '8':27, '6':28, 'Q':29, '4':30, '7':31, 'B':32, 'J':33, 'K':34}
        
    
    def __getitem__(self, idx):
        imageName = self.imageList[idx][1]
        className = self.imageList[idx][0]
        label = self.str2Class[className]
        # print(imageName)
        # input()
        loadImage = Image.open(imageName)
        npImage = np.array(loadImage)
        npImage = npImage[...,0:3]
        if self.transforms:
            npImage = self.transforms(npImage)
        return npImage, label
    
    def __len__(self):
        return len(self.imageList)