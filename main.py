import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
# from torchvision.transforms import v2
import torchvision.transforms as T
class Dataset:
    def __init__(self, root):
        self.root = root
        self.classes = []
        self.transform = T.RandomHorizontalFlip(p=0.5)
        self.imgs = []
        self.cls = []
        for pasta in os.listdir(self.root):
            self.classes.append(pasta)
            for arquivos in os.listdir(os.path.join(self.root, pasta)):
                self.imgs.append(os.path.join(pasta , arquivos))
                self.cls.append(len(self.classes)-1)
                # print(pasta, arquivos)

    def __getitem__(self, key):
        img_path = os.path.join(self.root, self.imgs[key])
        image = Image.open(img_path)
        image_array = np.array(image)
        label = self.cls[key]
        return image_array, label

    def __len__(self):
        return len(self.imgs)




def main ():
    teste = Dataset('/media/vitor/data/CNN_letters_custom')
    img_get_item, label = teste.__getitem__(2200)
    img = transform(image)
    print(label)
    plt.imshow(img_get_item)
    plt.show()
    print(teste.classes[label])


if "__main__" == __name__:
    main()