import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
# from torchvision.transforms import v2
import torchvision.transforms as T
from torch.utils.data import DataLoader


class Dataset:
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.classes = []
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
        # print(np.array(image).shape, img_path)
        image = self.transform(image)
        image_array = np.array(image)
        label = self.cls[key]
        return image_array, label

    def __len__(self):
        return len(self.imgs)




def dataLoader(i, n, dataset):
    img_array_list= []
    for x in range(i, 1+n):
        image_array, _, image= dataset.__getitem__(x)

        image = image.resize((100,100))
        image_array = np.array(image)

        img_array_list.append(image_array)
    
    img_stack = np.stack(img_array_list)
    # print(img_stack.shape)
    return  img_stack

def main ():
    transform_rezise = T.Resize((100,100))
    teste = Dataset('/media/vitor/data/CNN_letters_custom', transform_rezise)
    # teste = Dataset('/home/vitor/Documents/CNN_letters_custom')
    # img_get_item, label, image = teste.__getitem__(2200)
    # converted_image = Image.fromarray(img_get_item)
    # transformHorizontal = T.RandomHorizontalFlip(p=0.5)
    # transformVertical = T.RandomVerticalFlip(p=0.5)
    # img = transformHorizontal(converted_image)
    # imgVertical = transformVertical(converted_image)
    # print(label)
    # plt.imshow(img)
    # plt.imshow(imgVertical)
    # plt.show()
    # print(teste.classes[label])

    # dataload = dataLoader(1,2200,teste)
    # for i in dataload:
    #     plt.imshow(i)
    #     plt.show()

    # dataload = dataLoader(1,2200,teste)
    # for i in dataload:
    #     for _ in range(4):
    #         transform = T.RandomVerticalFlip(p=0.7)
    #         arrayForImage = Image.fromarray(i)
    #         img_transform = transform(arrayForImage)
    #         plt.imshow(img_transform)
    #         plt.show()
    train_dataloader = DataLoader(teste, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    x_arr, _ = next(iter(train_dataloader))
    print(x_arr.shape)
    

if "__main__" == __name__:
    main()