import os
from PIL import Image
import numpy as np
# class Dataset:
#     def __init__(self, nome, sobrenome):
#         self.nome = nome
#         self.sobrenome = sobrenome
#     def unir_nomes(self):
#         self.nome_completo = self.nome + " " +self.sobrenome
#         print(f"O nome completo é: {self.nome_completo}.")

# nome = 'Vitor'
# sobrenome = 'Padilha'

# pessoa = Dataset(nome, sobrenome)

# pessoa.unir_nomes()

# import os
# import pandas as pd
# from torchvision.io import decode_image

# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = decode_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label

# pasta_raiz = '/media/vitor/data/CNN_letters_custom'

# for raiz, pastas, arquivos in os.walk(pasta_raiz):
#     for nome_pasta in pastas:
#         print(f"o nome das pasta são: {os.path.join(raiz,nome_pasta)}")
#     for nome_arquivos in arquivos:
#         print(f"o nome das pasta são: {os.path.join(raiz,nome_arquivos)}")

#capturar no construtor usando a os para verificar as subpastas. cada uma vai virar uma classe



class Dataset:
    def __init__(self, root):
        self.root = root
        self.classes = []
        self.imgs = []
        self.cls = []
        for pasta in os.listdir(root):
            self.classes.append(pasta)
            for arquivos in os.listdir(os.path.join(root, pasta)):
                self.imgs.append(os.path.join(pasta,arquivos))
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
#    print(img_get_item)
    print(label)
    print(teste.classes[label])
#    print(len(teste))

if "__main__" == __name__:
    main()