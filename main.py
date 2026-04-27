from dataset import DatasetCatAndDog
import os
import torch
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model import CnnModel
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
def main():
    listImage = []
    
    for i in os.listdir("/media/vitor/data/dogs-vs-cats/train"):
        listImage.append(i)
    transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    X_train, X_val = train_test_split(
    listImage, test_size=0.20, shuffle=False
)
    datasetTrain = DatasetCatAndDog('/media/vitor/data/dogs-vs-cats/',X_train, transforms)
    datasetVal = DatasetCatAndDog('/media/vitor/data/dogs-vs-cats/',X_val, transforms)

    dataLoaderTrain = DataLoader(datasetTrain, 16, True)
    dataLoaderVal = DataLoader(datasetVal, 1, False)

    testCnn = CnnModel()
    image, label = next(iter(dataLoaderTrain))
    resultado = testCnn(image)

    maior = torch.argmax(resultado, dim=1)
    print(testCnn)
    print(resultado.shape, maior.shape)
    print(maior)
    print(label)
    epocas = 10
    learning_rate = 0.002
    for i in range(epocas):
        for image, label in dataLoaderTrain:
            image = image.cuda()

if __name__ == "__main__":
    main()