import torch 
import torch.nn as nn

class CnnModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = nn.Sequential(
            DoubbleConv(3,32),
            DoubbleConv(32,64),
            nn.MaxPool2d((2,2)),
            DoubbleConv(64,128),
            DoubbleConv(128,256),
            nn.MaxPool2d((2,2)),
            DoubbleConv(256,512),
            DoubbleConv(512,1024)
            )
        self.cls = nn.Sequential(
            nn.Flatten(),
            #mudei de 2 para 32, pois antes tinhas apenas 2 classes (gato e cachorro), agora temos 35 (0-9 e a-z)
            nn.Linear(1024*56*56,2)
        )




    def forward(self, x):
        # print(x.shape)
        x = self.backbone(x)
        # print(x.shape)
        # input()
        x = self.cls(x)
        #print(x.shape)
        return x


class DigitModel(nn.Module):

    def __init__(self, n_classes = 35):
        super().__init__()
        self.backbone = nn.Sequential(
            SkipConv(3,32),
            nn.MaxPool2d((2,2)),
            SkipConv(32,64),
            nn.MaxPool2d((2,2)),
            SkipConv(64,128),
            nn.MaxPool2d((2,2)),
            SkipConv(128,256)
            )
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*16*16,1024),
            nn.Linear(1024, 512),
            nn.Linear(512,128),
            nn.Linear(128,n_classes)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.backbone(x)
        # print(x.shape)
        # input()
        x = self.cls(x)
        #print(x.shape)
        return x

# 3 32
class ConvolucionBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(ConvolucionBlock, self).__init__()
        # print(type(inchannels), type(outchannels))
        self.block = nn.Sequential(
            nn.Conv2d(inchannels,outchannels, (3,3), stride = 1,padding="same"),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),

            # nn.Conv2d(outchannels, outchannels, (3,3), stride = 1,padding="same"),
            # nn.BatchNorm2d(outchannels),
            # nn.ReLU(inplace=True),

        )
    def forward(self, x):
        
        return self.block(x)
    
class DoubbleConv(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(DoubbleConv, self).__init__()
        self.block = nn.Sequential(
           ConvolucionBlock(inchannels, int(outchannels/2)),
           ConvolucionBlock(int(outchannels/2), outchannels)

        )
    def forward(self, x):
        
        return self.block(x)

class SkipConv(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(SkipConv, self).__init__()
        self.block1=  DoubbleConv(inchannels, int(outchannels/2))

        self.block2 = DoubbleConv(int(outchannels/2), outchannels)

        self.downsample1 =  nn.Conv2d(inchannels, int(outchannels/2), (1,1))
        self.downsample2 =  nn.Conv2d(int(outchannels/2), outchannels, (1,1))
        
    

    def forward(self, x):

        xBlock1 = self.block1(x) + self.downsample1(x)
        xBlock2 = self.block2(xBlock1) + self.downsample2(xBlock1)
        ativacao = nn.ReLU()
        return ativacao(xBlock2)