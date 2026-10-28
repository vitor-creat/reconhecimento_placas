import torch.nn as nn

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
            nn.Conv2d(inchannels,outchannels, (5,3), stride = 1,padding=(2,1)),
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