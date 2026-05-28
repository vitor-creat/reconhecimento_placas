import torch 
import torch.nn as nn

class CnnModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = nn.Sequential(
            ConvolucionBlock(3,32),
            nn.MaxPool2d((2,2)),
            ConvolucionBlock(32,64),
            nn.MaxPool2d((2,2)),
            ConvolucionBlock(64,128),
            ConvolucionBlock(128,256),
            nn.MaxPool2d((2,2)),
            ConvolucionBlock(256,512),
            nn.MaxPool2d((2,2))
            )
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*14*14,2)
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
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inchannels,outchannels, (3,3), padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(outchannels),
        )
    def forward(self, x):
        
        return self.block(x)