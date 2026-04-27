import torch 
import torch.nn as nn

class CnnModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(3,32, (3,3), bias=False)
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*222*222,2)
        )




    def forward(self, x):
        #print(x.shape)
        x = self.conv(x)
        #print(x.shape)
        x = self.cls(x)
        #print(x.shape)
        return x