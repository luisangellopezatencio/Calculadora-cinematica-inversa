import torch
from torch import nn

class CNN2D(nn.Module):
    def __init__(self, stride, padding, kernel, out_channels ):
        super(CNN2D, self).__init__()
        self.capa1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.Tanh()
        )
        self.capa2 = nn.Sequential(
            nn.Conv2d(in_channels = 6, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.Tanh()
        )
        self.capa_fc = nn.Sequential(
                 nn.Linear(6*6*5, 256),  #6canales*5height*4weight
                 nn.Sigmoid(),
                 nn.Linear(256, 128),
                 nn.Sigmoid(),
                 nn.Linear(128, 6),
                 nn.Tanh()
                )



    def forward(self, x):
        x = self.capa1(x)   #Capa1
        x = self.capa2(x)   #Capa2
        x = torch.flatten(x,1) 
        x = self.capa_fc(x)
        return(x)