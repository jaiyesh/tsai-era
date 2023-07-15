import torch
import torch.nn as nn
import torch.nn.functional as F
# dropout_value = 0.025
dropout_value = 0.01

class Net_s10(nn.Module):
    def __init__(self):
        super(Net_s10, self).__init__()
        # Prep Layer Conv 3x3 s1, p1) >> BN >> RELU [64k]
        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # convblock1 -Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k] 
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # resblock1 (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]  
        self.resblock1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ) # output_size = 30
        
        # convblock2 -Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [256k] 
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # convblock3 -Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k] 
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # resblock2 (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]  
        self.resblock2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ) # output_size = 30

        self.maxpool = nn.Sequential(
            nn.MaxPool2d(4,2)
#             nn.AdaptiveAvgPool2d(output_size=(1, 1))
#             nn.AvgPool2d(kernel_size=16)
        ) # output_size = 1

        self.fc = nn.Sequential(
            nn.Linear(512, 10)
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

    def forward(self, x):
        x = self.preplayer(x)
        x = self.convblock1(x)
        r1 = self.resblock1(x)
        x = x+r1
        x = self.convblock2(x)
        x = self.convblock3(x)
        r2 = self.resblock2(x)
        x = x+r2
        x = self.maxpool(x)
        x = self.fc(torch.squeeze(x))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
