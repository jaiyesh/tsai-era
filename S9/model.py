import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.01

class depthwise_separable_conv(nn.Module):
     def __init__(self, nin, kernels_per_layer, nout): 
       super(depthwise_separable_conv, self).__init__() 
       self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin) 
       self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1) 

     def forward(self, x): 
       out = self.depthwise(x) 
       out = self.pointwise(out) 
       return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),         # ) output_size = 32
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),         # ) output_size = 32
            depthwise_separable_conv(nin=16, kernels_per_layer=3, nout=16),
#             nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1, padding=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 30

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),   # ) output_size = 30
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),   # ) output_size = 30
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=0, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 30

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),  # ) output_size = 30
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1,dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),   # ) output_size = 30
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 28

        # OUTPUT BLOCK
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value), # )  output_size = 28
            nn.Conv2d(in_channels=64, out_channels=48, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(dropout_value), # )  output_size = 28
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 28

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
#             nn.AvgPool2d(kernel_size=16)
        ) # output_size = 1

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value), # )  output_size = 20

            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = self.convblock5(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)