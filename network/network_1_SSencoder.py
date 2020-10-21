#需要参照上述改动重写

import math
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

#把第一层换为2fc
class Generator_SS(nn.Module):
    def __init__(self, input_dim=256, output_channels=3, image_size=256, scale=8):
        super().__init__()
        layers = []
        up_times = math.log(image_size,2)- 3 # 减去前两次 1->2->4， 及最后一次， 方便中间写循环
        first_hidden_dim = input_dim*scale # 这里对应输入维度，表示《输入维度》对应《网络中间层维度（起点）》的放大倍数
        bias_flag = False

        self.fc = nn.Sequential(
            nn.Linear(256, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048*16),#[1024,128*4*4]-input_size=32
            nn.BatchNorm1d(2048*16),
            nn.ReLU(),
        )

        # 1: 1x1 -> 4x4
        # layers.append(nn.ConvTranspose2d(input_dim, first_hidden_dim, kernel_size=4,stride=1,padding=0,bias=bias_flag))
        # layers.append(nn.BatchNorm2d(first_hidden_dim))
        # layers.append(nn.ReLU())

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> 32*32
        hidden_dim = first_hidden_dim
        while up_times>0:
            layers.append(nn.ConvTranspose2d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1 ,bias=bias_flag))
            layers.append(nn.BatchNorm2d(hidden_dim//2))
            layers.append(nn.ReLU())
            up_times = up_times - 1
            hidden_dim = hidden_dim // 2

        # 3:end 
        layers.append(nn.ConvTranspose2d(hidden_dim,output_channels,kernel_size=4, stride=2, padding=1, bias=bias_flag))
        layers.append(nn.Tanh())

        # all
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1,2048,4,4)
        x = self.net(x)
        return x

class Discriminator_SS(nn.Module):
    def __init__(self, input_dim=256, input_channels=3, image_size=256, scale=4):
        super().__init__()
        layers=[]
        up_times = math.log(image_size,2)- 3
        first_hidden_dim = input_dim * scale // 16 # 默认为input_dim 
        bias_flag = False

        # 1:
        layers.append(spectral_norm(nn.Conv2d(input_channels, first_hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 2: 64*64 > 4*4
        hidden_dim = first_hidden_dim
        while up_times>0:  
            layers.append(spectral_norm(nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            hidden_dim = hidden_dim * 2
            up_times = up_times - 1

        # 3: 4*4 > 1*1
        #layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=4, stride=1, padding=0))
        ##layers.append(nn.Sigmoid())
        # all:
        self.net = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(2048*16,2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 256),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        y = self.net(x)
        y = y.view(-1,2048*16)
        y = self.fc(y)
        return y # [1,1,1,1]
