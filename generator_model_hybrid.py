import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from spectral import SpectralNorm
from blocks import Transblock, SE_Block
# from blocks import Multi_head_Self_Attn

# 输出 为 torch.Size([1, 3, 256, 256])
def create2DsobelFilter():

    sobelFilter_y = np.array([[[[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]],
                            [[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]],
                            [[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]],
                            ]])
    sobelFilter_x = np.array([[[[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]],
                               [[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]],
                               [[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]],
                               ]])
    sobelFilter = np.concatenate((sobelFilter_x, sobelFilter_y), axis = 0)
    return Variable(torch.from_numpy(sobelFilter).type(torch.cuda.FloatTensor))

def sobelLayer(input):
    kernel = create2DsobelFilter()
    act = nn.Tanh()
    fake_sobel = F.conv2d(input, kernel, padding=1, groups=1)/8     #  [-1,1] fake_sobel torch.Size([1, 2, 256, 256])
    # n,c,h,w = fake_sobel.size()
    # 转换为梯度norm2
    fake = torch.norm(fake_sobel,p=2,dim=1,keepdim=True)    # torch.Size([1, 1, 256, 256])  得到的梯度图
    fake_out = act(fake)
    # fake_out = act(fake)*2-1    # 使得输出[-1,1]

    return fake_out  # 得到归一化的边缘图

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            SpectralNorm(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            )
            if down
            else SpectralNorm(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down
        self.SpectralNorm = SpectralNorm

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64, norm_layer=None):
        super().__init__()
        # 256
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )  # 128
        # down
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)  # 64
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )  # 32
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )  # 16
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )  # 8
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )  # 4
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )  # 2
        self.trans = Transblock(in_dim=features*8, num_heads=8, attn_drop_ratio=0.1, proj_drop_ratio=0.1)

        # squeeze and excite
        self.se1 = SE_Block(features)
        self.se2 = SE_Block(features * 2)
        self.se3 = SE_Block(features * 4)
        self.se = SE_Block(features * 8)

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)  # 32

        self.up2 = Block(
            features * 8*2, features * 4, down=False, act="relu", use_dropout=True   # concatenate
        )  # 64
        self.up3 = Block(
            features * 4*2, features*2, down=False, act="relu", use_dropout=True
        )  # 128
        self.up4 = Block(
            features * 4, features, down=False, act="relu", use_dropout=True
        )  # 128
        self.up5 = nn.Sequential(
            Block(features * 2, 32, down=False, act="relu", use_dropout=False),
            nn.Conv2d(32, in_channels, 3, 1, 1),
            nn.Tanh()
        )  # 256

    def forward(self, x):
        # patch embedding
        d1 = self.initial_down(x)   # 128 dim=64
        d1 = self.se1(d1)
        d2 = self.down1(d1)           # 64 dim=128

        d2 = self.se2(d2)
        d3 = self.down2(d2)  # 32 dim = 256
        d3 = self.se3(d3)

        d4 = self.down3(d3)  # 16  dim=512
        d4 = self.se(d4)
        d5 = self.down4(d4)  # 8  dim=512

        # transformer
        d5 = self.trans(d5)
        d5 = self.trans(d5)
        d5 = self.trans(d5)
        d5 = self.trans(d5)
        d5 = self.trans(d5)
        d5 = self.trans(d5)

        up1 = self.up1(d5)  # 16 dim = 512
        up2 = self.up2(torch.cat((up1, d4), 1))  # 32 dim = 256
        up3 = self.up3(torch.cat((up2, d3), 1))  # 64 dim = 128
        up4 = self.up4(torch.cat((up3, d2), 1))  # 128 dim = 64
        up5 = self.up5(torch.cat((up4, d1), 1))
        return up5


def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)  # torch.Size([1, 3, 256, 256])


if __name__ == "__main__":
    test()
