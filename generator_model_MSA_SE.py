import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from spectral import SpectralNorm
from blocks import Self_Attn, Multi_head_Self_Attn, SE_Block
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
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        # 256
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )  # 128

        # self_attention
        self.Self_Attn0 = Multi_head_Self_Attn(in_dim=features, activation="relu",
                                               num_heads=8, attn_drop_ratio=0.1, proj_drop_ratio=0.1)
        self.Self_Attn1 = Multi_head_Self_Attn(in_dim=features * 2, activation="relu",
                                               num_heads=8, attn_drop_ratio=0.1, proj_drop_ratio=0.1)
        self.Self_Attn2 = Multi_head_Self_Attn(in_dim=features * 4, activation="relu",
                                               num_heads=8, attn_drop_ratio=0.1, proj_drop_ratio=0.1)
        self.Self_Attn3 = Multi_head_Self_Attn(in_dim=features * 8, activation="relu",
                                               num_heads=8, attn_drop_ratio=0.1, proj_drop_ratio=0.1)
        self.Self_Attn4 = Multi_head_Self_Attn(in_dim=features * 8 * 2, activation="relu",
                                               num_heads=8, attn_drop_ratio=0.1, proj_drop_ratio=0.1)
        # self.Self_Attn_last = Self_Attn(in_dim=in_channels, activation="relu")
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False) # 64
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
        # define bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )  # 1✖1
        # squeeze and excite
        self.se1 = SE_Block(features)
        self.se2 = SE_Block(features * 2)
        self.se3 = SE_Block(features * 4)
        self.se = SE_Block(features * 8)
        # Attention_Gate
        # self.Att3 = Attention_Gate(F_g=features * 4, F_l=features * 4, F_int=features)
        # self.Att2 = Attention_Gate(F_g=features * 2, F_l=features * 2, F_int=features)
        # self.Att1 = Attention_Gate(F_g=features * 1, F_l=features * 1, F_int=features)
        # self.Att = Attention_Gate(F_g=features * 8, F_l=features * 8, F_int=features)

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)  # 2

        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True   # concatenate
        )  # 4
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )  # 8
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )  # 16
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )  # 32
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )  # 64
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)  # 128
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            # Self_Attn(in_channels, "relu"),
            # nn.Tanh(),  # we want each pixel values between [-1,1]
        )  # 256

    def forward(self, x):
        d1 = self.initial_down(x)
        d1 = self.se1(d1)
        d2 = self.down1(d1)
        d2 = self.se2(d2)
        # d2, p1 = self.Self_Attn1(d2)
        d3 = self.down2(d2)
        d3 = self.se3(d3)
        # d3, p2 = self.Self_Attn2(d3)
        d4 = self.down3(d3)
        d4 = self.se(d4)
        # d4, p3 = self.Self_Attn3(d4)
        d5 = self.down4(d4)
        # d5, p4 = self.Self_Attn3(d5)
        d6 = self.down5(d5)
        d6, p5 = self.Self_Attn3(d6)
        d7 = self.down6(d6)
        d7, p6 = self.Self_Attn3(d7)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        # d7 = self.Att(up1, d7)
        out, p_1 = self.Self_Attn4((torch.cat([up1, d7], 1)))
        up2 = self.up2(out)
        # d6 = self.Att(up2, d6)
        out, p_2 = self.Self_Attn4((torch.cat([up2, d6], 1)))
        up3 = self.up3(out)  # 512
        # d5 = self.Att(up3, d5)
        # out, p_3 = self.Self_Attn4((torch.cat([up3, d5], 1)))
        up4 = self.up4((torch.cat([up3, d5], 1)))  # 512
        up4 = self.se(up4)
        # d4 = self.Att(up4, d4)
        # out, p_4 = self.Self_Attn4((torch.cat([up4, d4], 1)))
        up5 = self.up5(torch.cat([up4, d4], 1))  # 256
        up5 = self.se3(up5)
        # d3 = self.Att3(up5, d3)             # up5 256   d3 256
        # out, p_5 = self.Self_Attn3((torch.cat([up5, d3], 1)))
        up6 = self.up6(torch.cat([up5, d3], 1))  # 128
        up6 = self.se2(up6)
        # d2 = self.Att2(up6, d2)   # 128 128
        # out, p_6 = self.Self_Attn2((torch.cat([up6, d2], 1)))
        up7 = self.up7(torch.cat([up6, d2], 1))   # 64
        up7 = self.se1(up7)
        # d1 = self.Att1(up7, d1)   # 64 64
        # out, p_7 = self.Self_Attn1((torch.cat([up7, d1], 1)))    # 这一层第一次实验时没加
        out = torch.cat([up7, d1], 1)
        out = self.final_up(out)
        # out, attention_matrix_global = self.Self_Attn_last(out)
        return torch.tanh(out)
        # global 除了initial_down 和 final_up 都加了self_attention


def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)  # torch.Size([1, 3, 256, 256])


if __name__ == "__main__":
    test()
