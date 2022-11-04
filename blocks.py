import math

import torch
import torch.nn as nn
from spectral import SpectralNorm

class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias,
                                   stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.relu(out)
        out = self.pointwise(out)

        return out


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_avg = self.fc(y_avg).view(b, c, 1, 1)

        return x * y_avg.expand_as(x)


class Attention_Gate(nn.Module):
    def __init__(self, F_g, F_l, F_int):  # F_g, F_l, F_int ----> 下采样channels 对称编码层channels 输出channels
        super(Attention_Gate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()

        self.chanel_in = in_dim
        self.activation = activation
        if in_dim < 8:
            self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
            self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        else:
            self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
            self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class Multi_head_Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation, num_heads=8, attn_drop_ratio=0., proj_drop_ratio=0.):
        super(Multi_head_Self_Attn, self).__init__()

        self.num_heads = num_heads
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batchsize,self.num_heads,C//self.num_heads, width * height).permute(0, 1, 3, 2)# B X num_heads X(N) X C//num_heads
        proj_key = self.key_conv(x).view(batchsize, self.num_heads,C//self.num_heads, width * height).permute(0, 1, 3, 2) # B X num_heads X(N) X C//num_heads
        proj_value = self.value_conv(x).view(batchsize, self.num_heads, C // self.num_heads, width * height).permute(
            0, 1, 3, 2)  # B X num_heads X(N) X C//num_heads
        attn = proj_query @ proj_key.transpose(-2, -1)
        attn = attn / math.sqrt(C//self.num_heads)
        attn = self.softmax(attn)  # BX num_heads X (N) X (N)
        attn = self.attn_drop(attn)
        # attn @ proj_value   BX num_heads X (N) X (C//num_heads)
        out = (attn @ proj_value).transpose(1,2).reshape(batchsize,width * height,C)
        out = self.proj_drop(out)  # [batch_size,pixel_num,total_embed_dim]
        out = out.permute(0,2,1).view(batchsize, C, width, height)

        return out, attn

class Transblock(nn.Module):

    def __init__(self,
                 in_dim,
                 num_heads,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 norm_layer=nn.LayerNorm):
        super(Transblock, self).__init__()
        # layer norm
        self.norm1 = norm_layer(in_dim)
        self.norm2 = norm_layer(in_dim)
        # point-wise mlp equals to a conv1*1
        self.mlp = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_dim, in_dim, 1, 1, 0)),
            nn.ReLU(),
        )
        # self_attention
        self.Self_Attn = Multi_head_Self_Attn(in_dim=in_dim, activation="relu",
                                                 num_heads=num_heads, attn_drop_ratio=attn_drop_ratio,
                                                 proj_drop_ratio=proj_drop_ratio)

    def forward(self, x):
        batch, channel, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, channel)
        x = self.norm1(x)
        x = x.view(batch, h, w, channel).permute(0, 3, 1, 2)
        x = x.view(batch, channel, h, w)
        x = x + self.Self_Attn(x)[0]
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, channel)
        x = self.norm2(x)
        x = x.view(batch, h, w, channel).permute(0, 3, 1, 2)
        x = x.view(batch, channel, h, w)
        x = x + self.mlp(x)
        y = x
        return y