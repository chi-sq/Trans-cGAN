import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"   # 这里padding = 0  输出会[1,1,26,26]
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

# x,y <- concatenate these along channels
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2 + 1,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)
    # 了解forward函数用法

    def forward(self, x, y, z):   #  x是原图,y是label或者generator生成的图, z是sober算子处理后的边缘  z的channels数目为1
        x = torch.cat([x, y, z], dim=1)  # channel 维度 concatenate
        x = self.initial(x)
        x = self.model(x)
        return x


def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))      # 286
    z = torch.randn((1, 1, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x, y, z)
    print(model)
    print(preds.shape)   # 输出[1,1,30,30]  因为要对PatchGAN 进行判别


if __name__ == "__main__":
    test()
