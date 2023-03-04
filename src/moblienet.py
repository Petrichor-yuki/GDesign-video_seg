import torch
import torch.nn as nn
import torch.nn.functional as F


class hswish(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        print("hs:", out.size(), x.size())
        return x * out


class hsigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6


# SE模块
class SEBlock(nn.Module):
    def __init__(self, channel, div=4):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.Linear(channel, channel // div),
            nn.ReLU(inplace=True),
            nn.Linear(channel // div, channel),
            hsigmoid()
        )
        # self.channel = channel
        # self.linear1 = nn.Linear(channel, channel // div)
        # self.relu = nn.ReLU(inplace=True)
        # self.linear2 = nn.Linear(channel // div, channel)
        # self.hs = hsigmoid()

    def forward(self, x):
        # out.size()[0]: batch_size
        # out.size()[1]: channel
        # out.size()[2]: height
        # out.size()[3]: width
        out = F.avg_pool2d(x, kernel_size=x.size()[2:4]).view(x.size()[0], -1)
        print("se:", out.size(), x.size())
        out = self.se(out)
        # print("channel: ", self.channel)
        # out = self.linear1(out)
        # out = self.relu(out)
        # out = self.linear2(out)
        # out = self.hs(out)

        # 重塑为四维，添加1,1两个维度
        out = out.view(out.size()[0], out.size()[1], 1, 1)
        print("se:", out.size(), x.size())
        return out * x


class Block(nn.Module):
    def __init__(self, kernel_size, expand_size, input, output, se, nl, stride):
        super(Block, self).__init__()

        self.output = output
        self.input = input
        self.kernel = kernel_size
        self.expand = expand_size

        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(input, expand_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expand_size),
            nl
        )
        self.dw_conv = nn.Sequential(
            nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=expand_size),
            nn.BatchNorm2d(expand_size),
            nl
        )
        self.pw_conv = nn.Sequential(
            nn.Conv2d(expand_size, output, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output),
        )
        self.se = se
        self.shortcut = nn.Sequential()
        if stride == 1 and input == output:
            self.residual = 1
            self.shortcut = nn.Sequential(
                nn.Conv2d(input, output, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(output)
            )
        else:
            self.residual = 0



    def forward(self, x):
        print("-----------------")
        print(self.kernel, self.expand)
        out = self.conv1(x)
        print("1", out.size())
        out = self.dw_conv(out)
        print("2", out.size())
        # 是否有注意力机制
        if self.se != None:
            out = self.se(out)
            print("3", out.size())
        out = self.pw_conv(out)

        print(out.size())
        print(x.size())
        # 是否有short cut连接
        if self.residual:
            # b, c, h, w = out.size()
            # a = self.shortcut(x).view(b, c, h, w)

            print(self.input, self.output)
            out += x

        return out


class MobileNetV3(nn.Module):
    def __init__(self, mode="Large", num_classes=1000, dropout=0.0):
        super(MobileNetV3, self).__init__()
        self.num_classes = num_classes
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.hs1 = hswish()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            hswish()
        )

        if mode == "Large":
            self.bneck = nn.Sequential(
                Block(3, 16, 16, 16, None, nn.ReLU6(inplace=True), 1),
                Block(3, 64, 16, 24, None, nn.ReLU6(inplace=True), 2),
                Block(3, 72, 24, 24, None, nn.ReLU6(inplace=True), 1),
                Block(5, 72, 24, 40, SEBlock(72), nn.ReLU6(inplace=True), 2),
                Block(5, 120, 40, 40, SEBlock(120), nn.ReLU6(inplace=True), 1),
                Block(5, 120, 40, 40, SEBlock(120), nn.ReLU6(inplace=True), 1),
                Block(3, 240, 40, 80, None, hswish(), 2),
                Block(3, 200, 80, 80, None, hswish(), 1),
                Block(3, 184, 80, 80, None, hswish(), 1),
                Block(3, 184, 80, 80, None, hswish(), 1),
                Block(3, 480, 80, 112, SEBlock(480), hswish(), 1),
                Block(3, 672, 112, 112, SEBlock(672), hswish(), 1),
                Block(5, 672, 112, 160, SEBlock(672), hswish(), 2),
                Block(5, 960, 160, 160, SEBlock(960), hswish(), 1),
                Block(5, 960, 160, 160, SEBlock(960), hswish(), 1)
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(960),
                hswish()
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(960, 1280, kernel_size=1, stride=1, bias=False),
                hswish(),
                nn.Dropout(dropout),
                nn.Conv2d(1280, self.num_classes, kernel_size=1, stride=1, bias=False)
            )

        elif mode == "Small":
            self.bneck = nn.Sequential(
                Block(3, 16, 16, 16, SEBlock(16), nn.ReLU6(), stride=2),
                Block(3, 72, 16, 24, None, nn.ReLU6(), stride=2),
                Block(3, 88, 24, 24, None, nn.ReLU6(), stride=1),
                Block(5, 96, 24, 40, SEBlock(96), hswish(), stride=2),
                Block(5, 240, 40, 40, SEBlock(240), hswish(), stride=1),
                Block(5, 240, 40, 40, SEBlock(240), hswish(), stride=1),
                Block(5, 120, 40, 48, SEBlock(120), hswish(), stride=1),
                Block(5, 144, 48, 48, SEBlock(144), hswish(), stride=1),
                Block(5, 288, 48, 96, SEBlock(288), hswish(), stride=2),
                Block(5, 576, 96, 96, SEBlock(576), hswish(), stride=1),
                Block(5, 576, 96, 96, SEBlock(576), hswish(), stride=1)
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(96, 576, kernel_size=1, stride=1),
                nn.BatchNorm2d(576),
                hswish(),
                SEBlock(576)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(576, 1024, kernel_size=1, stride=1, bias=False),
                hswish(),
                nn.Dropout(dropout),
                nn.Conv2d(1024, self.num_classes, kernel_size=1, stride=1, bias=False)
            )
        else:
            raise NotImplemented

        # 各层参数初始化
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bneck(out)
        out = self.conv2(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size[0], -1)
        out = self.conv3(out)

        return out


if __name__ == '__main__':
    net = MobileNetV3(mode="Small", num_classes=10)
    x = torch.zeros((1, 3, 224, 224))
    y = net(x)
    print("1:", y.shape())
