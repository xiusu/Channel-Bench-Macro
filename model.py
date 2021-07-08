import torch.nn as nn
import torch
from collections import OrderedDict

candidate_OP = ['id', 'ir_3x3_t3', 'ir_5x5_t6']
OPS = OrderedDict()
OPS['id'] = lambda inp, oup, stride: Identity(inp=inp, oup=oup, stride=stride)
OPS['ir_3x3_t3'] = lambda inp, oup, stride: InvertedResidual(inp=inp, oup=oup, t=3, stride=stride, k=3)
OPS['ir_5x5_t6'] = lambda inp, oup, stride: InvertedResidual(inp=inp, oup=oup, t=6, stride=stride, k=5)


class Identity(nn.Module):
    def __init__(self, inp, oup, stride):
        super(Identity, self).__init__()
        if stride != 1 or inp != oup:
            self.downsample = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return x



class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, t, k=3, activation=nn.ReLU, use_se=False, **kwargs):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.t = t
        self.k = k
        self.use_se = use_se
        assert stride in [1, 2]
        hidden_dim = round(inp * t)
        if t == 1:
            self.conv = nn.Sequential(
                # dw            
                nn.Conv2d(hidden_dim, hidden_dim, k, stride, padding=k//2, groups=hidden_dim, 
                              bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, k, stride, padding=k//2, groups=hidden_dim, 
                              bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        self.use_shortcut = inp == oup and stride == 1

    def forward(self, x):
        if self.use_shortcut:
            return self.conv(x) + x
        return self.conv(x)


class InvertedResidual_easy(nn.Module):
    def __init__(self, channels, stride=1, t=6, k=3, activation=nn.ReLU, use_se=False):
        super(InvertedResidual_easy, self).__init__()
        self.stride = stride
        self.t = t
        self.k = k
        self.use_se = use_se
        assert stride in [1, 2]
        inp = channels[0]
        hidden_dim = channels[1]
        oup = channels[2]
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, k, stride, padding=k//2, groups=hidden_dim, 
                            bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.use_shortcut = inp == oup and stride == 1

    def forward(self, x):
        if self.use_shortcut:
            return self.conv(x) + x
        return self.conv(x)


class MobileNet(nn.Module):
    def __init__(self, arch, num_classes=10):
        super(MobileNet, self).__init__()
        oris_ = [64, 384, 384,  384, 128, 768, 512]
        oris = [ i * 2 for i in oris_]
        assert len(arch) == len(oris), 'length wrong, len arch: {}, arch: {}'.format(len(arch), arch)
        channels = [round(float(arch[i]) * oris[i] / 4) for i in range(len(arch))]

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )

        features = []
        channels1 = [channels[0], channels[1], channels[0]]
        features.append(InvertedResidual_easy(channels1))

        channels2 = [channels[0], channels[2], channels[0]]
        features.append(InvertedResidual_easy(channels2))

        channels3 = [channels[0], channels[3], channels[4]]
        features.append(InvertedResidual_easy(channels3, stride=2))

        channels4 = [channels[4], channels[5], channels[4]]
        features.append(InvertedResidual_easy(channels4))

        self.features = nn.Sequential(*features)

        self.out = nn.Sequential(
            nn.Conv2d(channels[4], channels[6], 1, stride = 1, padding=0, bias=False),
            nn.BatchNorm2d(channels[6]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(channels[6], num_classes)


    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.out(x)
        out = self.classifier(x.view(x.size(0), -1))
        return out

class ResNet(nn.Module):
    def __init__(self, arch, num_classes=10):
        super(ResNet, self).__init__()
        oris_ = [128, 128, 128,  256, 256, 256, 256]
        oris = [ i * 2 for i in oris_]
        assert len(arch) == len(oris), 'length wrong, len arch: {}, arch: {}'.format(len(arch), arch)
        channels = [round(float(arch[i]) * oris[i] / 4) for i in range(len(arch))]

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[0], 3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[2], 3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[2], channels[0], 3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(channels[0], channels[3], 3, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[3], channels[4], 3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(channels[4]),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(channels[4], channels[5], 3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(channels[5]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[5], channels[4], 3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(channels[4]),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(channels[4], channels[6], 3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(channels[6]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[6], channels[4], 3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(channels[4]),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(channels[4], num_classes)


    def forward(self, x):
        x = self.stem(x)
        x = self.relu(x + self.block1(x))
        x = self.relu(x +self.block2(x))
        x = self.relu(self.block3(x))
        x = self.relu(x +self.block4(x))
        x = self.relu(x +self.block5(x))
        x = self.pool(x)
        out = self.classifier(x.view(x.size(0), -1))
        return out










class Network(nn.Module):
    def __init__(self, arch, num_classes=10, stages=[2, 3, 3], init_channels=32):
        super(Network, self).__init__()
        assert len(arch) == sum(stages)

        self.stem = nn.Sequential(
            nn.Conv2d(3, init_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        arch_ = arch.copy()
        features = []
        channels = init_channels
        for stage in stages:
            for idx in range(stage):
                op_func = OPS[candidate_OP[arch_.pop(0)]]
                if idx == 0:
                    # stride = 2 
                    features.append(op_func(channels, channels*2, 2))
                    channels *= 2
                else:
                    features.append(op_func(channels, channels, 1))
        self.features = nn.Sequential(*features)
        self.out = nn.Sequential(
            nn.Conv2d(channels, 1280, kernel_size=1, bias=False, stride=1),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(1280, num_classes)


    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.out(x)
        out = self.classifier(x.view(x.size(0), -1))
        return out