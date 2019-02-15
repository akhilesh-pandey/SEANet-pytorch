import math

import torch.nn as nn
from torchvision.models import ResNet
from se_module import SELayer, Aggregate


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)



class CifarSEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        # sice we are reducing the number of channels by k=4 in SE_module, we need to reshape the residual to output_channels/k
        #reduce_planes= planes//4
        #self.downsample1 = nn.Sequential(nn.Conv2d(planes, reduce_planes, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(reduce_planes))
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        #print("\n_____CifarSEBasicBlock__________\n")
        #print("x size: ",x.size())
        residual = self.downsample(x)
        #print("residual_size: ", residual.size())
        #print(self.downsample1)
        #residual=  self.downsample1(residual)
        #print("residual1_size: ", residual.size())
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print("x.shape After conv1: ",out.size())

        out = self.conv2(out)
        out = self.bn2(out)
        #print("x.shape After conv2: ",out.size())
        out = self.se(out)
        #print("x.shape After se: ",out.size())
        out += residual
        out = self.relu(out)
        #print("----------\n")

        return out


class CifarSEResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEResNet, self).__init__()
        self.inplane = 16
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 128, blocks=n_size, stride=1, reduction=reduction)
        self.agg1= Aggregate(4) 
        self.inplane= 32
        self.layer2 = self._make_layer(block, 256, blocks=n_size, stride=2, reduction=reduction)
        self.agg2= Aggregate(4) 
        self.inplane= 64
        self.layer3 = self._make_layer(block, 512, blocks=n_size, stride=2, reduction=reduction)
        self.agg3= Aggregate(4) 
        self.inplane= 128
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.inplane, num_classes)#number of channels reduces by factor of 4
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            #print("inplane: {} planes: {}".format(self.inplane, planes))
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        #print("\n_______in resnet block_______\n")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print("x.shape After conv1: ",x.size())
        x = self.layer1(x)
        #print("\nx.shape After layer1: ",x.size())
        x= self.agg1.aggregate(x)
        #print("\nx.shape After agg1: ",x.size())

        x = self.layer2(x)
        #print("\nx.shape After layer2: ",x.size())
        x= self.agg2.aggregate(x)
        #print("\nx.shape After agg2: ",x.size())

        x = self.layer3(x)
        #print("\nx.shape After layer3: ",x.size())
        x= self.agg3.aggregate(x)
        #print("\nx.shape After agg3: ",x.size())


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #print("\nx.shape After fc: ",x.size())
        return x


class CifarSEPreActResNet(CifarSEResNet):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEPreActResNet, self).__init__(block, n_size, num_classes, reduction)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


def se_resnet20(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = CifarSEResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


def se_resnet32(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarSEResNet(CifarSEBasicBlock, 5, **kwargs)
    return model


def se_resnet56(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarSEResNet(CifarSEBasicBlock, 9, **kwargs)
    return model


def se_preactresnet20(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


def se_preactresnet32(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 5, **kwargs)
    return model


def se_preactresnet56(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 9, **kwargs)
    return model
