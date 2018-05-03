import torch
import torch.nn as nn
import math

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, attention, num_classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.attention = attention

        if attention == "ChannelAttention":
            self.attn1 = ChannelAttention(256)
            self.attn2 = ChannelAttention(512)
            self.attn3 = ChannelAttention(1024)
        elif attention == "SpatialAttention":
            self.attn1 = SpatialAttention(256)
            self.attn2 = SpatialAttention(512)
            self.attn3 = SpatialAttention(1024)
        elif attention == "JointAttention":
            self.attn1 = JointAttention(256)
            self.attn2 = JointAttention(512)
            self.attn3 = JointAttention(1024)
        elif attention != "NoAttention":
            raise Exception('Unknown attention type')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        if self.attention != "NoAttention":
            x = self.attn1(x)

        x = self.layer2(x)
        if self.attention != "NoAttention":
            x = self.attn2(x)

        x = self.layer3(x)
        if self.attention != "NoAttention":
            x = self.attn3(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class SpatialAttention(nn.Module):

    def __init__(self, inplanes):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1)  # 1x1 conv
        self.conv2 = nn.Conv2d(inplanes // 16, inplanes // 16, kernel_size=4, stride=4)  # 4x4 conv
        self.upspl = nn.Upsample(scale_factor=4)
        self.conv3 = nn.Conv2d(inplanes // 16, 1, kernel_size=1, stride=1)  # 1x1 conv

    def forward(self, x):
        xs = self.conv1(x)
        xs = self.conv2(xs)
        xs = self.upspl(xs)
        xs = self.conv3(xs)
        return x * xs

class ChannelAttention(nn.Module):

    def __init__(self, inplanes, reduction_ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Output size of 1x1xC
        self.fc = nn.Sequential(
            nn.Linear(inplanes, inplanes // reduction_ratio),
            nn.ReLU(),
            nn.Linear(inplanes // reduction_ratio, inplanes),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        xc = self.avgpool(x).view(batch_size, num_channels)
        xc = self.fc(xc).view(batch_size, num_channels, 1, 1)
        return x * xc

class JointAttention(nn.Module):

    def __init__(self, inplanes):
        super(JointAttention, self).__init__()
        # Spatial Attention
        self.conv1 = nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1)  # 1x1 conv
        self.conv2 = nn.Conv2d(inplanes // 16, inplanes // 16, kernel_size=4, stride=4)  # 4x4 conv
        self.upspl = nn.Upsample(scale_factor=4)
        self.conv3 = nn.Conv2d(inplanes // 16, 1, kernel_size=1, stride=1)  # 1x1 conv
        # Channel Attention
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Output size of 1x1xC
        self.fc = nn.Sequential(
            nn.Linear(inplanes, inplanes // 16),
            nn.ReLU(),
            nn.Linear(inplanes // 16, inplanes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Spatial Attention module
        xs = self.conv1(x)
        xs = self.conv2(xs)
        xs = self.upspl(xs)
        xs = self.conv3(xs)

        # Channel Attention module
        batch_size, num_channels, _, _ = x.size()
        xc = self.avgpool(x).view(batch_size, num_channels)
        xc = self.fc(xc).view(batch_size, num_channels, 1, 1)

        # Joint Attention
        xj = xs + xc

        return x * xj + x


def resnet50(**kwargs):
    """
    Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], 'NoAttention', **kwargs)
    return model

def resnet50_sa(**kwargs):
    """
    Constructs a ResNet-50 model with spatial attention.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], 'SpatialAttention', **kwargs)
    return model

def resnet50_ca(**kwargs):
    """
    Constructs a ResNet-50 model with channel attention.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], 'ChannelAttention', **kwargs)
    return model

def resnet50_ja(**kwargs):
    """
    Constructs a ResNet-50 model with joint attention.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], 'JointAttention', **kwargs)
    return model
