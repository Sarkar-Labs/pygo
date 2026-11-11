"""ResNet-50 — He et al. 2016, "Deep Residual Learning for Image Recognition"."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """1x1 → 3x3 → 1x1 bottleneck block with 4x expansion."""
    expansion = 4

    def __init__(self, in_ch, mid_ch, stride=1, downsample=None, groups=1, base_width=64):
        super().__init__()
        width = int(mid_ch * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_ch, width, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1,
                               groups=groups, bias=False)
        self.bn2   = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, mid_ch * self.expansion, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(mid_ch * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64,
                 zero_init_residual=True, replace_stride_with_dilation=None):
        super().__init__()
        self.in_ch = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_ch, 7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(self.in_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights(zero_init_residual)

    def _make_layer(self, block, mid_ch, blocks, stride=1, dilate=False):
        downsample = None
        prev_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_ch != mid_ch * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, mid_ch * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(mid_ch * block.expansion),
            )

        layers = [block(self.in_ch, mid_ch, stride, downsample,
                        self.groups, self.base_width)]
        self.in_ch = mid_ch * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_ch, mid_ch, groups=self.groups,
                                base_width=self.base_width))
        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x, return_features=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        x = self.avgpool(c4)
        x = torch.flatten(x, 1)
        logits = self.fc(x)

        if return_features:
            return logits, {"c1": c1, "c2": c2, "c3": c3, "c4": c4}
        return logits


def resnet50(num_classes=1000, **kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def resnet101(num_classes=1000, **kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)


def resnet152(num_classes=1000, **kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)


def resnext50_32x4d(num_classes=1000, **kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
                 groups=32, width_per_group=4, **kwargs)


def wide_resnet50_2(num_classes=1000, **kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
                 width_per_group=128, **kwargs)


def load_pretrained(model: ResNet, weights_path: str) -> ResNet:
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model
