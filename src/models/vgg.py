"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn

__all__ = ['VGG16', 'VGG16BN', 'VGG19', 'VGG19BN', 'VGG16Drop',
           'VGG16BNDrop', 'VGG19Drop', 'VGG19BNDrop']


def make_layers(cfg, batch_norm=False, dropout_rate=0.0):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [nn.Dropout(dropout_rate), conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [nn.Dropout(dropout_rate), conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
         512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, batch_norm=False, dropout_rate=0.0):
        super(VGG, self).__init__()
        self.features = make_layers(cfg[depth], batch_norm, dropout_rate)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def __str__(self):
        return "{}".format(self.__class__.__name__)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG16(VGG):
    def __init__(self, **kwargs):
        """
        call VGG constructor and pass kwargs
        """
        super(VGG16, self).__init__(**kwargs)


class VGG16Drop(VGG):
    def __init__(self, **kwargs):
        """
        call VGG constructor and pass kwargs
        """
        super(VGG16Drop, self).__init__(dropout_rate=0.05, **kwargs)


class VGG16BN(VGG):
    def __init__(self, **kwargs):
        """
        call VGG constructor and pass kwargs
        """
        super(VGG16BN, self).__init__(batch_norm=True, **kwargs)


class VGG16BNDrop(VGG):
    def __init__(self, **kwargs):
        """
        call VGG constructor and pass kwargs
        """
        super(VGG16BNDrop, self).__init__(
            batch_norm=True, dropout_rate=0.05, **kwargs)


class VGG19(VGG):
    def __init__(self, **kwargs):
        """
        call VGG constructor and pass kwargs
        """
        super(VGG19, self).__init__(depth=19, **kwargs)


class VGG19Drop(VGG):
    def __init__(self, **kwargs):
        """
        call VGG constructor and pass kwargs
        """
        super(VGG19Drop, self).__init__(depth=19, dropout_rate=0.05, **kwargs)


class VGG19BN(VGG):
    def __init__(self, **kwargs):
        """
        call VGG constructor and pass kwargs
        """
        super(VGG19BN, self).__init__(depth=19, batch_norm=True, **kwargs)


class VGG19BNDrop(VGG):
    def __init__(self, **kwargs):
        """
        call VGG constructor and pass kwargs
        """
        super(VGG19BNDrop, self).__init__(
            depth=19, batch_norm=True, dropout_rate=0.05, **kwargs)
