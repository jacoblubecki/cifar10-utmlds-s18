import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from numpy import sqrt

from collections import OrderedDict


class BasicBlock(nn.Module):
    
    def __init__(self, in_out, stride=1, groups=2, first_layer=False):
        super().__init__()
        in_planes, out_planes = in_out
        
        groups_1 = groups if in_planes == out_planes else 1
        
        self.conv_3x3_a = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups_1, padding=1, bias=False)
        self.batch_norm_a = nn.BatchNorm2d(in_planes)
        self.conv_3x3_b = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.batch_norm_b = nn.BatchNorm2d(out_planes)
        self.shortcut = BasicBlock._shortcut
        if first_layer or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        
    def forward(self, x):
        # Not full pre-activation. Extremely unstable using full pre-activation.
        out = F.leaky_relu(self.batch_norm_a(x))
        skip = self.shortcut(out)
        out = self.conv_3x3_a(out)
        out = self.conv_3x3_b(F.leaky_relu(self.batch_norm_b(out)))
        out += skip
        return out
    
    @staticmethod
    def _shortcut(x):
        return x  # Fallback for blocks which don't downsample.


class Resnet(nn.Module):
    """Resnet Paper: https://arxiv.org/pdf/1512.03385v1.pdf"""
    
    def __init__(self, planes, blocks, poolsize, input_channels=3, groups=2, num_classes=None):
        super().__init__()
        
        cur_planes = planes[0]
        layers = OrderedDict([
            ('conv_in', nn.Conv2d(input_channels, cur_planes, kernel_size=3, stride=1, padding=1, bias=False)),
        ])
        
        for i, (nplanes, nblocks) in enumerate(zip(planes, blocks)):
            for j, s in enumerate([1 + int(i != 0)] + [1] * (nblocks - 1)):  # Strides
                layers['block_{}_{}'.format(i+1, j+1)] = BasicBlock((cur_planes, nplanes), stride=s, groups=groups, first_layer=(j == 0))
                cur_planes = nplanes
        
        layers['pool'] = nn.AvgPool2d(poolsize)
        self.sequential = nn.Sequential(layers)
        sequential_outputs = planes[-1]
        self.split = sequential_outputs // 2
        self.bilinear = nn.Bilinear(self.split, sequential_outputs - self.split, num_classes)
        
        # Weight initialization described here: https://arxiv.org/pdf/1502.01852.pdf
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        out = F.leaky_relu(self.sequential(x))
        out = out.view(out.size(0), -1)
        return self.bilinear(out[:, :self.split], out[:, self.split:])
    
def create_model(groups=2):
    # resnet-18 with a bilinear output configuration.
    return Resnet(planes=(64, 128, 256, 512), blocks=[2, 2, 2, 2], poolsize=4, num_classes=10)

if __name__ == '__main__':
	print(BasicBlock(in_out=(3, 64)))
	print(create_model())
