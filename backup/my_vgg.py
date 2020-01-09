'''VGG11/13/16/19 in Pytorch.'''
import sys
import torch
import torch.nn as nn
from mynn.my_module import my_ReLU, my_MaxPool2d, my_Conv2d, my_Linear, my_BatchNorm2d, my_AvgPool2d
from vgg import VGG

'''
Some functions of 'network' and variable are mandatory.
variable : _layers
function : fill_layers()
'''

my_cfg = {
    'VGGT': [4, 'M', 4, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class my_VGG(VGG):
    def __init__(self, vgg_name):
        super(my_VGG, self).__init__(vgg_name)
        self.features = self._make_layers(my_cfg[vgg_name])
#        self.classifier = my_Linear(25088, 4096)
        self.classifier = my_Linear(64, 20)

        ###!!! Mandatory variables
        # _layers
        self._layers = None 

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [my_MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [my_Conv2d(in_channels, x, kernel_size=3, padding=1),
                           my_BatchNorm2d(x),
                           my_ReLU(inplace=True)]
                in_channels = x
        layers += [my_AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    ###!! Mandatory functions
    # fill_layers()
    def fill_layers(self, x):
        self._layers = []
        name = self.classifier._get_name()
        shape = self.classifier.getOutShape()
        self._layers.append((name, shape, self.classifier))
        for fe in reversed(self.features): ## backward by the reversed order
            name = fe._get_name()
            shape = fe.getOutShape()
            self._layers.append((name, shape, fe))

        self._layers.append(('Input', x.shape, None))

if __name__ == '__main__':
    net = my_VGG('VGGT')
    x = torch.randn(1,3,4,4)
    y = net(x)
    print(y)
#    net = VGG('VGG16')
#    x = torch.randn(1,3,224,224)
#    find_path(net, x, 2, 100)   ## network, input, class, top-k


