import torch
import torch.nn as nn
from mynn.my_module import my_ReLU, my_MaxPool2d, my_Conv2d, my_Linear, my_BatchNorm2d, my_AvgPool2d
from torch.hub import load_state_dict_from_url
from torchvision.models import VGG

cfgs = {
    'VGGT': [4, 'M', 4, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class my_VGG(VGG):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(my_VGG, self).__init__(features, num_classes=num_classes, init_weights=init_weights)
        self.features = features
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool = my_AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Sequential(
            my_Linear(512 * 7 * 7, 4096),
#            my_Linear(36, 4096),
            my_ReLU(True),
            nn.Dropout(),
            my_Linear(4096, 4096),
            my_ReLU(True),
            nn.Dropout(),
            my_Linear(4096, num_classes),
        )
        self._layers = None 
        self._verbose = True
        if init_weights:
            self._initialize_weights()

    ###!! Mandatory functions
    # fill_layers()
    def fill_layers(self, x):
        self._layers = []
        #print(name, shape)
        for fe in reversed(self.classifier): ## backward by the reversed order
            name = fe._get_name()
            if('Dropout' in name): continue
            shape = fe.getOutShape()
            self._layers.append((name, shape, fe))
            #print(name, shape, fe)

        name = self.avgpool._get_name()
        shape = self.avgpool.getOutShape()
        self._layers.append((name, shape, self.avgpool))

        for fe in reversed(self.features): ## backward by the reversed order
            name = fe._get_name()
            shape = fe.getOutShape()
            self._layers.append((name, shape, fe))
        #    print(name, shape)

        self._layers.append(('Input', x.shape, None))
          

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [my_MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = my_Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, my_BatchNorm2d(v), my_ReLU(inplace=True)]
            else:
                layers += [conv2d, my_ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _my_vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = my_VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def vgg16(pretrained=False, progress=True, **kwargs):
    return _my_vgg('vgg16', 'VGG16', False, pretrained, progress, **kwargs)

def vggt(pretrained=False, progress=True, **kwargs):
    return _my_vgg('vggt', 'VGGT', False, pretrained, progress, **kwargs)
 
if __name__ == '__main__':
    net = my_vgg16(pretrained=True)
    x = torch.randn(1,2,3,3)
    net.fill_layers(x) 

    (name, value, mod) = net._layers[1]
    print(name, value, mod)
    print(mod.weight)
