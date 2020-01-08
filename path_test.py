import torch
from my_vgg import my_VGG
from Pathfinder import pathfinder

if __name__ == '__main__':
    net = my_VGG('VGGT')         ## network initialize
    #net.loadWeight()            ## To load pre-trained weight
    pfind = pathfinder(net)      ## pathfinder initialize

    x = torch.randn(1,3,4,4)     ## input
    pfind.find_path(x, 2, 100)   ## input, class, top-k

