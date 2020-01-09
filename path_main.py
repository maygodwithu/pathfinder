import sys
import torch
import my_vgg
from Pathfinder import pathfinder

if __name__ == '__main__':
    #net = my_vgg.vgg16(pretrained=True)
    net = my_vgg.vggt(pretrained=False)
    pfind = pathfinder(net)      ## pathfinder initialize

    fp = open("maxpath.out",'w')
    #fp = sys.stdout
    x = torch.randn(1,3,12,12)     ## input
    #x = torch.randn(1,3,224,224)     ## input
    pfind.find_path(x, 2, 10, fp)   ## input, class, top-k

