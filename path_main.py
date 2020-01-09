import sys
import torch
import my_vgg
from Pathfinder import pathfinder

if __name__ == '__main__':
    net = my_vgg.vgg16(pretrained=True)
    pfind = pathfinder(net)      ## pathfinder initialize

    fp = open("maxpath.out",'w') ## path output
    #fp = sys.stdout	         ##
    x = torch.randn(1,3,224,224)     ## input
    pfind.find_path(x, 2, 10, fp)   ## input, class, top-k

