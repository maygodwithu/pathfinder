import sys
import torch
import my_vgg
from Pathfinder import pathfinder
import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from PIL import Image

# Parser
parser = argparse.ArgumentParser(description='Ramdom paths visualization')
parser.add_argument('--data', default='imagenet', type=str, metavar='D',
                    help='dataset to use')
parser.add_argument('--net', default='vgg16', type=str, metavar='N',
                    help='network trained')
parser.add_argument('--image_index', default=0, type=int, metavar='II',
                    help='image index of whi to find paths')
parser.add_argument('--target', default='true_label', type=str, metavar='T',
                    help='target neuron for which to find paths')
parser.add_argument('--target_num', default=0, type=int, metavar='TN',
                    help='order of target neuron for which to find paths in the descending order')
parser.add_argument('--server', default='rtx2080', type=str, metavar='S',
                    help='name of server where we are running')
parser.add_argument('--train', default=False, type=bool, metavar='TR',
                    help='whether to use train image or not')
parser.add_argument('--epoch', default='best', type=str, metavar='E',
                    help='number of epochs of pretrained model')
parser.add_argument('--gpu_num', default=None, type=str, metavar='G',
                    help='number of GPU(s) which is/are available')
parser.add_argument('--original_image', default=False, type=bool, metavar='O',
                    help='save original image or not')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

def data(args):
    if args.server == 'rtx2080':
        data_path =  "/data/imagenet_data/"
    else:
        data_path = "/data1/imagenet/"

    valdir = os.path.join(data_path, 'val/')
        
    trans = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

    sel_table = pd.read_csv('/home/xlpczv/Pytorch/viterbi/codes/github/sel_dict.csv', sep=",")
    image_info = sel_table[sel_table['idx']==args.image_index]
    folder = str(image_info['folder'].values[0])
    filename = str(image_info['filename'].values[0])
    class_label = int(image_info['class'].values[0])
    class_name = str(image_info['classname'].values[0])
    image = trans(Image.open(valdir + '/' + folder + '/' + filename))

    cls_idx_name_dict = np.load('imagenet1000_clsidx_to_labels.npy',allow_pickle='TRUE').item()

    return image, class_label, class_name, cls_idx_name_dict

# Target
def target(image, class_label, model, args):
    if args.target == 'true_label':
        target_neuron = int(class_label)
    elif args.target == 'argmax':
        target_neuron = int(torch.argmax(torch.squeeze(model(torch.unsqueeze(image, 0)))))
    elif args.target == 'argmin':
        target_neuron = int(torch.argmin(torch.squeeze(model(torch.unsqueeze(image, 0)))))
    else:
        target_neuron = int(torch.squeeze(torch.sort(model(torch.unsqueeze(image, 0)), descending=True)[1])[args.target_num].cpu().detach().numpy())
    return target_neuron

if __name__ == '__main__':
    net = my_vgg.vgg16(pretrained=True)
    image, class_label, class_name, cls_idx_name_dict = data(args)
    target_neuron = target(image, class_label, net, args)

    fp = open("image_index" + str(args.image_index) + "_maxpath.out",'w') ## path output
    #fp = sys.stdout            ##
    x = torch.unsqueeze(image, 0)     ## input

    pfind = pathfinder(net)      ## pathfinder initialize
    pfind.find_path(x, class_label, 100, fp)   ## input, class, top-k
