import sys
import torch
import my_vgg
import my_resnet
import my_googlenet
from Pathfinder import pathfinder
import torchvision.transforms as transforms
from PIL import Image
import time

def img_data():
    ## 
    filename = 'ILSVRC2012_val_00023642.JPEG'
    class_label = 376

    trans = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

    image = trans(Image.open('./data/' + filename))

    return image, class_label


if __name__ == '__main__':
    #net = my_vgg.vgg16(pretrained=True)
    net = my_resnet.resnet18(pretrained=True)
    #net = my_resnet.resnet34(pretrained=True)
    #net = my_googlenet.googlenet(pretrained=True)
    pfind = pathfinder(net)      ## pathfinder initialize

    ## input : ILSVRC2012_val_00023642.JPEG, 376
    image, cls = img_data()
    image = torch.unsqueeze(image, 0)     ## input

    ##
    start_time = time.time()
    xgrad = pfind.find_numpath(image, Class=cls)
    print(xgrad)
    print("---{}s seconds---".format(time.time()-start_time))
    #print(len(xgrad[xgrad<0]))
    #print(len(xgrad[xgrad>0]))
