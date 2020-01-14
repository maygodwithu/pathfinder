import sys
import torch
import my_vgg
from Pathfinder import pathfinder
import torchvision.transforms as transforms
from PIL import Image

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

    return image, class_label, filename


if __name__ == '__main__':
    net = my_vgg.vgg16(pretrained=True)
    pfind = pathfinder(net)      ## pathfinder initialize

    fp = open("./result/maxpath_1000.out",'w') ## path output

    ## input : ILSVRC2012_val_00023642.JPEG, 376
    #x = torch.randn(1,3,224,224)     ## input
    image, cls = img_data()
    image = torch.unsqueeze(image, 0)     ## input

    ##
    pfind.find_path(image, Class=cls, Topk=1000, File=fp, Greedy=True)   ## input, class, top-k

