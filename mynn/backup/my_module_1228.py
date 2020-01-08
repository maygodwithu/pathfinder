import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch._six import container_abcs
from itertools import repeat
from .my_define import _MaxPoolNd, _ConvNd, _ntuple, _pair, _BatchNorm, _AvgPoolNd


class my_Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(my_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class my_ReLU(nn.Module):
    def __init__(self, inplace=False):
        super(my_ReLU, self).__init__()
        self.inplace = inplace
        self.mode = 0  ## 0 : normal, 1 : normal-save, 2 : find-path

    def setMode(self, m):
        self.mode = m 

    def forward(self, input):
        x = F.relu(input, inplace=self.inplace)
        #print(x) 
        #print((x>0).nonzero())
        return x
        #return F.relu(input, inplace=self.inplace)

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class my_Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.mode = 0    ## 0 : normal, 1 : normal-save info  2 : find path 
        self.max_index = None
        self.verbose = False
        super(my_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def setMode(self, m):
        self.mode = m

    def _conv_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        if(self.mode == 0):
            maxconv = self._conv_forward(input, self.weight) ## max tensor
            maxconv *= 0                 ## max-convolution
            maxpos = None                ## max-index
            ## modified begin
            tweight = self.weight.clone()
            ws = self.weight.shape          # conv-shape

            if self.verbose:
                print('===weight===')
                print(ws)
                print(tweight)

            for pz in range(ws[1]):         # in channel
                tx = []
                tx.append(maxconv)          ## add max 
                for px in range(ws[2]):     # x
                    for py in range(ws[3]): # y
                        tweight *= 0
                        for ch in range(ws[0]): # out channel
                            tweight[ch,pz,px,py] = self.weight[ch,pz,px,py]                          
                        tx.append(self._conv_forward(input, tweight))
    
                ## make maximum conv
                ts = torch.stack(tx)
                maxv = torch.max(ts, axis=0)
                maxconv = maxv[0].data

                ## make maximum index
                tpos = maxv[1].data                 ## current max index
                tpos[tpos != 0] += (pz*ws[2]*ws[3]) ## add x * y for ..

                if(maxpos is None):
                    maxpos = tpos.clone() * 0  ## for the first time
                ti = torch.stack([maxpos, tpos])
                maxi = torch.max(ti, axis=0)   ## find max index
                maxpos = maxi[0].data          ## set to maxpos

                #print('===iter-conv :', pz, ' ===')
                #print(maxconv)
                #print('===iter-pos :', pz, ' ===')
                #print(maxpos)
            ## copy to class variable
            self.max_index = maxpos.data

            if self.verbose: 
                print('===maxconv===')
                print(maxconv)
                print('===maxindex==')
                print(self.max_index)
                print('===bias===')
                print(self.bias)

            return maxconv   ## return maximum conv 
            ## modified end
        elif(self.mode == 1):
            print('mode1')
            
        return self._conv_forward(input, self.weight)


class my_BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


## pool
class my_MaxPool2d(_MaxPoolNd):
    def forward(self, input):
        return F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)

class my_AvgPool2d(_AvgPoolNd):
    __constants__ = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'divisor_override']

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super(my_AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input):
        return F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)


