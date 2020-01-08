import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch._six import container_abcs
from itertools import repeat
from .my_define import _MaxPoolNd, _ConvNd, _ntuple, _pair, _NormBase, _AvgPoolNd


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
        self._mode = 0
        self._verbose = False
        self._bverbose = True
        self._value = None     ## save mav value
        self._index = None     ## save max position

    def setMode(self, m):
        self._mode = m

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        tweight = self.weight.clone()
        if(self._mode == 2):  ## find path
            maxpos = None
            if self._verbose:
                print('== input ==')
                print(input.shape)
                print(input)
                print('== weight ==')
                print(self.weight.shape)
                print(self.weight)
                print('== bias ==')
                print(self.bias)
            tx = []
            ws = self.weight.shape
            bias = self.bias.clone()
            bias *= 0
            print('linear max node : ', ws[1], file=sys.stderr)
            for py in range(ws[1]):  ## out-feature
                tweight *= 0
                tweight[:,py] = self.weight[:,py].data 
                tx.append(F.linear(input, tweight, bias))
                if(py % 1000 == 0):
                    print('processed node : %d \r' % py, file=sys.stderr, end='')
                if self._verbose:
                    print('===iter ', py, ' ===')
                    print(tweight)
                    print(tx[py])
            ## make maximum result
            ts = torch.stack(tx)
            maxv = torch.max(ts, axis=0)

            self._value = maxv[0].data
            self._index = maxv[1].data

            if self._verbose:
                print(self._value) 
                print(self._index) 

            return self._value
            
        elif(self._mode == 1): ## normal mode
            return F.linear(input, self.weight, self.bias)

        else:
            return F.linear(input, self.weight, self.bias)
        
    def getValue(self, pos):
        return self._value.flatten()[pos] ## position 

    def getIndex(self, pos):
        tpos = self._index.flatten()[pos].item() ## 
        return tpos

    def getOutShape(self):
        return self._value.shape

    def backward(self, input):
        ## use last tensor ( upper layer result)
        current_pos = int(input[-1, 0].item())               ## current position
        current_val = self.getValue(current_pos)
        under_pos = self.getIndex(current_pos)
        under_out = torch.tensor([[under_pos, current_val, 0]])
        out = torch.cat([input, under_out], dim=0)
        if self._bverbose:
            print('=== linear backwrd ===')
            print('selected class = ', current_pos)
            print('max value = ', current_val)
            print('position in under layer = ',  under_pos)
            print('-- input')
            print(input)
            print('-- output')
            print(out)
            print('======')
        
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class my_ReLU(nn.Module):
    def __init__(self, inplace=False):
        super(my_ReLU, self).__init__()
        self.inplace = inplace
        self._mode = 0  ## 0 : find-path, 1 : normal-save, 2 : normal
        self._value  = None
        self.relu_index = None ## save relu index
        self._verbose = False  ## for debugging
        self._bverbose = True ## for backward

    def setMode(self, m):
        self._mode = m 

    def forward(self, input):
        if(self._mode == 1):   # nomal run with saving info
            x = F.relu(input, inplace=self.inplace)
            self.relu_index = x.clone()
            self.relu_index[self.relu_index>0] = 1
            if self._verbose:
                print('== relu mode1 ==')
                print('org out shape = ', x.shape)
                print(x) 
                print(self.relu_index)
            return x
        elif(self._mode == 2): # find path
            if(self.relu_index is None):
                raise ValueError('relu_index is None')

            x = input * self.relu_index ## make output
            self._value = x.data

            if self._verbose:
                print('== relu mode2 ==')
                print(input)
                print('out shape = ', x.shape)
                print(x)
            return x
        else: 
            return F.relu(input, inplace=self.inplace)

    def getIndex(self, pos):
        return self.relu_index.flatten()[pos].item() ## pos in a channel

    def getOutShape(self):
        return self._value.shape

    def backward(self, input):
        current_pos = int(input[-1, 0].item())               ## current position
        current_val = input[-1,1].item()
        under_pos = current_pos
        under_out = torch.tensor([[under_pos, current_val, 0]])
        out = torch.cat([input, under_out], dim=0)

        if(self.getIndex(current_pos) != 1):
            raise ValueError('relu_index is not 1')

        if self._bverbose:
            print('=== ReLU backwrd ===')
            print('selected class = ', current_pos)
            print('relu value is 1')
            print('max value = ', current_val)
            print('position in under layer = ',  under_pos)
            print('-- input')
            print(input)
            print('-- output')
            print(out)
            print('======')
            
        return out

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
        self._mode = 0    ## 0 : normal, 1 : normal-save info  2 : find path 
        self.input_shape = None
        self._index = None
        self._value = None
        self._verbose = False
        self._bverbose = True
        super(my_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def setMode(self, m):
        self._mode = m

    def _conv_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def _my_conv_forward(self, input, weight):
        bias = self.bias.clone()
        bias *= 0    ## set bias= 0
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)


    def forward(self, input):
        if(self._mode == 2):
            maxconv = self._my_conv_forward(input, self.weight) ## max tensor
            maxconv *= 0                 ## max-convolution
            maxpos = None                ## max-index
            ## modified begin
            tweight = self.weight.clone()
            ws = self.weight.shape          # conv-shape

            print('conv : ', ws, file=sys.stderr)
            if self._verbose:
                print('=== conv mode2 ===')
                print('===weight===')
                print(ws)
                print(tweight)

            for pz in range(ws[1]):         # in channel
                tx = []
                tx.append(maxconv)          ## add max 
                for px in range(ws[2]):     # x
                    for py in range(ws[3]): # y
                        tweight *= 0
                        tweight[:,pz,px,py] = self.weight[:,pz,px,py]
                        #for ch in range(ws[0]): # out channel 
                        #    tweight[ch,pz,px,py] = self.weight[ch,pz,px,py]                          
                        tx.append(self._my_conv_forward(input, tweight))
    
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

                #del(ts)
                #del(tx)
                #del(ti)
            ## copy to class variable
            self._index = maxpos.data
            self._value = maxconv.data

            if self._verbose: 
                print('===maxconv===')
                print(maxconv)
                print('===maxindex==')
                print(self._index)
                print('===bias===')
                print(self.bias)

            return maxconv   ## return maximum conv 
            ## modified end
        elif(self._mode == 1):
            self.input_shape = input.shape
            if self._verbose:
                print('=== conv mode1 ===')
                print('save input shape = ', self.input_shape)
            return self._conv_forward(input, self.weight)
        else: 
            return self._conv_forward(input, self.weight)
 
    def getValue(self, pos):
        return self._value.flatten()[pos] ## 

    def getOutShape(self):
        return self._value.shape

    def getIndex(self, pos):
        #1. find out channel 
        ws = self._value.shape
        out_channel = pos // (ws[2] * ws[3])
        #2. find weight position and set 1.0 
        tweight = self.weight.clone()
        tweight *= 0
        ts = tweight.shape
        rpos = self._index.flatten()[pos].item() -1 ## relitive position
        apos = out_channel * (ts[1]*ts[2]*ts[3]) + rpos ## absolute position
        tweight.flatten()[apos] = 1.0 ## set 1 others 0
        #3. make fake input for finding position 
        fake_input = torch.zeros(self.input_shape)
        in_channel = rpos // (ts[2] * ts[3])  ## input channel
        n=in_channel * (self.input_shape[2] * self.input_shape[3])
        for px in range(self.input_shape[2]):
            for py in range(self.input_shape[3]):
                fake_input[0,in_channel,px,py] = n
                n += 1
        #4. do conv and find position
        fake_conv = self._my_conv_forward(fake_input, tweight)
        tpos = fake_conv.flatten()[pos].item()

        if(self._bverbose and self._berbose):
            print('== convindex ==')
            print('pos = ', pos)
            print('relitive pos = ', rpos)
            print('absolute pos = ', apos)
            print('conv shape =', self._value.shape)
            print('out_channel =', out_channel)
            print('weight shape =', tweight.shape)
            print('input shape = ', self.input_shape)
            print('in_channel =', in_channel)
            print('end pos =', n )
            print('fake input', fake_input)
            print('fake conv', fake_conv)
            print('pos value =', tpos) 

        return tpos

    def backward(self, input):
        current_pos = int(input[-1, 0].item())               ## current position
        current_val = self.getValue(current_pos)
        under_pos = self.getIndex(current_pos)
        under_out = torch.tensor([[under_pos, current_val, 0]])
        out = torch.cat([input, under_out], dim=0)
 
        if self._bverbose:
            print('=== conv backward ===')
            print('selected position = ', current_pos)
            print('max value = ', current_val)
            print('position in under layer = ',  under_pos)
            print('-- input')
            print(input)
            print('-- output')
            print(out)
            print('======')
    
        return out


class _BatchNorm(_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        self._mode = 0
        self._value = None
        self._verbose = False
        self._bverbose = True
        super(_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def setMode(self, m):
        self._mode = m

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if(self._mode == 2):   # find path
            #print('batch norm', file=sys.stderr)
            t_running_mean = self.running_mean.clone()
            t_bias = self.bias.clone()
            t_running_mean *= 0
            t_bias *= 0
            x = F.batch_norm(
                input, t_running_mean, self.running_var, self.weight, t_bias,
                False,
                exponential_average_factor, self.eps)

            self._value = x.data

            if self._verbose:
                print('=== BatchNorm mode 2 ===') 
                print('set mean = ', t_running_mean)
                print('set bias = ', t_bias)
                print('set weight = ', self.weight)
                print('set var = ', self.running_var)
                print(x) 

            return x
        elif(self._mode == 1):
            x = F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)

            if self._verbose:
                print('=== BatchNorm mode 1 ===')
                print('org mean = ', self.running_mean)
                print('org bias = ', self.bias)
                print('org weight = ', self.weight) 
                print('org var = ', self.running_var) 

            return x

        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)

class my_BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def getValue(self, pos):
        return self._value.flatten()[pos] ## 

    def getOutShape(self):
        return self._value.shape

    def backward(self, input):
        current_pos = int(input[-1, 0].item())               ## current position
        current_val = self.getValue(current_pos)
        under_pos = current_pos
        under_out = torch.tensor([[under_pos, current_val, 0]])
        out = torch.cat([input, under_out], dim=0)
        if self._bverbose:
            print('=== batchnorm backward ===')
            print('selected position = ', current_pos)
            print('value = ', current_val)
            print('position in under layer = ',  under_pos)
            print('-- input')
            print(input)
            print('-- output')
            print(out)
            print('======')
 
        return out

## pool
class my_MaxPool2d(_MaxPoolNd):
    def forward(self, input):
        if(self._mode == 2):
            #print('max pool', file=sys.stderr)
            x = F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
            merged = []
            xs = x.shape
            for tx in range(xs[1]): # select values of max indexes. 
                pooled = torch.index_select(input[0][tx].flatten(), 0, self._index[0][tx].flatten())
                merged.append(pooled.data)
            
            #st_merged = torch.stack(merged)
            out = torch.stack(merged).view(xs[0], xs[1], xs[2], xs[3])   ## reshape
            self._value = out.data

            if self._verbose:
                print('=== mode2 : maxpool index apply ===')
                print('=== input ===')
                print(input)
                print('=== Pool index ===')
                print(self._index)
                print('=== pool out ===')
                print(out)

            return out 

        elif(self._mode == 1):
            return_indices = True  ## index
            x = F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            return_indices)

            self._channel_size = input.shape[2]*input.shape[3] ## save one channel size

            if self._verbose:
                print('=== mode1 : maxpool index save ===')
                print('=== input ===')
                print('channel size = ', self._channel_size)
                print(input)
                print('=== Pool result ===')
                print(x[0])
                print('=== Pool index ===')
                print(x[1])

            self._index = x[1].data

            return x[0]
        else:  
            return F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)

    def getValue(self, pos):
        return self._value.flatten()[pos] ## 

    def getIndex(self, pos):
        tpos = self._index.flatten()[pos].item() ## pos in a channel
        channel_jump = self._index.shape[2] * self._index.shape[3] ## 'th channel
        npos = tpos + (pos // channel_jump) * self._channel_size ## channel * under channel_size
        return npos

    def getOutShape(self):
        return self._value.shape

    def backward(self, input):
        current_pos = int(input[-1, 0].item())               ## current position
        current_val = self.getValue(current_pos)
        under_pos = self.getIndex(current_pos)
        under_out = torch.tensor([[under_pos, current_val, 0]])
        out = torch.cat([input, under_out], dim=0)
 
        if self._bverbose:
            print('=== Max pool backward ===')
            print('selected position = ', current_pos)
            print('max value = ', current_val)
            print('position in under layer = ',  under_pos)
            print('-- input')
            print(input)
            print('-- output')
            print(out)
            print('======')
 
        return out


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
        self._mode = 0
        self._verbose = False
        self._bverbose = True
        self._index = None
        self._value = None
        self._channel_size = 0 ## under layer one channel size ( x * y)

    def setMode(self, m):
        self._mode = m

    def forward(self, input):
        ## using fake MaxPool. 
        if(self._mode == 2):
            #print('avg pool', file=sys.stderr)
            return_indices = True  ## index
            dilation = 1           ##
            ceil_mode = False      ##  
            divisor = self.kernel_size * self.kernel_size

            x = F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, dilation, ceil_mode, return_indices)

            self._index = x[1].data    ## save max position
            out = x[0].data / divisor  ## make output with MAXPOOL instead of AVGPOOL
            self._value = out.clone()
            
            self._channel_size = input.shape[2]*input.shape[3] ## save one channel size
            if self._verbose:
                print('== Avg pool ==')
                print('=== input ===')
                print(input)
                print('one channel size = ', self._channel_size)
                print('=== pool ===')
                print('divisor = ', divisor)
                print(out)
                print('=== index ===')
                print(self._index)
               
            return out ##

        else: 
            out = F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)
            if self._verbose:
                print('== Avg pool normal ==')
                print(out)

            return out  
                  
    def getValue(self, pos):
        return self._value.flatten()[pos] ## 

    def getIndex(self, pos):
        tpos = self._index.flatten()[pos].item() ## pos in a channel
        channel_jump = self._index.shape[2] * self._index.shape[3] ## 'th channel
        pos = tpos + (pos // channel_jump) * self._channel_size ## channel * under channel_size
        return pos

    def getOutShape(self):
        return self._value.shape

    def backward(self, input):
        current_pos = int(input[-1, 0].item())               ## current position
        current_val = self.getValue(current_pos)
        under_pos = self.getIndex(current_pos)
        under_out = torch.tensor([[under_pos, current_val, 0]])
        out = torch.cat([input, under_out], dim=0)
 
        if self._bverbose:
            print('=== Avg pool backward ===')
            print('selected position = ', current_pos)
            print('one channel size = ', self._channel_size)
            print('max value = ', current_val)
            print('position in under layer = ',  under_pos)
            print('-- input')
            print(input)
            print('-- output')
            print(out)
            print('======')
                    
        return out

