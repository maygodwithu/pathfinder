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
        self._bverbose = False
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
            if(input.shape[0] > 1):  ## max & min input
                max_input = input[0].clone().unsqueeze(0)  ## max
                min_input = input[1].clone().unsqueeze(0)  ## min
            else:                    ## max only
                max_input = input.clone()
                min_input = input.clone() * 0

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
            tx_min = []
            ws = self.weight.shape
            bias = self.bias.clone()
            bias *= 0
            print('linear max node : ', ws[1], file=sys.stderr)
            for py in range(ws[1]):  ## out-feature
                tweight *= 0
                tweight[:,py] = self.weight[:,py].data 
                tx.append(F.linear(max_input, tweight, bias))
                tx_min.append(F.linear(min_input, tweight, bias))
                if(py % 100 == 0):
                    print('processed node : %d \r' % py, file=sys.stderr, end='')
                if self._verbose:
                    print('===iter ', py, ' ===')
                    print(tweight)
                    print(tx[py])
                    print(tx_min[py])
            ## make maximum result
            maxv = torch.max(torch.stack(tx + tx_min), axis=0)
            minv = torch.min(torch.stack(tx + tx_min), axis=0)

            self._value = maxv[0].data
            self._value_min = minv[0].data

            maxi = maxv[1].data
            maxi[ maxi >= ws[1] ] *= -1
            maxi[ maxi < 0 ] += (ws[1]-1)  ## -1 부터 시작되도록
            self._index = maxi.data

            mini = minv[1].data
            mini[ mini >= ws[1] ] *= -1
            mini[ mini < 0 ] += (ws[1]-1)
            self._index_min = mini.data

            if self._verbose:
                #print(torch.stack(tx + tx_min))
                print(self._value) 
                print(self._index) 
                print(self._value_min) 
                print(self._index_min) 

            return torch.cat([self._value, self._value_min])
            
        elif(self._mode == 1): ## normal mode
            return F.linear(input, self.weight, self.bias)

        else:
            return F.linear(input, self.weight, self.bias)
        
    def getValue(self, pos):
        if(pos >= 0):
            v = self._value.flatten()[pos] ## position  
        else:
            npos = -1 * (pos+1) ## begins from -1
            v = self._value_min.flatten()[npos]
        return v   

    def getIndex(self, pos):
        if(pos >= 0):
            tpos = self._index.flatten()[pos].item() ## 
        else: 
            npos = -1 * (pos+1) ## begins from -1
            tpos = self._index_min.flatten()[npos].item()
        return tpos

    def getOutShape(self):
        if(self._value is None): return None
        return self._value.shape

    def getWeight(self, cpos, upos):   ## cpos : current, upos : under pos
        if(cpos < 0):
            cpos = -1 * (cpos+1)
        if(upos < 0):
            upos = -1 * (upos+1)
        return self.weight[cpos, upos]

    def backward(self, input):
        ##1. use last tensor ( upper layer result)
        current_pos = int(input[-1, 0].item())               ## current position
        current_val = self.getValue(current_pos) 
        input[-1,1] = current_val  ## set current val
        ##2. make under layer information
        under_pos = self.getIndex(current_pos)
        under_out = torch.tensor([[under_pos, current_val, 0.0, 0.0]])
        #for saving weight
        weight = self.getWeight(current_pos, under_pos)
        input[-1,2] = weight.data
        out = torch.cat([input, under_out], dim=0)
        if self._bverbose:
            print('=== linear backwrd ===')
            print('selected class = ', current_pos)
            print('max value = ', current_val)
            print('position in under layer = ',  under_pos)
            print('used weigh = ', weight)
            print('-- input')
            print(input)
            print('-- output')
            print(out)
            print('======')
        
        return out

    def back_candidate(self, path, underpath, not_input):
        p = []
        cp = int(path[0].item())
        up = int(underpath[0].item())
        for px in range(self.weight.shape[1]):
            if(px == up): continue  ## check identity
            tweight = self.weight[cp, px]
            p.append(torch.tensor([px, tweight, 0.0]))      
            if(not_input):
                p.append(torch.tensor([-1*(px+1), tweight, 0.0]))      
        return p

    def path_forward(self, input_val, path):
        cpos = int(path[0].item())    # [cpos, value, weight]
        if(input_val is None):
            return self.getValue(cpos)
        
        cweight = path[2]
        return input_val * cweight 

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
        self._value_min = None
        self.relu_index = None ## save relu index
        self._verbose = False  ## for debugging
        self._bverbose = False ## for backward

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

            if(input.shape[0] > 1):  ## max & min input
                max_input = input[0].clone().unsqueeze(0)  ## max
                min_input = input[1].clone().unsqueeze(0)  ## min
            else:                    ## max only
                max_input = input.clone()
                min_input = input.clone() * 0

            x = max_input * self.relu_index ## make output
            x_min = min_input * self.relu_index
            self._value = x.data
            self._value_min = x_min.data

            if self._verbose:
                print('== relu mode2 ==')
                print(input)
                print('out shape = ', x.shape)
                print('out min shape = ', x_min.shape)
                print(x)
                print(x_min)

            return torch.cat([x, x_min])
            #return x
        else: 
            return F.relu(input, inplace=self.inplace)

    def getIndex(self, pos):
        if(pos >= 0):
            idx = self.relu_index.flatten()[pos].item() ## pos in a channel
        else:
            npos = -1 * (pos +1)
            idx = self.relu_index.flatten()[npos].item()

        return idx

    def getValue(self, pos):
        if(pos >= 0):  ## from max
            v = self._value.flatten()[pos] ## 
        else:
            npos = -1 * (pos + 1)
            v = self._value_min.flatten()[npos]

        return v

    def getOutShape(self):
        if(self._value is None): return None
        return self._value.shape

    def backward(self, input):
        current_pos = int(input[-1, 0].item())               ## current position
        current_val = input[-1,1].item()
        under_pos = current_pos
        under_out = torch.tensor([[under_pos, current_val, 0, 0]])
        out = torch.cat([input, under_out], dim=0)

        if(self.getIndex(current_pos) != 1):
            print('!!ERROR : ReLU is not 1', file=sys.stderr)
            #raise ValueError('relu_index is not 1')

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

    def path_forward(self, input_val, path):
        cpos = int(path[0].item())    # [cpos, value, weight]
        if(input_val is None):
            return self.getValue(cpos) 

        return input_val

    def back_candidate(self, path, underpath, not_input):
        return None        # no candidate


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
        self._index_min = None
        self._value_min = None
        self._verbose = False
        self._bverbose = False
        self._cverbose = False
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
        if(self._mode == 2): ## find max value & position
            if(input.shape[0] > 1):  ## max & min input
                max_input = input[0].clone().unsqueeze(0)   ## max
                min_input = input[1].clone().unsqueeze(0)  ## min
            else:                    ## max only
                max_input = input.clone()
                min_input = input.clone() * 0

            max_maxconv = self._my_conv_forward(max_input, self.weight) ## max tensor
            max_maxconv *= 0                 ## max-max convolution
            max_minconv = max_maxconv.clone()    ## max-min convolution
            min_maxconv = max_maxconv.clone()    ## min-max convolution
            min_minconv = max_maxconv.clone()    ## min-min convolution

            max_maxpos = None                ## max-max index
            max_minpos = None                ## min-min index
            min_maxpos = None                ## max-max index
            min_minpos = None                ## min-min index
            ## modified begin
            tweight = self.weight.clone()
            ws = self.weight.shape          # conv-shape

            print('conv : ', ws, file=sys.stderr)
            if self._verbose:
                print('=== conv mode2 ===')
                print('=== input ===')
                print(max_input)
                print(min_input)
                print('=== weight ===')
                print(ws)
                print(tweight)

            for pz in range(ws[1]):         # in channel
                tx = []
                tx_min = []
                tx.append(None)
                tx_min.append(None)
                for px in range(ws[2]):     # x
                    for py in range(ws[3]): # y
                        tweight *= 0
                        tweight[:,pz,px,py] = self.weight[:,pz,px,py]
                        #for ch in range(ws[0]): # out channel 
                        #    tweight[ch,pz,px,py] = self.weight[ch,pz,px,py]                          
                        tx.append(self._my_conv_forward(max_input, tweight))
                        tx_min.append(self._my_conv_forward(min_input, tweight))
    
                ## make maximum conv, minimum conv
                tx[0] = max_maxconv
                max_maxv = torch.max(torch.stack(tx), axis=0)        ## max of 'max input'
                tx[0] = max_minconv
                max_minv = torch.min(torch.stack(tx), axis=0)    ## min of 'max input'
                tx_min[0] = min_maxconv
                min_maxv = torch.max(torch.stack(tx_min), axis=0)    ## max of 'min input'
                tx_min[0] = min_minconv
                min_minv = torch.min(torch.stack(tx_min), axis=0)    ## min of 'min input'

                max_maxconv = max_maxv[0].data
                max_minconv = max_minv[0].data
                min_maxconv = min_maxv[0].data
                min_minconv = min_minv[0].data

                tpos_xx = max_maxv[1].data                 ## current max index
                tpos_xm = max_minv[1].data
                tpos_mx = min_maxv[1].data
                tpos_mm = min_minv[1].data

                ## add channel unit
                tpos_xx[tpos_xx != 0] += (pz*ws[2]*ws[3]) ## add x * y for ..
                tpos_xm[tpos_xm != 0] += (pz*ws[2]*ws[3]) ## add x * y for .. 
                tpos_mx[tpos_mx != 0] += (pz*ws[2]*ws[3]) ## add x * y for .. 
                tpos_mm[tpos_mm != 0] += (pz*ws[2]*ws[3]) ## add x * y for .. 

                if(max_maxpos is None): max_maxpos = tpos_xx.clone() * 0  ## for the first time
                if(max_minpos is None): max_minpos = tpos_xm.clone() * 0  ## for the first time
                if(min_maxpos is None): min_maxpos = tpos_mx.clone() * 0  ## for the first time
                if(min_minpos is None): min_minpos = tpos_mm.clone() * 0  ## for the first time

                max_maxpos = torch.max(torch.stack([max_maxpos, tpos_xx]), axis=0)[0].data ## max-max
                max_minpos = torch.max(torch.stack([max_minpos, tpos_xm]), axis=0)[0].data ## max-mix
                min_maxpos = torch.max(torch.stack([min_maxpos, tpos_mx]), axis=0)[0].data ## min-max
                min_minpos = torch.max(torch.stack([min_minpos, tpos_mm]), axis=0)[0].data ## min-min
                 
                if self._verbose:
                    print('==weight==')
                    print(tweight)
                    print('==conv==')
                    print('maxmax', max_maxconv)
                    print('minmax', min_maxconv)
                    print('maxmin', max_minconv)
                    print('minmin', min_minconv)
                    print('==index==')
                    print(max_maxpos)
                    print(min_maxpos)
                    print(max_minpos)
                    print(min_minpos)

            ## make real max and min
            tmax = torch.max(torch.stack([max_maxconv, min_maxconv]), axis=0)
            tmin = torch.min(torch.stack([max_minconv, min_minconv]), axis=0)             

            tmax_i = tmax[1].data   ## max index 0 or 1
            tmin_i = tmin[1].data   ## min index O or 1
            
            ## copy results to class variable. 
            self._value = tmax[0].data
            self._index = max_maxpos-(max_maxpos * tmax_i) + (-1 * min_maxpos * tmax_i)
            self._value_min = tmin[0].data
            self._index_min = max_minpos-(max_minpos * tmin_i) + (-1 * min_minpos * tmin_i)

            if self._verbose: 
                print('===max-min res===')
                print(tmax_i)
                print(tmin_i)
                print('===maxconv===')
                print(self._value)
                print(self._value_min)
                print('===maxindex==')
                print(self._index)
                print(self._index_min)
                print('===bias===')
                print(self.bias)

            #return self._value   ## return maximum conv 
            return torch.cat([self._value, self._value_min])
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
        if(pos >= 0):
            v = self._value.flatten()[pos] ## 
        else:
            npos = -1 * (pos+1)
            v = self._value_min.flatten()[npos]
        return v

    def getOutShape(self):
        if(self._value is None): return None
        return self._value.shape

    def getOutchannel(self, pos):
        if(pos >= 0):
            npos = pos
        else:
            npos = -1 * (pos+1)
        #1. find out channel 
        ws = self._value.shape
        out_channel = npos // (ws[2] * ws[3])
        return out_channel
 
    def getWeight_index(self, pos):
        if(pos >= 0):
            npos = pos
        else:
            npos = -1 * (pos+1)
        #1. find out channel 
        ws = self._value.shape
        out_channel = npos // (ws[2] * ws[3])
        #2. find weight position and set 1.0 
        if(pos >= 0):
            ridx = self._index.flatten()[npos].item() ## relitive weight position
        else:
            ridx = self._index_min.flatten()[npos].item() ## relitive weight position

        if(ridx >= 0):
            rpos = ridx -1
        else:
            rpos = -1*ridx -1
        
        return out_channel, rpos, ridx    
        #return self.weight[out_channel].flatten()[rpos]

    def getWeight(self, pos):
        out_channel, rpos, ridx = self.getWeight_index(pos)
        weight = self.weight[out_channel].flatten()[rpos]
        return weight, rpos
 
    def getIndex(self, pos):
        if(pos >= 0):
            npos = pos
        else:
            npos = -1 * (pos+1)
        #1. find out channel 
        ws = self._value.shape
        out_channel = npos // (ws[2] * ws[3])
        #2. find weight position and set 1.0 
        tweight = self.weight.clone()
        tweight *= 0
        ts = tweight.shape
        if(pos >= 0):
            ridx = self._index.flatten()[npos].item() ## relitive weight position
        else:
            ridx = self._index_min.flatten()[npos].item() ## relitive weight position

        if(ridx >= 0):
            rpos = ridx -1
        else:
            rpos = -1*ridx -1
            
        #apos = out_channel * (ts[1]*ts[2]*ts[3]) + rpos ## absolute weight position
        #tweight.flatten()[apos] = 1.0 ## set 1 others 0
        tweight[out_channel].flatten()[rpos] = 1.0
        #3. make fake input for finding position 
        fake_input = torch.zeros(self.input_shape)
        in_channel = rpos // (ts[2] * ts[3])  ## input channel
        n=in_channel * (self.input_shape[2] * self.input_shape[3])
        bn = n
        for px in range(self.input_shape[2]):
            for py in range(self.input_shape[3]):
                fake_input[0,in_channel,px,py] = n
                n += 1
        #4. do conv and find position
        fake_conv = self._my_conv_forward(fake_input, tweight)
        tpos = fake_conv.flatten()[npos].item()
        if(bn != 0 and tpos == 0):
            print("!!ERROR : fake conv fail", file=sys.stderr) 

        if(ridx >= 0):
            idx = tpos  
        else:
            idx = -1 * (tpos+1) 

        if(self._bverbose and self._verbose):
            print('== convindex ==')
            print('pos = ', pos)
            print('npos = ', npos)
            print('relitive idx = ', ridx)
            print('relitive pos = ', rpos)
            #print('absolute pos = ', apos)
            print('conv shape =', self._value.shape)
            print('out_channel =', out_channel)
            print('weight shape =', tweight.shape)
            print('input shape = ', self.input_shape)
            print('in_channel =', in_channel)
            print('begin pos =', bn )
            print('end pos =', n )
            print('weight', tweight)
            print('fake input', fake_input)
            print('fake conv', fake_conv)
            print('pos value =', tpos) 
            print('real idx =', idx) 

        return idx

    def backward(self, input):
        current_pos = int(input[-1, 0].item())               ## current position
        current_val = self.getValue(current_pos)
        ##1. set value
        input[-1,1] = current_val
        under_pos = self.getIndex(current_pos)
        ##2. set weight
        weight, rpos = self.getWeight(current_pos)
        input[-1,2] = weight.data
        input[-1,3] = rpos
        ##3. make under layer
        under_out = torch.tensor([[under_pos, current_val, 0.0, 0.0]])
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

    def get_input_position(self, pos):
        c_size = self.input_shape[2]*self.input_shape[3]
        tz = pos // c_size
        residual = pos % c_size
        tx = residual // self.input_shape[2]
        ty = residual % self.input_shape[2]
        return tz, tx, ty

    def modify_position(self, org, mv_x, mv_y, diff):
        new_val = org + diff
        z, x, y = self.get_input_position(org)  
        new_x = x + mv_x
        new_y = y + mv_y
        if(new_x < 0 or new_x >= self.input_shape[2]):
            new_val = -1
        if(new_y < 0 or new_y >= self.input_shape[3]):
            new_val = -1
        return new_val
        
    def back_candidate(self, path, underpath, not_input):
        ##1. weight index
        current_pos = int(path[0].item())
        #out_channel, rpos, ridx = self.getWeight_index(current_pos)
        out_channel = self.getOutchannel(current_pos)
        rpos = int(path[3].item())
        ##2. under pos
        under_pos = int(underpath[0].item())
        if(under_pos < 0):
            under_pos = -1 * (under_pos+1)

        ##3. make initial pair (weight, underpos)
        p = []       ## candidate
        ws = self.weight.shape
        c_size = self.input_shape[2]*self.input_shape[3]
        u_jump = (self.input_shape[2] - ws[2])  ## input x size
        if(u_jump < 0):
            print("!!ERROR : input is smaller than conv-weight")

        for wz in range(ws[1]):  # weight-z 
           un = c_size * wz
           for wx in range(ws[2]):  # weight-x
               for wy in range(ws[3]):  # weight-y
                   p.append(un)
                   un += 1 
               un = un + u_jump

        ## 4. modify position
        under_z, under_x, under_y = self.get_input_position(under_pos)
        zero_z, zero_x, zero_y = self.get_input_position(p[rpos])

        mv_x = under_x - zero_x   ## mv x 
        mv_y = under_y - zero_y   ## mv y

        if self._cverbose:
            print('zero pos = ', p[rpos])
            print('under pos = ', under_pos)
            print('zero_z  = ', zero_z, 'zero_x  = ', zero_x, 'zero_y  = ', zero_y)
            print('under_z = ', under_z, 'under_x = ', under_x, 'under_y = ', under_y)
            print('weight-underp pair =', p)

        if(under_z != zero_z):
            print("!!ERROR : make conv candidate error.") 

        diff = under_pos - p[rpos] # 
        for wn in range(len(p)):
            p[wn] = self.modify_position(p[wn], mv_x, mv_y, diff)

        if self._cverbose:
            print('weight shape = ', ws)
            print('input shape = ', self.input_shape)
            print('new weight-underp pair =', p)
            print('rpos = ', rpos)
            print('under_pos =', under_pos)
            print('pair pos = ', p[rpos])
            print('diff = ', diff)

        ## 5. make candidate
        candi = []
        for wn in range(len(p)):
            t_weight = self.weight[out_channel].flatten()[wn]
            t_pos = p[wn]
            if(t_pos < 0): continue  ## out of range
            candi.append(torch.tensor([t_pos, t_weight, wn]))      
            if(not_input):  ## input has no second source ( minimum)
                candi.append(torch.tensor([-1*(t_pos+1), t_weight, wn]))      
        
        return candi         

    def path_forward(self, input_val, path):
        cpos = int(path[0].item())    # [cpos, value, weight]
        if(input_val is None):
            return self.getValue(cpos)
        
        cweight = path[2]
        return input_val * cweight 



class _BatchNorm(_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        self._mode = 0
        self._value = None
        self._value_min = None
        self._verbose = False
        self._bverbose = False
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
            if(input.shape[0] > 1):  ## max & min input
                max_input = input[0].clone().unsqueeze(0)   ## max
                min_input = input[1].clone().unsqueeze(0)  ## min
            else:                    ## max only
                max_input = input.clone()
                min_input = input.clone() * 0

            t_running_mean = self.running_mean.clone()
            t_bias = self.bias.clone()
            t_running_mean *= 0
            t_bias *= 0
            x = F.batch_norm(
                max_input, t_running_mean, self.running_var, self.weight, t_bias,
                False,
                exponential_average_factor, self.eps)

            x_min = F.batch_norm(
                min_input, t_running_mean, self.running_var, self.weight, t_bias,
                False,
                exponential_average_factor, self.eps)

            self._value = x.data
            self._value_min = x_min.data

            if self._verbose:
                print('=== BatchNorm mode 2 ===') 
                print('set mean = ', t_running_mean)
                print('set bias = ', t_bias)
                print('set weight = ', self.weight)
                print('set var = ', self.running_var)
                print(x) 
                print(x_min) 

            return torch.cat([x, x_min])

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
        elif(self._mode == 3): ## path_forward
            t_running_mean = self.running_mean.clone()
            t_bias = self.bias.clone()
            t_running_mean *= 0
            t_bias *= 0
            x = F.batch_norm(
                input, t_running_mean, self.running_var, self.weight, t_bias,
                False,
                exponential_average_factor, self.eps)

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
        if(pos >= 0): 
            v = self._value.flatten()[pos] ## 
        else:
            npos = -1 * (pos+1)
            v = self._value_min.flatten()[npos]
        return v

    def getOutShape(self):
        if(self._value is None): return None
        return self._value.shape

    def backward(self, input):
        current_pos = int(input[-1, 0].item())               ## current position
        current_val = self.getValue(current_pos)
        input[-1,1] = current_val
        under_pos = current_pos
        under_out = torch.tensor([[under_pos, current_val, 0, 0.0]])
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

    def path_forward(self, input_val, path):
        cpos = int(path[0].item())    # [cpos, value, weight]
        if(input_val is None):
            return self.getValue(cpos) 

        vs = self._value.shape
        fake_input = torch.ones(1,vs[1],1,1)
        fake_input *= input_val 

        self.setMode(3) ## path_forward mode
        tx = self.forward(fake_input)
        return tx.flatten()[0].data

    def path_forward_org2(self, input_val, path):
        cpos = int(path[0].item())    # [cpos, value, weight]
        if(input_val is None):
            return self.getValue(cpos) 

        if(cpos < 0):
            cpos = -1 * (cpos +1)

        vs = self._value.shape
        ch = cpos // (vs[2]*vs[3])

        return input_val * self.weight[ch] / self.running_var[ch]
 
    def path_forward_org(self, input_val, path):
        cpos = int(path[0].item())    # [cpos, value, weight]
        if(input_val is None):
            return self.getValue(cpos) 
        ## make fake input & set value
        fake_input = self._value.clone() * 0
        if(cpos < 0):
            cpos = -1 * (cpos +1)
        fake_input.flatten()[cpos] = input_val
        self.setMode(3) ## path_forward mode
        tx = self.forward(fake_input)
        out_val = tx.flatten()[cpos].data

        return out_val

    def back_candidate(self, path, underpath, not_input):
        return None

## pool
class my_MaxPool2d(_MaxPoolNd):
    def forward(self, input):
        if(self._mode == 2):
            #print('max pool', file=sys.stderr)
            if(input.shape[0] > 1):  ## max & min input
                max_input = input[0].clone().unsqueeze(0)  ## max
                min_input = input[1].clone().unsqueeze(0)  ## min
            else:                    ## max only
                max_input = input.clone()
                min_input = input.clone() * 0


            x = F.max_pool2d(max_input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
            merged = []
            merged_min = []
            xs = x.shape
            for tx in range(xs[1]): # select values of max indexes. 
                pooled = torch.index_select(max_input[0][tx].flatten(), 0, self._index[0][tx].flatten())
                merged.append(pooled.data)
                pooled = torch.index_select(min_input[0][tx].flatten(), 0, self._index[0][tx].flatten())
                merged_min.append(pooled.data)
            
            #st_merged = torch.stack(merged)
            out = torch.stack(merged).view(xs[0], xs[1], xs[2], xs[3])   ## reshape
            out_min = torch.stack(merged_min).view(xs[0], xs[1], xs[2], xs[3])   ## reshape
            self._value = out.data
            self._value_min = out_min.data

            if self._verbose:
                print('=== mode2 : maxpool index apply ===')
                print('=== input ===')
                print(input)
                print('=== Pool index ===')
                print(self._index)
                print('=== pool out ===')
                print(out)
                print(out_min)

            return torch.cat([out, out_min])

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
        if(pos >= 0):
            v = self._value.flatten()[pos] ## 
        else:
            npos = -1 * (pos+1)
            v = self._value_min.flatten()[npos] ## 
             
        return v

    def getIndex(self, pos):
        channel_jump = self._index.shape[2] * self._index.shape[3] ## 'th channel
        if(pos >= 0):
            tpos = self._index.flatten()[pos].item() ## pos in a channel
            idx = tpos + (pos // channel_jump) * self._channel_size ## channel * under channel_size  
        else:
            npos = -1 * (pos+1)
            tpos = self._index.flatten()[npos].item() ## pos in a channel
            idx = tpos + (npos // channel_jump) * self._channel_size ## channel * under channel_size  
            idx = -1 * (idx+1)

        return idx 

    def getOutShape(self):
        if(self._value is None): return None
        return self._value.shape

    def backward(self, input):
        current_pos = int(input[-1, 0].item())               ## current position
        current_val = self.getValue(current_pos)
        input[-1,1] = current_val
        under_pos = self.getIndex(current_pos)
        under_out = torch.tensor([[under_pos, current_val, 0, 0.0]])
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

    def path_forward(self, input_val, path):
        cpos = int(path[0].item())    # [cpos, value, weight]
        if(input_val is None):
            return self.getValue(cpos) 

        return input_val

    def back_candidate(self, path, underpath, not_input):
        return None        # no candidate


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
        self._bverbose = False
        self._index = None
        self._index_min = None
        self._value = None
        self._value_min = None
        self._channel_size = 0 ## under layer one channel size ( x * y)
        self._divisor = None

    def setMode(self, m):
        self._mode = m

    def forward(self, input):
        ## using fake MaxPool. 
        if(self._mode == 2):
            #print('avg pool', file=sys.stderr)
            if(input.shape[0] > 1):  ## max & min input
                max_input = input[0].clone().unsqueeze(0)  ## max
                min_input = input[1].clone().unsqueeze(0)  ## min
            else:                    ## max only
                max_input = input.clone()
                min_input = input.clone() * 0

            return_indices = True  ## index
            dilation = 1           ##
            ceil_mode = False      ##  
            divisor = self.kernel_size * self.kernel_size
            self._divisor = divisor  ## save divisor for path_forward

            ## max
            x = F.max_pool2d(max_input, self.kernel_size, self.stride,
                            self.padding, dilation, ceil_mode, return_indices)
            self._index = x[1].data    ## save max position
            out = x[0].data / divisor  ## make output with MAXPOOL instead of AVGPOOL
            self._value = out.clone()

            ## min
            x_min = F.max_pool2d(-1*min_input, self.kernel_size, self.stride,
                            self.padding, dilation, ceil_mode, return_indices)
            self._index_min = x_min[1].data    ## save max position
            out_min = -1 * x_min[0].data / divisor  ## make output with MAXPOOL instead of AVGPOOL
            self._value_min = out_min.clone()
            
            self._channel_size = input.shape[2]*input.shape[3] ## save one channel size
            if self._verbose:
                print('== Avg pool ==')
                print('=== input ===')
                print(input)
                print('one channel size = ', self._channel_size)
                print('=== pool ===')
                print('divisor = ', divisor)
                print(out)
                print(out_min)
                print('=== index ===')
                print(self._index)
                print(self._index_min)
               
            return torch.cat([out, out_min]) ##

        else: 
            out = F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)
            if self._verbose:
                print('== Avg pool normal ==')
                print(out)

            return out  
                  
    def getValue(self, pos):
        if(pos >= 0):  ## from max
            v = self._value.flatten()[pos] ## 
        else:
            npos = -1 * (pos + 1)
            v = self._value_min.flatten()[npos]

        return v

    def getIndex(self, pos):
        channel_jump = self._index.shape[2] * self._index.shape[3] ## 'th channel
        if(pos >= 0):
            tpos = self._index.flatten()[pos].item() ## pos in a channel
            idx = tpos + (pos // channel_jump) * self._channel_size ## channel * under channel_size
        else:
            npos = -1 * (pos + 1)
            tpos = self._index_min.flatten()[npos].item() ## pos in a channel
            idx = tpos + (npos // channel_jump) * self._channel_size ## channel * under channel_size
            idx = -1 * (idx + 1)
         
        return idx

    def getOutShape(self):
        if(self._value is None): return None
        return self._value.shape

    def backward(self, input):
        current_pos = int(input[-1, 0].item())               ## current position
        current_val = self.getValue(current_pos)
        input[-1,1] = current_val
        under_pos = self.getIndex(current_pos)
        under_out = torch.tensor([[under_pos, current_val, 0, 0.0]])
        #set divisor in weight pos
        input[-1,2] = self._divisor
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
 
    def path_forward(self, input_val, path):
        cpos = int(path[0].item())    # [cpos, value, weight]
        if(input_val is None):
            return self.getValue(cpos) 

        if(self._divisor is None):
            print('!!ERROR : divisor not set for AvgPool')

        return input_val / self._divisor

    def back_candidate(self, path, underpath, not_input):
        return None  ## no candidate 

