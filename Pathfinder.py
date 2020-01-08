import sys
import torch
from heapq_max import *

class SP:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __lt__(self, other):
        return self.value < other.value

    def __str__(self):
        return str("{} : {}".format(self.key, self.value))

class pathfinder():
    def __init__(self, net):
        self.net = net
        self._verbose = False
        self.maxheap = []   ## max heap
        self.rehash = {}    ## for redundant check
        self.outhash = {}    ## for redundant check
        self.pcount = 0     ## path count

    def first_pass(self, x):
        #self.net.set_mode(x, 1)
        self.set_mode(x, 1)
        y = self.net(x)
        return y

    def second_pass(self, x):
        #self.net.set_mode(x, 2)
        self.set_mode(x, 2)
        y = self.net(x)
        self.net.fill_layers(x) ## 'fill_layers' should be called here.
        return y

    def get_pvalue(self, path):
        return float(path[0,1].item())  ## (position, value, addictvie info)

    def get_input_value(self, path):
        cpos = int(path[0].item())
        if(cpos < 0):
            cpos = -1 * (cpos+1)  ## input has  only one..
        return x.flatten()[cpos]

    def backward(self, x, path):  ## cls is a position of class.. 
        ## 
        path_size = path.shape[0]
        #for name, shape, mod in self.net._layers:
        for i in range(len(self.net._layers)):
            if(i+1<path_size): continue  
            (name, shape, mod) = self.net._layers[i]
            if(mod is not None): 
                path = mod.backward(path)
                if self._verbose:
                    print(name, shape, path[-1], file=sys.stderr)
            if(name == 'Input'):  ## input has no module
                path[-1,1] = self.get_input_value(path[-1])
        return path

    def set_mode(self, x, m):
        if(self.net._layers is None):
            self.net.fill_layers(x)

        for name, shape, mod in self.net._layers:
            if(mod is None): continue   ## input layer has no module.
            mod.setMode(m)

    def backtraverse(self, path, i, x, not_input):
        v_path = []
        (name, shape, mod) = self.net._layers[i]
        if(mod is not None):
            candi = mod.back_candidate(path[i], path[i+1], not_input) ## 
            if(candi is None): return None   ## ReLU....
            for tp in candi:
                t_path = path[:i+2].clone()   ## make temp path. ex)Linear-input
                ## set temp path
                #1. set weight & conv weight pos
                t_path[-2,2] = tp[1].data  # copy weight
                t_path[-2,3] = tp[2].data  # copy conv weight pos
                #2. set under-position
                t_path[-1,0] = tp[0].data
                #3. comput score & save (score, path)
                if self._verbose:
                    print('=== candidate ===')
                    print(tp)
                    print('tpath  =', t_path)
                t_score = self.path_forward(t_path, x)
                v_path.append((t_score.data, t_path))
            return v_path
        else:
            return None

    def path_forward(self, path, x):
        depth = path.shape[0]
        if self._verbose:
            print('==path forward == ')
            print('depth ', depth)
        out = None
        for i in reversed(range(depth)):
            (name, value, mod) = self.net._layers[i]
            if(name == 'Input'):
                out = self.get_input_value(path[i])
            else:
                out = mod.path_forward(out, path[i])
            # set value to path
            path[i,1] = out 
            if self._verbose:
                print('out = ', out)

        return out

    def make_key(self, path):
        key = 'p'
        for p in path:
            key += ',' + str(int(p[0].item()))
        return key

    def push_rehash(self, path):
        self.rehash[self.make_key(path)] = 1

    def push_outhash(self, path):
        self.outhash[self.make_key(path)] = 1

    def push_path(self, path):
        heappush_max(self.maxheap, (SP(path, self.get_pvalue(path))))
        self.push_rehash(path)

    def get_maxpath(self):
        sp = heappop_max(self.maxheap)
        return (sp.key, sp.value) 

    def recheck(self, path):
        return self.make_key(path) in self.rehash

    def outcheck(self, path):
        return self.make_key(path) in self.outhash

    def find_topk(self, x, topk):
        while self.maxheap:
            short_path, v = self.get_maxpath()
            path = self.backward(x, short_path)  ## make full path
            if(self.outcheck(path)): continue    ## get outpath already
            ##
            self.push_rehash(short_path)           ## insert short path 
            self.push_rehash(path)           ## insert fullpath 
            self.print_kpath(v, short_path, path)  ## print path to out file
            #print(v, self.make_key(short_path))
            #print(path)
            for i in range(path.shape[0]-1):  ## last layer is input 
                if(self.recheck(path[:i+1])): continue         
                self.push_rehash(path[:i+1])     ## subset must be in HASH
                v_paths = self.backtraverse(path, i, x, (i+2)!=path.shape[0])
                if(v_paths is None): 
                    if self._verbose:
                        print(i, '\'th path :  no path foudn', file=sys.stderr)
                    continue
                for (v, t_path) in v_paths:
                    if(self.recheck(t_path)): continue         
                    self.push_path(t_path)
                #if self._verbose:
                #print(i, '\'th path : ', len(v_paths), ' path added to heap', file=sys.stderr)
                
                self.print_status(str("{}\'th layer:{} added to heap".format(i, len(v_paths))))
            self.pcount += 1
            if(self.pcount >= topk): break
       
        self.print_status('\n')
        #while maxheap:
        #    sp = heappop_max(maxheap)
        #    print(sp.value, sp.key)
    
    def find_path(self, x, cls, topk):
        ##1. walk on normal path with saving info.
        print('>>first forward process (saving information) ', file=sys.stderr)
        y = self.first_pass(x)
    
        ##2. compute max value
        print('>>second forward process (finding max value) ', file=sys.stderr)
        y = self.second_pass(x)
    
        ##3. find max path.. 
        print('>>backword process ( making the max path)', file=sys.stderr)
        init_path = torch.zeros(1,4)  ## make initial path [[class, value, weight, weight2]]
        init_path[0,0] = cls          ## setting class
        #maxpath = self.net.backward(x, init_path)   ## parameter : path 
        maxpath = self.backward(x, init_path)   ## parameter : path 
        self.print_path(maxpath)
    
        ##4. find top-k max path
        print('>>Top-k process', file=sys.stderr)
        self.push_path(maxpath)
        self.find_topk(x, topk)

    def print_status(self, state, tend=''):
        print(str("Top {} path found : {} heap size : {} rehash size. {}                            \r".format(self.pcount, len(self.maxheap), len(self.rehash), state)), end=tend, file=sys.stderr)

    def print_path(self, path):
        print('=== max path ===')
        print('[flattened pos]\t[shape]\t[name]\t[value]')
        for i in range(len(net._layers)):
            name = self.net._layers[i][0]
            shape = self.net._layers[i][1]
            pos = path[i,0].item()
            val = path[i,1].item()
            print(pos,  '\t', shape, '\t', name, '\t', val)

    def print_kpath(self, value, short_path, path):
        self.print_status("                    ")
        #print(value, self.make_key(path), self.make_key(short_path))
        print(value, self.make_key(path))
        self.push_outhash(path)          ## insert outhash


if __name__ == '__main__':
    from my_vgg import my_VGG
    net = my_VGG('VGG16')
    x = torch.randn(1,3,224,224)
#    net = my_VGG('VGGT')
#    x = torch.randn(1,3,16,16)

    pfind = pathfinder(net)
    pfind.find_path(x, 2, 1000)   ## input, class, top-k

