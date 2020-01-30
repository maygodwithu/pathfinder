import sys
import torch
from heapq_max import *
from torch.autograd import Variable

class SP:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __lt__(self, other):
        return self.value < other.value

    def get_val(self):
        return self.value

    def __str__(self):
        return str("{} : {}".format(self.key, self.value))

class pathfinder():
    def __init__(self, net):
        self.net = net
        self._verbose = False
        self.maxheap = []   ## max heap
        self.rehash = {}    ## for redundant check
        self.outhash = {}    ## for redundant check
        self.inhash = {}    ## for input-pos check
        self.pcount = 0     ## path count
        self.upcount = 0     ## path count
        self._max_hash_size = 2000000 ## max hash size
        self._max_heap_size = 100000

    def first_pass(self, x):
        self.set_mode(x, 1)
        y = self.net(x)
        return y

    def num_pass(self, x):
        self.set_mode(x, 9) 
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

    def get_input_value(self, path, x):
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
                path[-1,1] = self.get_input_value(path[-1], x)
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
                out = self.get_input_value(path[i], x)
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
        #if(len(self.rehash) > self._max_hash_size):
            #self.rehash = {}
        self.rehash[self.make_key(path)] = 1

    def push_outhash(self, path):
        self.outhash[self.make_key(path)] = 1

    def push_inhash(self, path):
        input_pos = int(path[-1, 0].item())
        self.inhash[input_pos] = 1

    def push_path(self, path):
        size = len(self.maxheap)
        value = self.get_pvalue(path)
        #if(size >= self._max_heap_size):   ## if heapsize is over max_size, path is discarded
            #if(value < self.maxheap[-1].get_val()):
                #return    

        heappush_max(self.maxheap, (SP(path, value)))
        self.push_rehash(path)

    def get_maxpath(self):
        sp = heappop_max(self.maxheap)
        return (sp.key, sp.value) 

    def recheck(self, path):
        return self.make_key(path) in self.rehash

    def outcheck(self, path):
        return self.make_key(path) in self.outhash

    def incheck(self, path):
        return int(path[-1,0].item()) in self.inhash

    def find_topk(self, x, topk, fp=sys.stdout):
        while self.maxheap:
            short_path, v = self.get_maxpath()
            path = self.backward(x, short_path)  ## make full path
            if(self.outcheck(path)): continue    ## get outpath already
            if(not self.incheck(path)):              ## input check
                    self.upcount += 1
            ##
            self.push_rehash(short_path)           ## insert short path 
            self.push_rehash(path)           ## insert fullpath 
            self.print_kpath(v, short_path, path, fp)  ## print path to out file
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
            if(self.pcount >= topk): break
       
        self.print_status('\n')
        #while maxheap:
        #    sp = heappop_max(maxheap)
        #    print(sp.value, sp.key)

    def flushHeap(self, x, fp=sys.stdout):
        print("# flush heap. size = ", len(self.maxheap))
        while self.maxheap:
            short_path, v = self.get_maxpath()
            path = self.backward(x, short_path)  ## make full path
            if(self.outcheck(path)): continue    ## get outpath already
            self.print_kpath(v, short_path, path, fp, NOPUSH=True)

    def find_greedy_topk(self, x, topk, fp=sys.stdout):
        while self.maxheap:
            short_path, v = self.get_maxpath()
            path = self.backward(x, short_path)  ## make full path
            if(self.outcheck(path)): continue    ## get outpath already
            if(self.incheck(path)):              ## input check
                self.print_kpath(v, short_path, path, fp)
                continue
            ##
            self.push_rehash(short_path)           ## insert short path 
            self.push_rehash(path)           ## insert fullpath 
            self.print_kpath(v, short_path, path, fp)  ## print path to out file
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
            self.upcount += 1
            if(self.upcount >= topk): break
       
        self.print_status('\n')
        #while maxheap:
        #    sp = heappop_max(maxheap)
        #    print(sp.value, sp.key)
    
    def find_path(self, x, Class=0, Topk=10, File=sys.stdout, Greedy=False, FlushHeap=False):
        ##1. walk on normal path with saving info.
        print('>>first forward process (saving information) ', file=sys.stderr)
        y = self.first_pass(x)
    
        ##2. compute max value
        print('>>second forward process (finding max value) ', file=sys.stderr)
        y = self.second_pass(x)
    
        ##3. find max path.. 
        print('>>backword process ( making the max path)', file=sys.stderr)
        init_path = torch.zeros(1,4)  ## make initial path [[class, value, weight, weight2]]
        init_path[0,0] = Class          ## setting class
        #maxpath = self.net.backward(x, init_path)   ## parameter : path 
        maxpath = self.backward(x, init_path)   ## parameter : path 
        self.print_path(maxpath, File)
    
        ##4. find top-k max path
        print('>>Top-k process', file=sys.stderr)
        self.push_path(maxpath)
        if Greedy:
            self.find_greedy_topk(x, Topk, File)
        else:
            self.find_topk(x, Topk, File)
        
        ##5. flush heap
        if FlushHeap:
            self.flushHeap(x, File)

    def print_status(self, state, tend=''):
        print(str("Top {} unique path found, {} path found : {} heap size : {} rehash size. {}                            \r".format(self.upcount, self.pcount, len(self.maxheap), len(self.rehash), state)), end=tend, file=sys.stderr)

    def print_path(self, path, fp=sys.stdout):
        print('#=== max path ===', file=fp)
        print('#\t[flattened pos]\t[shape]\t[name]\t[value]', file=fp)
        for i in range(len(self.net._layers)):
            name = self.net._layers[i][0]
            shape = self.net._layers[i][1]
            pos = path[i,0].item()
            val = path[i,1].item()
            print('#\t', pos,  '\t', shape, '\t', name, '\t', val, file=fp)

    def print_kpath(self, value, short_path, path, fp=sys.stderr, NOPUSH=False):
        self.pcount += 1
        self.print_status("                    ")
        print(value, self.make_key(path), file=fp)
        if NOPUSH: return
        self.push_outhash(path)          ## insert outhash
        self.push_inhash(path)

    def find_numpath(self, x, Class=0, File=sys.stdout):
        ##1. walk on normal path with saving info.
        print('>>first forward process (saving information) ', file=sys.stderr)
        with torch.no_grad():
            y = self.first_pass(x)
            print(x.shape)
            print(y.shape)
            #y[0,Class].backward()
            #print(x.grad)
    
        ##2. find num path.. 
        print('>>find numpath process', file=sys.stderr)
        x = Variable(x, requires_grad=True)
        #print(img)
        out = self.num_pass(x)
        out_c = out[0,Class]
        out_c.backward()
        #print(out)
        #x_grad = torch.squeeze(x.grad) 

        return x.grad
    
if __name__ == '__main__':
    import my_vgg 
#    net = my_vgg.vgg16(pretrained=True)
#    x = torch.randn(1,3,224,224)
    net = my_vgg.vggt()
    x = torch.randn(1,3,16,16)

    pfind = pathfinder(net)
    x_grad = pfind.find_numpath(x, 2, 100)   ## input, class, top-k
    print(x_grad)
