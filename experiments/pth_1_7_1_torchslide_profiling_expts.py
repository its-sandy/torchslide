import math
import numpy as np
import os
import sys
import time
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

HOME_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HOME_DIR + '/SLIDE') 
from slideLayer import slideLayer
from crossEntropyLoss import softmax_cross_entropy_with_logits

def truncated_normal_init_(tensor, mean=0, std=1):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class Net(nn.Module):
    def __init__(self, config_dict):
        super(Net, self).__init__()

        in_dim = config_dict['in_dim']
        hidden_dim = config_dict['hidden_dim']
        out_dim = config_dict['out_dim']
        last_K = config_dict['last_K']
        last_L = config_dict['last_L']
        hash_fn = config_dict['hash_fn']
        bucket_size = config_dict['bucket_size']
        fill_mode = config_dict['fill_mode']
        sample_mode = config_dict['sample_mode']
        if hash_fn == 'wta':
            perm_size = config_dict['perm_size']
        else:
            perm_size = None

        self.slide1 = slideLayer(in_dim=in_dim, out_dim=hidden_dim)
        truncated_normal_init_(self.slide1.linear.weight, std=2.0/math.sqrt(in_dim+hidden_dim))
        truncated_normal_init_(self.slide1.linear.bias, std=2.0/math.sqrt(in_dim+hidden_dim))
        
        self.slide2 = slideLayer(in_dim=hidden_dim, out_dim=out_dim)
        truncated_normal_init_(self.slide2.linear.weight, std=2.0/math.sqrt(hidden_dim+out_dim))
        truncated_normal_init_(self.slide2.linear.bias, std=2.0/math.sqrt(hidden_dim+out_dim))
        self.slide2.initialize_LSH(hash_fn=hash_fn,
                                   K=last_K, L=last_L, bucket_size=bucket_size,
                                   fill_mode=fill_mode, sample_mode=sample_mode,
                                   perm_size=perm_size)

    def forward(self, in_values, active_in_indices, active_label_indices=None, presample_counts=None):
        val1, ind1 = self.slide1(in_values, active_in_indices) # ind1 is None
        val1 = F.relu(val1)

        val2, ind2 = self.slide2(in_values=val1, active_in_indices=ind1,
                                 active_out_indices=active_label_indices,
                                 presample_counts=presample_counts)
        return val2, ind2


def run_profiling(config_dict):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.float32)

    in_dim = config_dict['in_dim']
    active_in_dim = config_dict['active_in_dim']
    hidden_dim = config_dict['hidden_dim']
    out_dim = config_dict['out_dim']
    active_out_dim = config_dict['active_out_dim']
    batch_size = config_dict['batch_size']

    last_K = config_dict['last_K']
    last_L = config_dict['last_L']
    hash_fn = config_dict['hash_fn']
    bucket_size = config_dict['bucket_size']
    fill_mode = config_dict['fill_mode']
    sample_mode = config_dict['sample_mode']
    rehash_freq = config_dict['rehash_freq']
    repermute_freq = config_dict['repermute_freq']
    perm_size = config_dict['perm_size']

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    num_threads = config_dict['num_threads']
    torch.set_num_threads(num_threads)
    device = torch.device('cpu')
    profiler_use_cuda = False

    log_file = config_dict['log_file']
    num_iter = config_dict['num_iter']

    net = Net(config_dict).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    begin_time = time.time()
    with profiler.profile(use_cuda=profiler_use_cuda, record_shapes=True, with_stack=True) as prof:
        for i in range(num_iter):

            if i%50 == 0:
                print('iter =', i)

            # numpy operations not part of profiling
            # these should be int32 for torchslide
            active_in_indices = np.stack([
                np.random.choice(in_dim, size=active_in_dim, replace=False).astype('int32')
                for i in range(batch_size)
            ])
            active_in_indices = torch.from_numpy(active_in_indices)
            in_values = np.random.default_rng().standard_normal(size=[batch_size,active_in_dim], dtype='float32')
            in_values = torch.from_numpy(in_values)
            
            if active_out_dim == -1: # dense
                active_out_indices = None
                presample_counts = None
                y_probs = np.random.default_rng().random(size=[batch_size, out_dim], dtype='float32')
                y_probs /= y_probs.sum(1)[:, np.newaxis]
                y_probs = torch.from_numpy(y_probs)
            else:
                active_out_indices = torch.empty([batch_size, active_out_dim], dtype=torch.int32, device=device)
                presample_counts = None


            optimizer.zero_grad()
            out_logits, out_inds = net(in_values=in_values,
                                        active_in_indices=active_in_indices,
                                        active_label_indices=active_out_indices,
                                        presample_counts=presample_counts)
            
            if active_out_dim == -1: # dense
                loss = -(((torch.nn.LogSoftmax(dim=1)(out_logits) * y_probs).sum(1)).mean())
            else:
                # giving equal weightage to all logits
                presample_counts = torch.IntTensor([active_out_dim for i in range(batch_size)])
                loss = softmax_cross_entropy_with_logits(out_logits, presample_counts)
            loss.backward()
            optimizer.step()

            # rehash nodes
            if active_out_dim != -1 and i % repermute_freq == repermute_freq - 1:
                net.slide2.rehash_nodes(reset_hashes=True, reset_randperm_nodes=True)
            elif active_out_dim != -1 and i % rehash_freq == rehash_freq - 1:
                net.slide2.rehash_nodes(reset_hashes=True, reset_randperm_nodes=False)
    end_time = time.time()
    
    with open(log_file, 'a') as out:
        print('\n--------------------------------------------', file=out, flush=True)
        print(os.path.basename(__file__), file=out, flush=True)
        print(config_dict, file=out, flush=True)
        print('device =', device, file=out, flush=True)
        print(file=out, flush=True)
        print('total train time =', end_time-begin_time, file=out, flush=True)

        print('sorting by cpu_time_total', file=out, flush=True)
        print(prof.key_averages(group_by_stack_n=8).table(sort_by='cpu_time_total', top_level_events_only=True, row_limit=200), file=out, flush=True)
        print(file=out, flush=True)
        print(prof.key_averages(group_by_stack_n=8).table(sort_by='cpu_time_total', row_limit=200), file=out, flush=True)
        print(file=out, flush=True)


if __name__ == '__main__':

    # this config is similar to amazon670k
    config_dict = {}
    config_dict['in_dim'] = 131072 # 136k in amazon
    config_dict['active_in_dim'] = 128 # 74 in amazon 
    config_dict['hidden_dim'] = 128
    config_dict['out_dim'] = 524288 # 670k in amazon
    config_dict['active_out_dim'] = 4096
    config_dict['batch_size'] = 128

    config_dict['last_K'] = 6
    config_dict['last_L'] = 50
    config_dict['hash_fn'] = 'wta'
    config_dict['bucket_size'] = 128
    config_dict['fill_mode'] = 'reservoir_sampling'
    config_dict['sample_mode'] = 'vanilla'
    config_dict['perm_size'] = 4
    config_dict['rehash_freq'] = 50
    config_dict['repermute_freq'] = 50

    # config_dict['gpus'] = ''
    config_dict['num_threads'] = 48

    config_dict['log_file'] = Path(os.path.basename(__file__)).stem + '_out'
    config_dict['num_iter'] = 600
    #################################################

    # base case similar to amazon
    config_dict_cur = config_dict.copy()
    run_profiling(config_dict_cur)

    # varying in_dim, active_in_dim (maintaining in sparsity)
    config_dict_cur = config_dict.copy()
    config_dict_cur['in_dim'] *= 4
    config_dict_cur['active_in_dim'] *= 4
    run_profiling(config_dict_cur)

    # varying hidden_dim
    config_dict_cur = config_dict.copy()
    config_dict_cur['hidden_dim'] *= 4
    run_profiling(config_dict_cur)

    # varying out_dim, active_out_dim (maintaining out sparsity)
    config_dict_cur = config_dict.copy()
    config_dict_cur['out_dim'] *= 4
    config_dict_cur['active_out_dim'] *= 4
    run_profiling(config_dict_cur)

    # varying active_out_dim (sparsity) 1
    config_dict_cur = config_dict.copy()
    config_dict_cur['active_out_dim'] = 1024
    run_profiling(config_dict_cur)

    # varying active_out_dim (sparsity) 2
    config_dict_cur = config_dict.copy()
    config_dict_cur['active_out_dim'] = 16384
    run_profiling(config_dict_cur)
    # For these high values of active_out_dim, most nodes come from rand_perm only.
    # But that's fine for profiling 

    # varying active_out_dim (sparsity) 3
    config_dict_cur = config_dict.copy()
    config_dict_cur['active_out_dim'] = 131072
    run_profiling(config_dict_cur)

    # varying active_out_dim (sparsity) 4
    config_dict_cur = config_dict.copy()
    config_dict_cur['active_out_dim'] = -1 # dense
    run_profiling(config_dict_cur)
    # No rehashing performed in this case

    # varying L (num_hashes) 1
    config_dict_cur = config_dict.copy()
    config_dict_cur['last_L'] = 25
    run_profiling(config_dict_cur)

    # varying L (num_hashes) 2
    config_dict_cur = config_dict.copy()
    config_dict_cur['last_L'] = 100
    run_profiling(config_dict_cur)