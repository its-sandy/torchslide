import math
import numpy as np
import os
import time
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path


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

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        truncated_normal_init_(self.fc1.weight, std=2.0/math.sqrt(in_dim+hidden_dim))
        truncated_normal_init_(self.fc1.bias, std=2.0/math.sqrt(in_dim+hidden_dim))
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        truncated_normal_init_(self.fc2.weight, std=2.0/math.sqrt(hidden_dim+out_dim))
        truncated_normal_init_(self.fc2.bias, std=2.0/math.sqrt(hidden_dim+out_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def run_profiling(config_dict):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.float32)

    in_dim = config_dict['in_dim']
    active_in_dim = config_dict['active_in_dim']
    hidden_dim = config_dict['hidden_dim']
    out_dim = config_dict['out_dim']
    # active_out_dim = config_dict['active_out_dim']
    batch_size = config_dict['batch_size']

    gpus = config_dict['gpus']
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    num_threads = config_dict['num_threads']
    torch.set_num_threads(num_threads)
    if gpus == '':
        device = torch.device('cpu')
        profiler_use_cuda = False
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        device = torch.device('cuda:0')
        profiler_use_cuda = True

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

            idxs_batch = np.ndarray([2, batch_size*active_in_dim], dtype='int64')
            for i in range(batch_size):
                idxs_batch[0, i*active_in_dim:(i+1)*active_in_dim] = i
                idxs_batch[1, i*active_in_dim:(i+1)*active_in_dim] = np.random.choice(in_dim, size=active_in_dim, replace=False)
            vals_batch = np.random.default_rng().standard_normal(size=batch_size*active_in_dim, dtype='float32')
            
            y = np.random.default_rng().random(size=[batch_size, out_dim], dtype='float32')
            y /= y.sum(1)[:, np.newaxis]

            x = torch.sparse_coo_tensor(idxs_batch, vals_batch,
                                        size=(batch_size, in_dim),
                                        device=device,
                                        requires_grad=False)
            optimizer.zero_grad()
            logits = net(x)
            loss = -(((torch.nn.LogSoftmax(dim=1)(logits) * torch.from_numpy(y).to(device)).sum(1)).mean())
            loss.backward()
            optimizer.step()
    end_time = time.time()
    
    with open(log_file, 'a') as out:
        print('\n--------------------------------------------', file=out, flush=True)
        print(os.path.basename(__file__), file=out, flush=True)
        print(config_dict, file=out, flush=True)
        print('device =', device, file=out, flush=True)
        if(device.type == 'cuda'):
            print('gpu =', torch.cuda.get_device_name(0), file=out, flush=True)
        print(file=out, flush=True)
        print('total train time =', end_time-begin_time, file=out, flush=True)

        if(device.type == 'cuda'):
            print('sorting by cuda_time_total', file=out, flush=True)
            print(prof.key_averages(group_by_stack_n=8).table(sort_by='cuda_time_total', top_level_events_only=True, row_limit=200), file=out, flush=True)
            print(file=out, flush=True)
            print(prof.key_averages(group_by_stack_n=8).table(sort_by='cuda_time_total', row_limit=200), file=out, flush=True)
            print(file=out, flush=True)
        else:
            print('sorting by cpu_time_total', file=out, flush=True)
            print(prof.key_averages(group_by_stack_n=8).table(sort_by='cpu_time_total', top_level_events_only=True, row_limit=200), file=out, flush=True)
            print(file=out, flush=True)
            print(prof.key_averages(group_by_stack_n=8).table(sort_by='cpu_time_total', row_limit=200), file=out, flush=True)
            print(file=out, flush=True)
    global counter
    counter += 1
    prof.export_chrome_trace(log_file + '_' + str(counter) + '.json')


if __name__ == '__main__':

    # this config is similar to amazon670k
    config_dict = {}
    config_dict['in_dim'] = 131072 # 136k in amazon
    config_dict['active_in_dim'] = 128 # 74 in amazon 
    config_dict['hidden_dim'] = 128
    config_dict['out_dim'] = 524288 # 670k in amazon
    # config_dict['active_out_dim'] = 4096
    config_dict['batch_size'] = 128

    config_dict['gpus'] = '0'
    config_dict['num_threads'] = 48

    config_dict['log_file'] = Path(os.path.basename(__file__)).stem + '_out'
    config_dict['num_iter'] = 500

    global counter
    counter = 0
    # check if both gpu and cpu profiler are working
    #################################################
    # gpu
    config_dict['gpus'] = '0'

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

    # varying out_dim
    config_dict_cur = config_dict.copy()
    config_dict_cur['out_dim'] *= 4
    run_profiling(config_dict_cur)


    # cpu
    config_dict['gpus'] = ''

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

    # varying out_dim
    config_dict_cur = config_dict.copy()
    config_dict_cur['out_dim'] *= 4
    run_profiling(config_dict_cur)
