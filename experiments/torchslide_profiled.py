import glob
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

from config import configs
from data_utils import data_generator_tslide, get_padded_tensor, get_y_probs_from_y_inds

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
    def __init__(self, feature_dim, hidden_dim, n_classes):
        super(Net, self).__init__()

        self.slide1 = slideLayer(in_dim=feature_dim, out_dim=hidden_dim,
                                 K=0, L=0, bucket_size=128,
                                 fill_mode='FIFO', sample_mode='vanilla')
        truncated_normal_init_(self.slide1.linear.weight, std=2.0/math.sqrt(feature_dim+hidden_dim))
        truncated_normal_init_(self.slide1.linear.bias, std=2.0/math.sqrt(feature_dim+hidden_dim))
        
        self.slide2 = slideLayer(in_dim=hidden_dim, out_dim=n_classes,
                                 K=9, L=50, bucket_size=128,
                                 fill_mode='FIFO', sample_mode='vanilla')
        truncated_normal_init_(self.slide2.linear.weight, std=2.0/math.sqrt(hidden_dim+n_classes))
        truncated_normal_init_(self.slide2.linear.bias, std=2.0/math.sqrt(hidden_dim+n_classes))

    def forward(self, in_values, active_in_indices, active_label_indices=None, presample_counts=None):
        val1, ind1 = self.slide1(in_values, active_in_indices) # ind1 is None
        val1 = F.relu(val1)

        val2, ind2 = self.slide2(in_values=val1, active_in_indices=ind1,
                                 active_out_indices=active_label_indices,
                                 presample_counts=presample_counts)
        return val2, ind2


def main(config_dict):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.float32)

    feature_dim = config_dict['feature_dim']
    n_classes = config_dict['n_classes']
    hidden_dim = config_dict['hidden_dim']
    n_train = config_dict['n_train']
    n_test = config_dict['n_test']
    n_epochs = 1
    batch_size = config_dict['batch_size']
    lr = config_dict['lr']

    # TorchSLIDE params
    n_label_samples = config_dict['n_label_samples']
    rehash_freq = config_dict['rehash_freq']
    repermute_freq = config_dict['repermute_freq']
    #
    torch.set_num_threads(config_dict['num_threads'])
    device = torch.device('cpu')
    #
    train_files = glob.glob(config_dict['data_path_train'])
    test_files = glob.glob(config_dict['data_path_test'])
    training_data_generator = data_generator_tslide(train_files, batch_size)
    steps_per_epoch = n_train//batch_size
    n_steps = n_epochs*steps_per_epoch
    val_freq = config_dict['val_freq']

    net = Net(feature_dim, hidden_dim, n_classes).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    #

    begin_time = time.time()        
    with profiler.profile(record_shapes=True) as prof:
        for i in range(n_steps):
            # train
            x_inds, x_vals, y_inds = next(training_data_generator)
            x_inds = get_padded_tensor(x_inds, torch.int32)
            x_vals = get_padded_tensor(x_vals, torch.get_default_dtype())
            if n_label_samples == -1: # dense
                y_probs = get_y_probs_from_y_inds(y_inds, n_classes)
                y_inds = presample_counts = None
            else:
                presample_counts = torch.IntTensor([len(y_ind) for y_ind in y_inds])
                y_inds = get_padded_tensor(y_inds, torch.int32, padded_length=presample_counts.max().item()+n_label_samples)
                # y_inds = get_padded_tensor(y_inds, torch.int32, padded_length=n_label_samples)
            
            optimizer.zero_grad()
            out_logits, out_inds = net(in_values=x_vals,
                                       active_in_indices=x_inds,
                                       active_label_indices=y_inds,
                                       presample_counts=presample_counts)
            if n_label_samples == -1: # dense
                loss = -(((torch.nn.LogSoftmax(dim=1)(out_logits) * y_probs).sum(1)).mean())
            else:
                loss = softmax_cross_entropy_with_logits(out_logits, presample_counts)
            loss.backward()
            optimizer.step()

            # rehash nodes
            if n_label_samples != -1 and i % repermute_freq == repermute_freq - 1:
                net.slide2.rehash_nodes(reset_hashes=True, reset_randperm_nodes=True)
            elif n_label_samples != -1 and i % rehash_freq == rehash_freq - 1:
                net.slide2.rehash_nodes(reset_hashes=True, reset_randperm_nodes=False)

    end_time = time.time()

    with open(config_dict['log_file'], 'a') as out:
        print('\n--------------------------------------------', file=out, flush=True)
        print(os.path.basename(__file__), file=out, flush=True)
        print(config_dict, file=out, flush=True)
        print('train_files =', train_files, file=out, flush=True)
        print('test_files =', test_files, file=out, flush=True)
        print('device =', device, file=out, flush=True)
        print('torch.get_default_dtype() =', torch.get_default_dtype(), file=out, flush=True) 
        print('random seed =', seed, file=out, flush=True) 
        print(file=out, flush=True)
        print('model Layers', file=out, flush=True)
        print('net.slide1:', vars(net.slide1), file=out, flush=True)
        print('net.slide2:', vars(net.slide2), file=out, flush=True)
        print('using padded_length=presample_counts.max().item()+n_label_samples', file=out, flush=True)
        # print('using padded_length=n_label_samples', file=out, flush=True)
        print(file=out, flush=True)
        print('total train time =', end_time-begin_time, file=out, flush=True)
        print(prof.key_averages(group_by_input_shape=True).table(), file=out, flush=True)
        print(file=out, flush=True)
        print(prof.key_averages(group_by_input_shape=True).table(sort_by='cpu_time_total'), file=out, flush=True)
        print(file=out, flush=True)


if __name__ == '__main__':
    # execute only if run as a script
    this_config = configs['delicious200k']
    this_config['n_label_samples'] = 2048
    main(this_config)
    # this_config['n_label_samples'] = 1024
    # main(this_config)