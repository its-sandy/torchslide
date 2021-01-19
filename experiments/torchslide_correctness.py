import glob
import math
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import configs
from data_utils import data_generator_tslide, get_padded_tensor, get_y_probs_from_y_inds, data_generator_train_pth

HOME_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HOME_DIR + '/SLIDE') 
from slideLayer import slideLayer


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
    feature_dim = config_dict['feature_dim']
    n_classes = config_dict['n_classes']
    hidden_dim = config_dict['hidden_dim']
    batch_size = config_dict['batch_size']

    # TorchSLIDE params
    n_label_samples = config_dict['n_label_samples']
    torch.set_num_threads(config_dict['num_threads'])
    device = torch.device('cpu')
    #

    net = Net(feature_dim, hidden_dim, n_classes).to(device)
    #
    print('device =', device, flush=True)
    print('net.slide1:', vars(net.slide1), flush=True)
    print('net.slide2:', vars(net.slide2), flush=True)
    print(flush=True)

    # optimizer = optim.Adam(net.parameters(), lr=lr)

    train_files = glob.glob(config_dict['data_path_train'])

    ################################################################
    # compute for PyTorch baseline
    training_data_generator_pth = data_generator_train_pth(train_files, batch_size, n_classes)
    idxs_pth, vals_pth, y_pth = next(training_data_generator_pth)
    x_pth = torch.sparse_coo_tensor(idxs_pth, vals_pth,
                                size=(batch_size, feature_dim),
                                dtype=torch.float32, device=device,
                                requires_grad=False)
    # optimizer.zero_grad()
    net.zero_grad()
    out1_pth = F.relu(net.slide1.linear(x_pth))
    out2_pth = net.slide2.linear(out1_pth)
    loss_pth = -(((torch.nn.LogSoftmax(dim=1)(out2_pth) * torch.from_numpy(y_pth).to(device)).sum(1)).mean())
    loss_pth.backward()
    s2_w_grad_pth = net.slide2.linear.weight.grad.detach().clone()
    s2_b_grad_pth = net.slide2.linear.bias.grad.detach().clone()
    s1_w_grad_pth = net.slide1.linear.weight.grad.detach().clone()
    s1_b_grad_pth = net.slide1.linear.bias.grad.detach().clone()

    ################################################################
    # compute for torchSLIDE
    training_data_generator_tslide = data_generator_tslide(train_files, batch_size)
    x_inds, x_vals, y_inds = next(training_data_generator_tslide)
    x_inds = get_padded_tensor(x_inds, torch.int32)
    x_vals = get_padded_tensor(x_vals, torch.float32)
    if n_label_samples == -1: # dense
        y_probs = get_y_probs_from_y_inds(y_inds, n_classes)
        y_inds = presample_counts = None
    else:
        presample_counts = torch.IntTensor([len(y_ind) for y_ind in y_inds])
        y_inds = get_padded_tensor(y_inds, torch.int32, padded_length=n_label_samples)
    
    # optimizer.zero_grad()
    net.zero_grad()
    out2_ts, out_inds = net(in_values=x_vals,
                                active_in_indices=x_inds,
                                active_label_indices=y_inds,
                                presample_counts=presample_counts)
    if n_label_samples == -1: # dense
        loss_ts = -(((torch.nn.LogSoftmax(dim=1)(out2_ts) * y_probs).sum(1)).mean())
    else:
        loss_ts = softmax_cross_entropy_with_logits(out2_ts, presample_counts)
    loss_ts.backward()
    s2_w_grad_ts = net.slide2.linear.weight.grad.detach().clone()
    s2_b_grad_ts = net.slide2.linear.bias.grad.detach().clone()
    s1_w_grad_ts = net.slide1.linear.weight.grad.detach().clone()
    s1_b_grad_ts = net.slide1.linear.bias.grad.detach().clone()

    print('torch.abs(loss_ts-loss_pth)', torch.abs(loss_ts-loss_pth))
    print('torch.abs(out2_ts-out2_pth).max().item()', torch.abs(out2_ts-out2_pth).max().item())
    print('torch.abs(s2_w_grad_ts-s2_w_grad_pth).max().item()', torch.abs(s2_w_grad_ts-s2_w_grad_pth).max().item())
    print('torch.abs(s2_b_grad_ts-s2_b_grad_pth).max().item()', torch.abs(s2_b_grad_ts-s2_b_grad_pth).max().item())
    print('torch.abs(s1_w_grad_ts-s1_w_grad_pth).max().item()', torch.abs(s1_w_grad_ts-s1_w_grad_pth).max().item())
    print('torch.abs(s1_b_grad_ts-s1_b_grad_pth).max().item()', torch.abs(s1_b_grad_ts-s1_b_grad_pth).max().item())
    

if __name__ == '__main__':
    # execute only if run as a script
    main(configs['delicious200k'])