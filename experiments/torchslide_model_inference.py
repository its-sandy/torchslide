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
    def __init__(self, config_dict):
        super(Net, self).__init__()

        feature_dim = config_dict['feature_dim']
        hidden_dim = config_dict['hidden_dim']
        n_classes = config_dict['n_classes']
        last_K = config_dict['last_K']
        last_L = config_dict['last_L']
        hash_fn = config_dict['hash_fn']
        bucket_size = config_dict['bucket_size']
        fill_mode = config_dict['fill_mode']
        sample_mode = config_dict['sample_mode']
        if hash_fn == 'wta':
            perm_size = config_dict['perm_size']

        self.slide1 = slideLayer(in_dim=feature_dim, out_dim=hidden_dim)
        truncated_normal_init_(self.slide1.linear.weight, std=2.0/math.sqrt(feature_dim+hidden_dim))
        truncated_normal_init_(self.slide1.linear.bias, std=2.0/math.sqrt(feature_dim+hidden_dim))
        
        self.slide2 = slideLayer(in_dim=hidden_dim, out_dim=n_classes)
        truncated_normal_init_(self.slide2.linear.weight, std=2.0/math.sqrt(hidden_dim+n_classes))
        truncated_normal_init_(self.slide2.linear.bias, std=2.0/math.sqrt(hidden_dim+n_classes))
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


def main(config_dict):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.float32)

    feature_dim = config_dict['feature_dim']
    n_classes = config_dict['n_classes']
    hidden_dim = config_dict['hidden_dim']
    n_test = config_dict['n_test']
    batch_size = config_dict['batch_size']

    # TorchSLIDE params
    n_label_samples = config_dict['n_label_samples']
    last_K = config_dict['last_K']
    last_L = config_dict['last_L']
    #
    torch.set_num_threads(config_dict['num_threads'])
    device = torch.device('cpu')
    #
    test_files = glob.glob(config_dict['data_path_test'])
    net = Net(config_dict).to(device)
    #
    with open(config_dict['log_file'], 'a') as out:
        print('\n--------------------------------------------', file=out, flush=True)
        print(os.path.basename(__file__), file=out, flush=True)
        print(config_dict, file=out, flush=True)
        print('test_files =', test_files, file=out, flush=True)
        print('device =', device, file=out, flush=True)
        print('torch.get_default_dtype() =', torch.get_default_dtype(), file=out, flush=True) 
        print('random seed =', seed, file=out, flush=True) 
        print(file=out, flush=True)
        print('model Layers', file=out, flush=True)
        print('net.slide1:', vars(net.slide1), file=out, flush=True)
        print('net.slide2:', vars(net.slide2), file=out, flush=True)
        print('using padded_length=n_label_samples', file=out, flush=True)
        print(file=out, flush=True)

        begin_time = time.time()
        total_time = 0

        MODEL_PATH = config_dict['model_save_file_prefix'] + 'torchslide.pth'
        net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print('loaded model from', MODEL_PATH, file=out, flush=True)
        net.slide2.rehash_nodes(reset_hashes=True, reset_randperm_nodes=True)

        num_val_batches = n_test//batch_size  # precision on entire test data
            
        test_data_generator = data_generator_tslide(test_files, batch_size)
        p_at_k = 0
        with torch.no_grad():
            for l in range(num_val_batches):
                if l%100 == 0:
                    print(l)

                x_inds, x_vals, labels_batch = next(test_data_generator)
                x_inds = get_padded_tensor(x_inds, torch.int32)
                x_vals = get_padded_tensor(x_vals, torch.get_default_dtype())

                if n_label_samples == -1: # dense
                    y_inds = None
                else:
                    y_inds = torch.zeros([batch_size, n_label_samples], dtype=torch.int32)
                
                out_logits, out_inds = net(in_values=x_vals,
                                        active_in_indices=x_inds,
                                        active_label_indices=y_inds,
                                        presample_counts=None)

                k=1
                if n_label_samples == -1:
                    if k==1:
                        top_k_classes = torch.argmax(out_logits, dim=1)
                    else:
                        top_k_classes = torch.topk(out_logits, k, sorted=False)[1]
                else:
                    if k==1:
                        top_k_classes = out_inds.gather(1, torch.argmax(out_logits, dim=1).view(-1,1))
                    else:
                        top_k_classes = out_inds.gather(1, torch.topk(out_logits, k, sorted=False)[1])
                p_at_k += np.mean([len(np.intersect1d(top_k_classes[j],labels_batch[j]))/min(k,len(labels_batch[j])) for j in range(len(top_k_classes))])
            #
            print('inference_time=',time.time() - begin_time,
                    'num_val_batches',num_val_batches,
                    'p_at_1=',p_at_k/num_val_batches,
                    file=out, flush=True)
            #


if __name__ == '__main__':
    # execute only if run as a script
    this_config = configs['amazon670k']
    main(this_config)
