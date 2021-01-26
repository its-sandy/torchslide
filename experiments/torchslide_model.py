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
    def __init__(self, feature_dim, hidden_dim, n_classes, last_K, last_L):
        super(Net, self).__init__()

        self.slide1 = slideLayer(in_dim=feature_dim, out_dim=hidden_dim,
                                 K=0, L=0, bucket_size=128,
                                 fill_mode='FIFO', sample_mode='vanilla')
        truncated_normal_init_(self.slide1.linear.weight, std=2.0/math.sqrt(feature_dim+hidden_dim))
        truncated_normal_init_(self.slide1.linear.bias, std=2.0/math.sqrt(feature_dim+hidden_dim))
        
        self.slide2 = slideLayer(in_dim=hidden_dim, out_dim=n_classes,
                                 K=last_K, L=last_L, bucket_size=128,
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
    n_epochs = config_dict['n_epochs']
    batch_size = config_dict['batch_size']
    lr = config_dict['lr']

    # TorchSLIDE params
    n_label_samples = config_dict['n_label_samples']
    rehash_freq = config_dict['rehash_freq']
    repermute_freq = config_dict['repermute_freq']
    last_K = config_dict['last_K']
    last_L = config_dict['last_L']
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

    net = Net(feature_dim, hidden_dim, n_classes, last_K, last_L).to(device)
    #

    begin_time = time.time()
    total_time = 0
    #
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
        # print('using padded_length=presample_counts.max().item()+n_label_samples', file=out, flush=True)
        print('using padded_length=n_label_samples', file=out, flush=True)
        print(file=out, flush=True)

        MODEL_PATH = config_dict['model_save_file_prefix'] + 'torchslide.pth'
        # net.load_state_dict(torch.load(MODEL_PATH))
        # print('loaded model from', MODEL_PATH, file=out, flush=True)

        optimizer = optim.Adam(net.parameters(), lr=lr)

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
                # y_inds = get_padded_tensor(y_inds, torch.int32, padded_length=presample_counts.max().item()+n_label_samples)
                y_inds = get_padded_tensor(y_inds, torch.int32, padded_length=n_label_samples)
            
            # time each line and layer in pytorch baseline..including time for fetching input
            # first try training with dense output/label layer...once that works as expected, introduce sparsity in last layer
            # slide model (with dense out) took a more iterations to converge compared to the pytorch baseline...why was this the case?...was it just coincidence?
            # make sure pytorch and tf baselines use float32 and not float64

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

            # validate
            if i%steps_per_epoch==steps_per_epoch-1 or i%val_freq==0:
                total_time+=time.time()-begin_time

                if i%steps_per_epoch==steps_per_epoch-1 or config_dict['num_val_batches']==-1:
                    num_val_batches = n_test//batch_size  # precision on entire test data
                else:
                    num_val_batches = config_dict['num_val_batches'] # precision on first x batches

                test_data_generator = data_generator_tslide(test_files, batch_size)
                p_at_k = 0
                with torch.no_grad():
                    for l in range(num_val_batches):
                        x_inds, x_vals, labels_batch = next(test_data_generator)
                        x_inds = get_padded_tensor(x_inds, torch.int32)
                        x_vals = get_padded_tensor(x_vals, torch.get_default_dtype())

                        optimizer.zero_grad()

                        out_logits, _ = net(in_values=x_vals, active_in_indices=x_inds)
                        k=1
                        if k==1:
                            top_k_classes = torch.argmax(out_logits, dim=1)
                        else:
                            top_k_classes = torch.topk(out_logits, k, sorted=False)[1]
                        p_at_k += np.mean([len(np.intersect1d(top_k_classes[j],labels_batch[j]))/min(k,len(labels_batch[j])) for j in range(len(top_k_classes))])
                #
                print('step=',i,
                      'train_time=',total_time,
                      'num_val_batches',num_val_batches,
                      'p_at_1=',p_at_k/num_val_batches,
                      file=out, flush=True)
                #
                begin_time = time.time()
        
        print('Finished Training', file=out, flush=True)
        torch.save(net.state_dict(), MODEL_PATH)
        print('saved model to', MODEL_PATH, file=out, flush=True)


if __name__ == '__main__':
    # execute only if run as a script
    # this_config = configs['delicious200k']
    # this_config['n_label_samples'] = 2048
    # main(this_config)
    # this_config['n_label_samples'] = 1024
    # main(this_config)
    this_config = configs['amazon670k']
    this_config['n_label_samples'] = 4096
    main(this_config)
    this_config['n_label_samples'] = 6144
    main(this_config)