import glob
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import configs
from data_utils import data_generator_train_pth, data_generator_test_pth

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
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        truncated_normal_init_(self.fc1.weight, std=2.0/math.sqrt(feature_dim+hidden_dim))
        truncated_normal_init_(self.fc1.bias, std=2.0/math.sqrt(feature_dim+hidden_dim))
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        truncated_normal_init_(self.fc2.weight, std=2.0/math.sqrt(hidden_dim+n_classes))
        truncated_normal_init_(self.fc2.bias, std=2.0/math.sqrt(hidden_dim+n_classes))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main(config_dict):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.float32)
    # torch.set_default_dtype(torch.float64)

    feature_dim = config_dict['feature_dim']
    n_classes = config_dict['n_classes']
    hidden_dim = config_dict['hidden_dim']
    n_test = config_dict['n_test']
    batch_size = config_dict['batch_size']

    #
    os.environ['CUDA_VISIBLE_DEVICES'] = config_dict['GPUs']
    torch.set_num_threads(config_dict['num_threads'])
    if config_dict['GPUs'] == '':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        device = torch.device('cuda:0')
    #
    test_files = glob.glob(config_dict['data_path_test'])
    net = Net(feature_dim, hidden_dim, n_classes).to(device)
    #

    with open(config_dict['log_file'], 'a') as out:
        print('\n--------------------------------------------', file=out, flush=True)
        print(os.path.basename(__file__), file=out, flush=True)
        print(config_dict, file=out, flush=True)
        print('test_files =', test_files, file=out, flush=True)
        print('device =', device, file=out, flush=True)
        print('torch.get_default_dtype() =', torch.get_default_dtype(), file=out, flush=True) 
        print('random seed =', seed, file=out, flush=True) 
        if(device.type == 'cuda'):
            print('gpu =', torch.cuda.get_device_name(0), file=out, flush=True)
        print(file=out, flush=True)

        MODEL_PATH = config_dict['model_save_file_prefix'] + '_pth_1_6_baseline.pth'
        net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print('loaded model from', MODEL_PATH, file=out, flush=True)

        begin_time = time.time()        
        num_val_batches = n_test//batch_size  # precision on entire test data
        test_data_generator = data_generator_test_pth(test_files, batch_size)
        p_at_k = 0
        with torch.no_grad():
            for l in range(num_val_batches):
                if l % 100 == 0:
                    print(l)
                idxs_batch, vals_batch, labels_batch = next(test_data_generator)
                x = torch.sparse_coo_tensor(idxs_batch, vals_batch,
                                            size=(batch_size, feature_dim),
                                            device=device,
                                            requires_grad=False)
                logits = net(x)
                k=1
                if k==1:
                    top_k_classes = torch.argmax(logits, dim=1).cpu()
                else:
                    top_k_classes = torch.topk(logits, k, sorted=False)[1].cpu()
                p_at_k += np.mean([len(np.intersect1d(top_k_classes[j],labels_batch[j]))/min(k,len(labels_batch[j])) for j in range(len(top_k_classes))])
        #
        print('inference_time=',time.time() - begin_time,
              'num_val_batches',num_val_batches,
              'p_at_1=',p_at_k/num_val_batches,
              file=out, flush=True)
        #

if __name__ == '__main__':
    # execute only if run as a script
    # main(configs['delicious200k'])
    main(configs['amazon670k'])