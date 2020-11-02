import os
import sys
import time
import torch
HOME_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HOME_DIR + '/SLIDE') 

from pyslide import pySlideLayer
from cppslide import cppSlideLayer, cppSlideMultiply


def get_inputs(batch_size, in_dim, out_dim, active_in_dim, active_out_dim, device):
    if active_in_dim == -1:
        active_in_indices = None
        in_values = torch.rand(batch_size, in_dim, device=device, requires_grad=True)
    else:
        active_in_indices = torch.randint(0, in_dim, [batch_size, active_in_dim], device=device)
        in_values = torch.rand(batch_size, active_in_dim, device=device, requires_grad=True)

    if active_out_dim == -1:
        active_out_indices = None
        # random_weights = torch.ones(batch_size, out_dim, device=device)
        random_weights = torch.rand(batch_size, out_dim, device=device)
    else:
        active_out_indices = torch.randint(0, out_dim, [batch_size, active_out_dim], device=device)
        # random_weights = torch.ones(batch_size, active_out_dim, device=device)
        random_weights = torch.rand(batch_size, active_out_dim, device=device)
    return in_values, active_in_indices, active_out_indices, random_weights


def get_sample_deviations(batch_size, in_dim, out_dim, active_in_dim, active_out_dim):
    # print('torch.get_num_threads()', torch.get_num_threads())
    device = torch.device('cpu')
    mylayer = pySlideLayer(in_dim, out_dim).to(device)

    in_values, active_in_indices, active_out_indices, random_weights = get_inputs(
        batch_size, in_dim, out_dim, active_in_dim, active_out_dim, device)

    py_out = mylayer(in_values, active_in_indices, active_out_indices)
    # py_out.retain_grad()
    loss = (py_out * random_weights).sum()
    loss.backward()
    py_i_grad = in_values.grad.detach().clone()
    py_w_grad = mylayer.linear.weight.grad.detach().clone()
    py_b_grad = mylayer.linear.bias.grad.detach().clone()

    mylayer.zero_grad()
    in_values.grad.data.zero_()

    cpp_out = cppSlideMultiply.apply(
        in_values, mylayer.linear.weight, mylayer.linear.bias, active_in_indices, active_out_indices)
    # cpp_out.retain_grad()
    loss = (cpp_out * random_weights).sum()
    loss.backward()
    cpp_i_grad = in_values.grad.detach().clone()
    cpp_w_grad = mylayer.linear.weight.grad.detach().clone()
    cpp_b_grad = mylayer.linear.bias.grad.detach().clone()
    
    o_dev = torch.abs(py_out-cpp_out)
    i_grad_dev = torch.abs(py_i_grad-cpp_i_grad)
    w_grad_dev = torch.abs(py_w_grad-cpp_w_grad)
    b_grad_dev = torch.abs(py_b_grad-cpp_b_grad)

    return (o_dev, i_grad_dev, w_grad_dev, b_grad_dev)


def get_mean_deviations(num_iter, num_threads, batch_size, in_dim, out_dim, active_in_dim, active_out_dim):
    torch.set_num_threads(num_threads)
    devs = {x:[0]*4 for x in ['max', 'mean', 'frac_e6', 'frac_e7']}
    for i in range(num_iter):
        # ret = (o_dev, i_grad_dev, w_grad_dev, b_grad_dev)
        ret = get_sample_deviations(batch_size=batch_size,
                                    in_dim=in_dim,
                                    out_dim=out_dim,
                                    active_in_dim=active_in_dim,
                                    active_out_dim=active_out_dim)
        for j in range(4):
            devs['max'][j] += ret[j].max().item()/num_iter
            devs['mean'][j] += ret[j].mean().item()/num_iter
            devs['frac_e6'][j] += ((ret[j] > 0.000001).sum().item()/torch.numel(ret[j]))/num_iter
            devs['frac_e7'][j] += ((ret[j] > 0.0000001).sum().item()/torch.numel(ret[j]))/num_iter

    print()
    print('num_iter', num_iter, 'num_threads', num_threads, 'batch_size', batch_size, 'in_dim', in_dim, 'out_dim', out_dim, 'active_in_dim', active_in_dim, 'active_out_dim', active_out_dim)
    entries = ['o_dev', 'i_grad_dev', 'w_grad_dev', 'b_grad_dev']
    for key, val in devs.items():
        print(key)
        for j in range(4):
            print(entries[j], "{:.5e}".format(val[j]))
    
    return devs

def run_deviation_tests():
    num_iter = 10
    num_threads = 48
    batch_size = 128
    in_dim = 4096
    out_dim = 4096

    print('-'*20)
    # dido
    active_in_dim = -1
    active_out_dim = -1
    devs = get_mean_deviations(num_iter=num_iter, num_threads=num_threads, batch_size=batch_size, in_dim=in_dim, out_dim=out_dim, active_in_dim=active_in_dim, active_out_dim=active_out_dim)
    
    print('-'*20)
    # sido
    active_out_dim = -1
    for i in range(13):
        active_in_dim = 2**i
        devs = get_mean_deviations(num_iter=num_iter, num_threads=num_threads, batch_size=batch_size, in_dim=in_dim, out_dim=out_dim, active_in_dim=active_in_dim, active_out_dim=active_out_dim)

    print('-'*20)
    # diso
    active_in_dim = -1
    for i in range(13):
        active_out_dim = 2**i
        devs = get_mean_deviations(num_iter=num_iter, num_threads=num_threads, batch_size=batch_size, in_dim=in_dim, out_dim=out_dim, active_in_dim=active_in_dim, active_out_dim=active_out_dim)

    print('-'*20)
    # siso
    for i in range(13):
        active_in_dim = 2**i
        active_out_dim = 2**i
        devs = get_mean_deviations(num_iter=num_iter, num_threads=num_threads, batch_size=batch_size, in_dim=in_dim, out_dim=out_dim, active_in_dim=active_in_dim, active_out_dim=active_out_dim)

    # similar to amazon 670k dataset's
    # input layer 135909, hidden layer 128, output layer 670091 (sparsity 0.5% = 3000), batch size 128
    print('-'*20)
    in_dim = 128
    out_dim = 524288
    active_in_dim = -1
    active_out_dim = 16384
    devs = get_mean_deviations(num_iter=num_iter, num_threads=num_threads, batch_size=batch_size, in_dim=in_dim, out_dim=out_dim, active_in_dim=active_in_dim, active_out_dim=active_out_dim)
    print('-'*20)
    in_dim = 131072
    out_dim = 128
    active_in_dim = 2048
    active_out_dim = -1
    devs = get_mean_deviations(num_iter=num_iter, num_threads=num_threads, batch_size=batch_size, in_dim=in_dim, out_dim=out_dim, active_in_dim=active_in_dim, active_out_dim=active_out_dim)

    # variation in number of threads (siso)
    num_iter=1
    batch_size=128
    in_dim=1024
    out_dim=1024
    active_in_dim=256
    active_out_dim=256
    print('-'*20)
    for num_threads in [1, 8, 48]:
        devs = get_mean_deviations(num_iter=num_iter, num_threads=num_threads, batch_size=batch_size, in_dim=in_dim, out_dim=out_dim, active_in_dim=active_in_dim, active_out_dim=active_out_dim)


def get_layer(mode, in_dim, out_dim):
    if mode == 0:
        return cppSlideLayer(in_dim, out_dim).to(torch.device('cpu'))
    elif mode == 1:
        return pySlideLayer(in_dim, out_dim).to(torch.device('cpu'))
    elif mode == 2:
        return pySlideLayer(in_dim, out_dim).to(torch.device('cuda:0'))


def get_mean_latency(mode, in_dim, active_in_dim, out_dim, active_out_dim, batch_size, num_iter, do_backprop, num_threads):
    try:
        # mode 0 - cpp slide cpu
        # mode 1 - py slide cpu
        # mode 2 - py slide gpu

        torch.set_num_threads(num_threads)

        if mode == 0 or mode == 1:
            device = torch.device('cpu')
        elif mode == 2:
            device = torch.device('cuda:0')

        mylayer = get_layer(mode, in_dim, out_dim)
        total_time = 0
        for iter in range(1, num_iter+1):
            
            if do_backprop:
                mylayer.zero_grad()
            
            in_values, active_in_indices, active_out_indices, _ = get_inputs(
                batch_size, in_dim, out_dim, active_in_dim, active_out_dim, device)
            
            torch.cuda.synchronize()
            t1 = time.time()
            y = mylayer(in_values, active_in_indices, active_out_indices)
            if do_backprop:
                yy = y.sum()
                yy.backward()
            torch.cuda.synchronize()
            t2 = time.time()

            if iter>num_iter/2:
                # warm start
                total_time += t2-t1
        
        print('mode',mode,'in_dim',in_dim,'active_in_dim',active_in_dim,'out_dim',out_dim,'active_out_dim',active_out_dim,'batch_size',batch_size,'num_iter',num_iter,'do_backprop',do_backprop,'num_threads',torch.get_num_threads(),'avg_iter_time=%f' % (total_time/(num_iter/2)*1000))
    except:
        print('mode',mode,'in_dim',in_dim,'active_in_dim',active_in_dim,'out_dim',out_dim,'active_out_dim',active_out_dim,'batch_size',batch_size,'num_iter',num_iter,'do_backprop',do_backprop,'num_threads',torch.get_num_threads(),'test failed')


def run_latency_tests():
    for mode in range(3):
        for do_backprop in [False, True]:
            num_iter = 50
            num_threads = 48
            batch_size = 128
            in_dim = 4096
            out_dim = 4096
            # dido
            active_in_dim = -1
            active_out_dim = -1
            get_mean_latency(mode, in_dim, active_in_dim, out_dim, active_out_dim, batch_size, num_iter, do_backprop, num_threads)
            # sido
            active_out_dim = -1
            for i in range(13):
                active_in_dim = 2**i
                get_mean_latency(mode, in_dim, active_in_dim, out_dim, active_out_dim, batch_size, num_iter, do_backprop, num_threads)
            # diso
            active_in_dim = -1
            for i in range(13):
                active_out_dim = 2**i
                get_mean_latency(mode, in_dim, active_in_dim, out_dim, active_out_dim, batch_size, num_iter, do_backprop, num_threads)
            # siso
            for i in range(13):
                active_in_dim = 2**i
                active_out_dim = 2**i
                get_mean_latency(mode, in_dim, active_in_dim, out_dim, active_out_dim, batch_size, num_iter, do_backprop, num_threads)
            # similar to amazon 670k dataset's
            # input layer 135909, hidden layer 128, output layer 670091 (sparsity 0.5% = 3000), batch size 128
            in_dim = 128
            out_dim = 524288
            active_in_dim = -1
            active_out_dim = 16384
            get_mean_latency(mode, in_dim, active_in_dim, out_dim, active_out_dim, batch_size, num_iter, do_backprop, num_threads)
            active_out_dim = -1
            get_mean_latency(mode, in_dim, active_in_dim, out_dim, active_out_dim, batch_size, num_iter, do_backprop, num_threads)
            in_dim = 131072
            out_dim = 128
            active_in_dim = 2048
            active_out_dim = -1
            get_mean_latency(mode, in_dim, active_in_dim, out_dim, active_out_dim, batch_size, num_iter, do_backprop, num_threads)
            active_in_dim = -1
            get_mean_latency(mode, in_dim, active_in_dim, out_dim, active_out_dim, batch_size, num_iter, do_backprop, num_threads)
            # variation in number of threads (siso)
            in_dim=4096
            out_dim=4096
            active_in_dim=1024
            active_out_dim=1024
            for num_threads in [12, 24, 48]:
                get_mean_latency(mode, in_dim, active_in_dim, out_dim, active_out_dim, batch_size, num_iter, do_backprop, num_threads)

if __name__ == "__main__":

    torch.manual_seed(0)
    # torch.set_default_dtype(torch.float64)
    run_deviation_tests()
    run_latency_tests()
