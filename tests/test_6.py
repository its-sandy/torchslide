import torch
import torch.autograd.profiler as profiler
import time
import sys
sys.path.insert(0, '/home/santhosh/torchslide/SLIDE') 
from srpHash import srpHashTable

def get_hashes_scatter(input, plus_mask):
    # ratio is 1 by default
    with torch.no_grad():
        # [num_hashes, batch_size, in_dim]
        num_hashes = plus_mask.size(0)
        batch_size = input.size(0)
        input = torch.unsqueeze(input, 0).expand(num_hashes, -1, -1)
        plus_mask = torch.unsqueeze(plus_mask, 1).expand(-1, batch_size, -1)
        out = torch.zeros(num_hashes, batch_size, 2)
        out.scatter_add_(2, plus_mask, input)
        out = torch.t(out[:,:,1] - out[:,:,0])
        # print('out', out)
        return out

def get_hashes_multiply(input, full_mask):
    # ratio need not be 1, but doesn't matter
    # input -> batch_size,in_dim
    # full_mask -> num_hashes,in_dim
    with torch.no_grad():
        return torch.mm(input, full_mask.t())

def get_indices_from_hashes(hash_values, K, L):
    with torch.no_grad():
        hash_values = (hash_values > 0).reshape(hash_values.size(0), L, K)
        twopowers = torch.IntTensor([2**i for i in range(K)])
        hash_indices = (hash_values*twopowers).sum(2)
        return hash_indices

def get_python_imp_latency(K, L, in_dim, batch_size, num_iter, num_threads):
    torch.set_num_threads(num_threads)
    device = torch.device('cpu')
    total_time_scatter = 0
    total_time_multiply = 0
    total_time_indices = 0
    num_hashes = K*L
    for iter in range(1, num_iter+1):
        input = torch.rand(batch_size, in_dim, device=device, requires_grad=True)
        plus_mask = (torch.rand(num_hashes,in_dim) > 0.5).to(torch.int64)
        full_mask = (plus_mask*2 - 1).float() # has to be same type as input

        t1 = time.time()
        hash_values_1 = get_hashes_scatter(input, plus_mask)
        t2 = time.time()
        hash_values_2 = get_hashes_multiply(input, full_mask)
        t3 = time.time()
        hash_indices = get_indices_from_hashes(hash_values_2, K, L)
        t4 = time.time()
        # print(torch.abs(hash_values_1-hash_values_2).max())

        if iter>num_iter/2:
            # warm start
            total_time_scatter += t2-t1
            total_time_multiply += t3-t2
            total_time_indices += t4-t3
    avg_time_scatter = (total_time_scatter/(num_iter/2)*1000)
    avg_time_multiply = (total_time_multiply/(num_iter/2)*1000)
    avg_time_indices = (total_time_indices/(num_iter/2)*1000)
    print('\nget_python_imp_latency','K',K,'L',L,'in_dim',in_dim,'batch_size',batch_size,'num_iter',num_iter,'num_threads',torch.get_num_threads())
    print('avg_time_scatter=%f' % (avg_time_scatter), 'avg_time_multiply=%f' % (avg_time_multiply), 'avg_time_indices=%f' % (avg_time_indices))


def get_cpp_imp_latency(K, L, in_dim, ratio, batch_size, num_iter, num_threads):
    torch.set_num_threads(num_threads)
    device = torch.device('cpu')
    ht = srpHashTable(K, L, in_dim, ratio)
    total_time_hash_values = 0
    total_time_hash_indices = 0
    for iter in range(1, num_iter+1):
        in_values = torch.rand(batch_size, in_dim, device=device, requires_grad=True)

        t1 = time.time()
        hash_values = ht.get_hash_values(in_values)
        t2 = time.time()
        hash_indices = ht.get_hash_indices_from_values(hash_values)
        t3 = time.time()
        if iter>num_iter/2:
            # warm start
            total_time_hash_values += t2-t1
            total_time_hash_indices += t3-t2
    avg_time_hash_values = (total_time_hash_values/(num_iter/2)*1000)
    avg_time_hash_indices = (total_time_hash_indices/(num_iter/2)*1000)
    print('\nget_cpp_imp_latency','K',K,'L',L,'in_dim',in_dim,'ratio',ratio,'batch_size',batch_size,'num_iter',num_iter,'num_threads',torch.get_num_threads())
    print('avg_time_hash_values=%f' % (avg_time_hash_values), 'avg_time_hash_indices=%f' % (avg_time_hash_indices))

def profile_latency(K, L, in_dim, batch_size, num_iter, num_threads):
    torch.set_num_threads(num_threads)
    device = torch.device('cpu')
    num_hashes = K*L
    for iter in range(1, num_iter+1):
        input = torch.rand(batch_size, in_dim, device=device, requires_grad=True)
        plus_mask = (torch.rand(num_hashes,in_dim) > 0.5).to(torch.int64)
        full_mask = (plus_mask*2 - 1).float() # has to be same type as input

        with profiler.profile(record_shapes=True) as prof_scatter:
            hash_values_1 = get_hashes_scatter(input, plus_mask)
        # with profiler.profile(record_shapes=True) as prof_multiply:
            # hash_values_2 = get_hashes_multiply(input, full_mask)
        with profiler.profile(record_shapes=True) as prof_indices:
            hash_indices = get_indices_from_hashes(hash_values_1, K, L)

        if iter == num_iter:
            # warm start
            print('\n\nprof_scatter')
            print(prof_scatter)
            print('\n\nprof_indices')
            print(prof_indices)
            
if __name__ == '__main__':
    # try all configurations used in previous report
    # split the hash computation and index computation steps for c++ implementation and find complexities
    # profile the python implementations to see if there is any memory copy

    K=9
    L=50
    num_iter = 100
    num_threads = 48

    ratio = 3
    # get_cpp_imp_latency(K=3, L=2, in_dim=8, ratio=ratio, batch_size=3, num_iter=num_iter, num_threads=num_threads)
    # get_cpp_imp_latency(K=K, L=L, in_dim=128, ratio=ratio, batch_size=128, num_iter=num_iter, num_threads=num_threads)
    # get_cpp_imp_latency(K=K, L=L, in_dim=128, ratio=ratio, batch_size=65536, num_iter=num_iter, num_threads=num_threads)
    # get_cpp_imp_latency(K=K, L=L, in_dim=128, ratio=ratio, batch_size=524288, num_iter=num_iter, num_threads=num_threads)
    # get_cpp_imp_latency(K=K, L=L, in_dim=4096, ratio=ratio, batch_size=4096, num_iter=num_iter, num_threads=num_threads)

    # get_python_imp_latency(K=3, L=2, in_dim=8, batch_size=3, num_iter=num_iter, num_threads=num_threads)
    # get_python_imp_latency(K=K, L=L, in_dim=128, batch_size=128, num_iter=num_iter, num_threads=num_threads)
    # get_python_imp_latency(K=K, L=L, in_dim=128, batch_size=65536, num_iter=num_iter, num_threads=num_threads)
    # get_python_imp_latency(K=K, L=L, in_dim=128, batch_size=524288, num_iter=num_iter, num_threads=num_threads)
    # get_python_imp_latency(K=K, L=L, in_dim=4096, batch_size=4096, num_iter=num_iter, num_threads=num_threads)

    profile_latency(K=K, L=L, in_dim=128, batch_size=65536, num_iter=5, num_threads=num_threads)


# get_python_imp_latency K 3 L 2 in_dim 8 batch_size 3 num_iter 100 num_threads 48
# avg_time_scatter=0.032072 avg_time_multiply=0.008135 avg_time_indices=0.030732

# get_python_imp_latency K 9 L 50 in_dim 128 batch_size 128 num_iter 100 num_threads 48
# avg_time_scatter=7.078056 avg_time_multiply=0.070691 avg_time_indices=0.309110

# get_python_imp_latency K 9 L 50 in_dim 128 batch_size 65536 num_iter 100 num_threads 48
# avg_time_scatter=282.678714 avg_time_multiply=44.840393 avg_time_indices=63.175011

# get_python_imp_latency K 9 L 50 in_dim 128 batch_size 524288 num_iter 100 num_threads 48
# avg_time_scatter=2285.107217 avg_time_multiply=127.950664 avg_time_indices=479.823742

# get_python_imp_latency K 9 L 50 in_dim 4096 batch_size 4096 num_iter 30 num_threads 48
# avg_time_scatter=512.334394 avg_time_multiply=12.313398 avg_time_indices=2.303298



###### test_6_out
# get_cpp_imp_latency K 3 L 2 in_dim 8 ratio 3 batch_size 3 num_iter 100 num_threads 48
# avg_time_hash_values=0.026622 avg_time_hash_indices=0.028100

# get_cpp_imp_latency K 9 L 50 in_dim 128 ratio 3 batch_size 128 num_iter 100 num_threads 48
# avg_time_hash_values=0.451207 avg_time_hash_indices=0.033121

# get_cpp_imp_latency K 9 L 50 in_dim 128 ratio 3 batch_size 65536 num_iter 100 num_threads 48
# avg_time_hash_values=220.999942 avg_time_hash_indices=12.384357

# get_cpp_imp_latency K 9 L 50 in_dim 128 ratio 3 batch_size 524288 num_iter 100 num_threads 48
# avg_time_hash_values=1617.763958 avg_time_hash_indices=75.198703

# get_cpp_imp_latency K 9 L 50 in_dim 4096 ratio 3 batch_size 4096 num_iter 100 num_threads 48
# avg_time_hash_values=435.085211 avg_time_hash_indices=1.128502

# get_python_imp_latency K 3 L 2 in_dim 8 batch_size 3 num_iter 100 num_threads 48
# avg_time_scatter=0.048094 avg_time_multiply=0.011978 avg_time_indices=0.047417

# get_python_imp_latency K 9 L 50 in_dim 128 batch_size 128 num_iter 100 num_threads 48
# avg_time_scatter=8.405485 avg_time_multiply=0.077491 avg_time_indices=0.292997

# get_python_imp_latency K 9 L 50 in_dim 128 batch_size 65536 num_iter 100 num_threads 48
# avg_time_scatter=316.956663 avg_time_multiply=41.595068 avg_time_indices=64.725332

# get_python_imp_latency K 9 L 50 in_dim 128 batch_size 524288 num_iter 100 num_threads 48
# avg_time_scatter=2199.914598 avg_time_multiply=129.728227 avg_time_indices=500.688524

# get_python_imp_latency K 9 L 50 in_dim 4096 batch_size 4096 num_iter 100 num_threads 48
# avg_time_scatter=507.452431 avg_time_multiply=11.031532 avg_time_indices=1.824517
