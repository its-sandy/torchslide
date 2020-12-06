import torch
import sys
import time
sys.path.insert(0, '/home/santhosh/torchslide/SLIDE') 
from srpHash import srpHashTable

def manual_tests():
    L = 2
    K = 3
    dim = 8
    b = 4
    active_in_dim = 2
    t = srpHashTable(K, L, dim)
    print('t.hash_vecs\n', t.hash_vecs)

    in_values = torch.rand(b, dim, requires_grad=True)
    print('in_values\n', in_values)
    print('t.get_hash_indices(in_values)\n', t.get_hash_indices(in_values))

    in_values = torch.rand(b, active_in_dim, requires_grad=True)
    active_in_indices = torch.randint(0, dim, [b, active_in_dim])
    print('in_values\n', in_values)
    print('active_in_indices\n', active_in_indices)
    print('active_in_indices.dtype', active_in_indices.dtype)
    print('t.get_hash_indices(in_values, active_in_indices)\n', t.get_hash_indices(in_values, active_in_indices))

def correctness_tests():
    L = 50
    K = 30
    dim = 128
    b = 4096
    t = srpHashTable(K, L, dim)
    in_values = torch.rand(b, dim, requires_grad=True)
    hash_indices_1 = t.get_hash_indices(in_values)
    active_in_indices = torch.arange(0, dim).repeat(b, 1)
    hash_indices_2 = t.get_hash_indices(in_values, active_in_indices)
    print('(hash_indices_1 - hash_indices_2 != 0).sum()', (hash_indices_1 - hash_indices_2 != 0).sum())

def get_hash_reset_latency(K, L, in_dim, num_iter, num_threads):
    torch.set_num_threads(num_threads)
    ht = srpHashTable(K, L, in_dim)
    total_time = 0
    for iter in range(1, num_iter+1):
        t1 = time.time()
        ht.reset_hashes()
        t2 = time.time()
        if iter>num_iter/2:
            # warm start
            total_time += t2-t1

    print('get_reset_hashes_latency','K',K,'L',L,'in_dim',in_dim,'num_iter',num_iter,'num_threads',torch.get_num_threads(),'avg_iter_time=%f' % (total_time/(num_iter/2)*1000))

def get_hash_computation_latency(K, L, in_dim, batch_size, active_in_dim, num_iter, num_threads):
    torch.set_num_threads(num_threads)
    device = torch.device('cpu')
    ht = srpHashTable(K, L, in_dim)
    total_time = 0
    for iter in range(1, num_iter+1):
        if active_in_dim == -1:
            active_in_indices = None
            in_values = torch.rand(batch_size, in_dim, device=device, requires_grad=True)
        else:
            active_in_indices = torch.randint(0, in_dim, [batch_size, active_in_dim], device=device)
            in_values = torch.rand(batch_size, active_in_dim, device=device, requires_grad=True)

        t1 = time.time()
        hash_indices = ht.get_hash_indices(in_values, active_in_indices)
        t2 = time.time()
        if iter>num_iter/2:
            # warm start
            total_time += t2-t1
    print('get_hash_computation_latency','K',K,'L',L,'in_dim',in_dim,'batch_size',batch_size,'active_in_dim',active_in_dim,'num_iter',num_iter,'num_threads',torch.get_num_threads(),'avg_iter_time=%f' % (total_time/(num_iter/2)*1000))
    
if __name__ == "__main__":

    # torch.manual_seed(0)
    # manual_tests()
    # correctness_tests()
    num_threads=48
    num_iter=500


    # pytorch c++ extension sheet on google sheets
    for K, L in [(9, 50), (6, 50), (6, 25)]:
        for in_dim, batch_size in [(64, 64),
                                   (64, 16384),
                                   (128, 128),
                                   (128, 65536),
                                   (128, 524288),
                                   (4096, 128),
                                   (4096, 4096)]:
            get_hash_computation_latency(K=K,
                                         L=L,
                                         in_dim=in_dim,
                                         batch_size=batch_size,
                                         active_in_dim=-1,
                                         num_iter=num_iter,
                                         num_threads=num_threads)


    # sparse
    K=6
    L=50
    get_hash_reset_latency(K=K,
                           L=L,
                           in_dim=128,
                           num_iter=num_iter,
                           num_threads=num_threads)
    get_hash_computation_latency(K=K,
                                 L=L,
                                 in_dim=128,
                                 batch_size=128,
                                 active_in_dim=64,
                                 num_iter=num_iter,
                                 num_threads=num_threads)
    get_hash_computation_latency(K=K,
                                 L=L,
                                 in_dim=128,
                                 batch_size=524288,
                                 active_in_dim=64,
                                 num_iter=num_iter,
                                 num_threads=num_threads)
