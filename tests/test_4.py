import torch
import sys
import time
sys.path.insert(0, '/home/santhosh/torchslide/SLIDE') 
from srpHash import srpHashTable

def manual_tests():
    L = 2
    K = 3
    dim = 8
    ratio = 3
    b = 4
    active_in_dim = 2
    t = srpHashTable(K, L, dim, ratio)
    print('t.nz_dim', t.nz_dim)
    print('t.nz_indices', t.nz_indices)
    print('t.plus_mask', t.plus_mask)
    print('t.minus_mask', t.minus_mask)

    in_values = torch.rand(b, dim, requires_grad=True)
    print('in_values', in_values)
    print('t.get_hash_indices(in_values)', t.get_hash_indices(in_values))

    in_values = torch.rand(b, active_in_dim, requires_grad=True)
    active_in_indices = torch.randint(0, dim, [b, active_in_dim])
    print('in_values', in_values)
    print('active_in_indices', active_in_indices)
    print('active_in_indices.dtype', active_in_indices.dtype)
    print('t.get_hash_indices(in_values, active_in_indices)', t.get_hash_indices(in_values, active_in_indices))

def get_hash_reset_latency(K, L, in_dim, ratio, num_iter, num_threads):
    torch.set_num_threads(num_threads)
    ht = srpHashTable(K, L, in_dim, ratio)
    total_time = 0
    for iter in range(1, num_iter+1):
        t1 = time.time()
        ht.reset_hashes()
        t2 = time.time()
        if iter>num_iter/2:
            # warm start
            total_time += t2-t1

    print('get_reset_hashes_latency','K',K,'L',L,'in_dim',in_dim,'ratio',ratio,'num_iter',num_iter,'num_threads',torch.get_num_threads(),'avg_iter_time=%f' % (total_time/(num_iter/2)*1000))

def get_hash_computation_latency(K, L, in_dim, ratio, batch_size, active_in_dim, num_iter, num_threads):
    torch.set_num_threads(num_threads)
    device = torch.device('cpu')
    ht = srpHashTable(K, L, in_dim, ratio)
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
    print('get_hash_computation_latency','K',K,'L',L,'in_dim',in_dim,'ratio',ratio,'batch_size',batch_size,'active_in_dim',active_in_dim,'num_iter',num_iter,'num_threads',torch.get_num_threads(),'avg_iter_time=%f' % (total_time/(num_iter/2)*1000))
    
if __name__ == "__main__":

    # torch.manual_seed(0)
    manual_tests()
    num_threads=48
    num_iter=100

    # get_hash_computation_latency(K=9,
    #                              L=50,
    #                              in_dim=4096,
    #                              ratio=3,
    #                              batch_size=128,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    # get_hash_computation_latency(K=9,
    #                              L=50,
    #                              in_dim=4096,
    #                              ratio=3,
    #                              batch_size=4096,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    # get_hash_computation_latency(K=6,
    #                              L=50,
    #                              in_dim=4096,
    #                              ratio=3,
    #                              batch_size=128,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    # get_hash_computation_latency(K=6,
    #                              L=50,
    #                              in_dim=4096,
    #                              ratio=3,
    #                              batch_size=4096,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    # get_hash_computation_latency(K=6,
    #                              L=25,
    #                              in_dim=4096,
    #                              ratio=3,
    #                              batch_size=128,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    # get_hash_computation_latency(K=6,
    #                              L=25,
    #                              in_dim=4096,
    #                              ratio=3,
    #                              batch_size=4096,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)







    # K=6
    # L=50
    # get_hash_reset_latency(K=K,
    #                        L=L,
    #                        in_dim=128,
    #                        ratio=3,
    #                        num_iter=num_iter,
    #                        num_threads=num_threads)
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=128,
    #                              ratio=3,
    #                              batch_size=128,
    #                              active_in_dim=64,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=128,
    #                              ratio=3,
    #                              batch_size=524288,
    #                              active_in_dim=64,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)

    # # pytorch c++ extension sheet on google sheets
    # K=9
    # L=50
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=64,
    #                              ratio=3,
    #                              batch_size=64,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=64,
    #                              ratio=3,
    #                              batch_size=16384,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=128,
    #                              ratio=3,
    #                              batch_size=128,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=128,
    #                              ratio=3,
    #                              batch_size=65536,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=128,
    #                              ratio=3,
    #                              batch_size=524288,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)

    # # changing K and L
    # K=6
    # L=50
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=64,
    #                              ratio=3,
    #                              batch_size=64,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=64,
    #                              ratio=3,
    #                              batch_size=16384,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=128,
    #                              ratio=3,
    #                              batch_size=128,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=128,
    #                              ratio=3,
    #                              batch_size=65536,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=128,
    #                              ratio=3,
    #                              batch_size=524288,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    
    # # changing K and L
    # K=6
    # L=25
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=64,
    #                              ratio=3,
    #                              batch_size=64,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=64,
    #                              ratio=3,
    #                              batch_size=16384,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=128,
    #                              ratio=3,
    #                              batch_size=128,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=128,
    #                              ratio=3,
    #                              batch_size=65536,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
    # get_hash_computation_latency(K=K,
    #                              L=L,
    #                              in_dim=128,
    #                              ratio=3,
    #                              batch_size=524288,
    #                              active_in_dim=-1,
    #                              num_iter=num_iter,
    #                              num_threads=num_threads)
