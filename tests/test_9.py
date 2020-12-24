import torch
import sys
sys.path.insert(0, '/home/santhosh/torchslide/SLIDE')
from bucketsTable import bucketsTable
import time

def run_experiments():
    num_iter = 200 # multiple of 4
    num_threads = 48
    max_presample_size = 16
    bucket_size = 128

    print('num_iter',num_iter,'num_threads',torch.get_num_threads())
    print('bucket_size',bucket_size)
    for K, L in [(12, 50), (9, 50), (6, 50), (6, 25)]:
        for num_nodes, batch_size in [(16384, 64), (65536, 128), (524288, 128)]:
            print('---------------------------------------------------')
            print('\n\nL',L,'K',K,'num_nodes',num_nodes,'batch_size',batch_size)
            num_buckets = 2**K
            torch.set_num_threads(num_threads)

            buckets_table = bucketsTable(L, num_buckets, bucket_size)
            hash_indices_nodes = torch.randint(0, num_buckets, [num_nodes, L], dtype=torch.int32)

            total_time = 0
            for iter in range(num_iter):
                buckets_table.clear_buckets()
                t1 = time.time()
                buckets_table.fill_buckets_FIFO(hash_indices_nodes)
                t2 = time.time()
                if iter>=num_iter/4:
                    total_time += t2-t1
            print('\nfill_buckets_FIFO', (total_time/(3*num_iter/4)*1000))
            
            total_time = 0
            for iter in range(num_iter):
                buckets_table.clear_buckets()
                t1 = time.time()
                buckets_table.fill_buckets_reservoir_sampling(hash_indices_nodes)
                t2 = time.time()
                if iter>=num_iter/4:
                    total_time += t2-t1
            print('fill_buckets_reservoir_sampling', (total_time/(3*num_iter/4)*1000))
                    
            buckets_table.generate_randperm_nodes()

            for sample_size in [1024, 2048, 4096]:
                sampled_nodes = torch.randint(0, num_nodes, [batch_size, sample_size], dtype=torch.int32)
                presample_counts = torch.randint(0, max_presample_size, [batch_size], dtype=torch.int32)
                
                total_time = 0
                for iter in range(num_iter):
                    hash_indices_batch = torch.randint(0, num_buckets, [batch_size, L], dtype=torch.int32)
                    t1 = time.time()
                    buckets_table.sample_nodes_vanilla(hash_indices_batch, sampled_nodes, presample_counts)
                    t2 = time.time()
                    if iter>=num_iter/4:
                        total_time += t2-t1
                print('before_clearing', 'sample_size', sample_size, 'avg_time', (total_time/(3*num_iter/4)*1000))
                    
            # buckets_table.clear_buckets()
            buckets_table.bucket_counts.fill_(0)

            for sample_size in [1024, 2048, 4096]:
                sampled_nodes = torch.randint(0, num_nodes, [batch_size, sample_size], dtype=torch.int32)
                presample_counts = torch.randint(0, max_presample_size, [batch_size], dtype=torch.int32)
                
                total_time = 0
                for iter in range(num_iter):
                    hash_indices_batch = torch.randint(0, num_buckets, [batch_size, L], dtype=torch.int32)
                    t1 = time.time()
                    buckets_table.sample_nodes_vanilla(hash_indices_batch, sampled_nodes, presample_counts)
                    t2 = time.time()
                    if iter>=num_iter/4:
                        total_time += t2-t1
                print('after_clearing', 'sample_size', sample_size, 'avg_time', (total_time/(3*num_iter/4)*1000))
                    

def run_experiments_perm():
    num_iter = 200
    num_threads = 48
    torch.set_num_threads(num_threads)

    for num_nodes in [16384, 65536, 524288]:
        t1 = time.time()
        for iter in range(num_iter):
            randperm_nodes = torch.randperm(num_nodes, dtype=torch.int32)
        t2 = time.time()
        print('randperm_nodes', num_nodes, 'nodes' , (t2-t1)/num_iter*1000)

if __name__ == '__main__':

    run_experiments()
    run_experiments_perm()