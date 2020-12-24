import torch
import bucketsTable_cpp
import time

def run_experiments():
    num_iter = 96 # multiple of 4

    num_threads = 48
    batch_size = 128
    sample_size = 2048
    # sample_size = 4096
    max_presample_size = 16
    L = 48
    K = 12
    num_buckets = 2**K
    bucket_size = 128 # assume power of 2
    num_nodes = 524288

    # for num_threads in [12, 24, 48]:
    #     for L in [12, 24, 48]:
    #         for num_nodes in [8192, 65536, 524288]:
    #             for num_buckets in [512, 4096]:

    #                 torch.set_num_threads(num_threads)
    #                 print('---------------------------------------------------')
    #                 print('\n\nL',L,'num_buckets',num_buckets,'bucket_size',bucket_size,'num_nodes',num_nodes,'num_iter',num_iter,'num_threads',torch.get_num_threads())

    #                 hash_indices_nodes = torch.randint(0, num_buckets, [num_nodes, L], dtype=torch.int32)
    #                 buckets = torch.empty([L, num_buckets, bucket_size], dtype=torch.int32).fill_(-1)
    #                 bucket_counts = torch.zeros([L, num_buckets], dtype=torch.int32)
    #                 randperm_nodes = torch.randperm(num_nodes, dtype=torch.int32)

    #                 total_time = 0
    #                 for iter in range(num_iter):
    #                     # buckets.fill_(-1)
    #                     bucket_counts.fill_(0)
    #                     t1 = time.time()
    #                     bucketsTable_cpp.fill_buckets_FIFO(hash_indices_nodes, buckets, bucket_counts)
    #                     t2 = time.time()
    #                     if iter>=num_iter/4:
    #                         total_time += t2-t1
    #                 print('\nfill_buckets_FIFO', (total_time/(3*num_iter/4)*1000))
                    
    #                 total_time = 0
    #                 for iter in range(num_iter):
    #                     # buckets.fill_(-1)
    #                     bucket_counts.fill_(0)
    #                     t1 = time.time()
    #                     bucketsTable_cpp.fill_buckets_FIFO_rand(hash_indices_nodes, buckets, bucket_counts)
    #                     t2 = time.time()
    #                     if iter>=num_iter/4:
    #                         total_time += t2-t1
    #                 print('\nfill_buckets_FIFO_rand', (total_time/(3*num_iter/4)*1000))
                    
    #                 total_time = 0
    #                 for iter in range(num_iter):
    #                     # buckets.fill_(-1)
    #                     bucket_counts.fill_(0)
    #                     t1 = time.time()
    #                     bucketsTable_cpp.fill_buckets_reservoir_sampling(hash_indices_nodes, buckets, bucket_counts)
    #                     t2 = time.time()
    #                     if iter>=num_iter/4:
    #                         total_time += t2-t1
    #                 print('\nfill_buckets_reservoir_sampling', (total_time/(3*num_iter/4)*1000))

    num_iter = 100
    num_threads = 48
    max_presample_size = 16
    time_diffs = list()

    # num_buckets = 2**K
    # for L in [12, 24, 48]:
    #     for num_buckets in [512, 4096]:
    #         for bucket_size in [64, 128]:
    #             for num_nodes in [1, 1024, 524288]:
    #                 for batch_size in [64, 128]:
    #                     for sample_size in [1024, 2048, 4096]:
    for L in [12, 24, 48]:
        for num_buckets in [512, 4096]:
            for bucket_size in [64, 128]:
                # for num_nodes in [1, 524288]:
                for num_nodes in [8192, 524288]:
                    for batch_size in [64, 128]:
                        for sample_size in [1024, 2048, 4096]:

                            torch.set_num_threads(num_threads)
                            print('---------------------------------------------------')
                            print('\n\nL',L,'num_buckets',num_buckets,'bucket_size',bucket_size,'num_nodes',num_nodes,'max_presample_size',max_presample_size,'batch_size',batch_size,'sample_size',sample_size,'num_iter',num_iter,'num_threads',torch.get_num_threads())

                            hash_indices_nodes = torch.randint(0, num_buckets, [num_nodes, L], dtype=torch.int32)
                            buckets = torch.empty([L, num_buckets, bucket_size], dtype=torch.int32).fill_(-1)
                            bucket_counts = torch.zeros([L, num_buckets], dtype=torch.int32)
                            randperm_nodes = torch.randperm(num_nodes, dtype=torch.int32)

                            # bucketsTable_cpp.fill_buckets_FIFO(hash_indices_nodes, buckets, bucket_counts)
                            bucketsTable_cpp.fill_buckets_reservoir_sampling(hash_indices_nodes, buckets, bucket_counts)

                            sampled_nodes = torch.randint(0, num_nodes, [batch_size, sample_size], dtype=torch.int32)
                            presample_counts = torch.randint(0, max_presample_size, [batch_size], dtype=torch.int32)
                            
                            total_time = 0
                            for iter in range(num_iter):
                                hash_indices_batch = torch.randint(0, num_buckets, [batch_size, L], dtype=torch.int32)
                                t1 = time.time()
                                bucketsTable_cpp.sample_nodes_vanilla_1(sampled_nodes, presample_counts, hash_indices_batch, buckets, bucket_counts, randperm_nodes)
                                t2 = time.time()
                                if iter>=num_iter/4:
                                    total_time += t2-t1
                            print('\nsample_nodes_vanilla_1', (total_time/(3*num_iter/4)*1000))
                            time_1 = (total_time/(3*num_iter/4)*1000)
                            
                            total_time = 0
                            for iter in range(num_iter):
                                hash_indices_batch = torch.randint(0, num_buckets, [batch_size, L], dtype=torch.int32)
                                t1 = time.time()
                                bucketsTable_cpp.sample_nodes_vanilla_2(sampled_nodes, presample_counts, hash_indices_batch, buckets, bucket_counts, randperm_nodes)
                                t2 = time.time()
                                if iter>=num_iter/4:
                                    total_time += t2-t1
                            print('\nsample_nodes_vanilla_2', (total_time/(3*num_iter/4)*1000))
                            time_2 = (total_time/(3*num_iter/4)*1000)

                            time_diffs.append(time_2 - time_1)

                            # total_time = 0
                            # for iter in range(num_iter):
                            #     hash_indices_batch = torch.randint(0, num_buckets, [batch_size, L], dtype=torch.int32)
                            #     t1 = time.time()
                            #     bucketsTable_cpp.sample_nodes_vanilla_3(sampled_nodes, presample_counts, hash_indices_batch, buckets, bucket_counts, randperm_nodes)
                            #     t2 = time.time()
                            #     if iter>=num_iter/4:
                            #         total_time += t2-t1
                            # print('\nsample_nodes_vanilla_3', (total_time/(3*num_iter/4)*1000))

    print(time_diffs)


if __name__ == '__main__':

    run_experiments()    


    # n = 100
    # x = torch.rand (48, n).argsort (dim = 1)
    # for i in range(100):
    #     # perm = torch.randperm(670091, dtype=torch.int32)
    #     # x = torch.rand (48, n).argsort (dim = 1)
    #     x = x[:,torch.randperm(n)]
    # b = time.time()
    # # print(perm)
    # print(x)
    # print(((b-a)/100)*1000)