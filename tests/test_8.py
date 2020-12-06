import torch
import bucketsTable_cpp
import time

def run_experiments():
    torch.set_num_threads(48)
    num_iter = 200

    batch_size = 128
    sample_size = 2048
    # sample_size = 4096
    max_presample_size = 16
    L = 48
    K = 9
    num_buckets = 2**K
    bucket_size = 128 # assume power of 2
    num_nodes = 524288

    print('L',L,'K',K,'num_buckets',num_buckets,'bucket_size',bucket_size,'num_nodes',num_nodes,'max_presample_size',max_presample_size,'batch_size',batch_size,'sample_size',sample_size,'num_iter',num_iter,'num_threads',torch.get_num_threads())

    hash_indices_nodes = torch.randint(0, num_buckets, [num_nodes, L], dtype=torch.int32)
    buckets = torch.empty([L, num_buckets, bucket_size], dtype=torch.int32).fill_(-1)
    bucket_counts = torch.zeros([L, num_buckets], dtype=torch.int32)
    randperm_nodes = torch.randperm(num_nodes, dtype=torch.int32)

    total_time = 0
    for iter in range(num_iter):
        # buckets.fill_(-1)
        bucket_counts.fill_(0)
        t1 = time.time()
        bucketsTable_cpp.fill_buckets_FIFO(hash_indices_nodes, buckets, bucket_counts)
        t2 = time.time()
        if iter>=num_iter/2:
            total_time += t2-t1
    print('\nfill_buckets_FIFO', (total_time/(num_iter/2)*1000))
    print('bucket_counts.sum(dim=1).true_divide(num_nodes)', bucket_counts.sum(dim=1).true_divide(num_nodes))
    print('bucket_counts.sum(dim=1).true_divide(num_nodes).median()', bucket_counts.sum(dim=1).true_divide(num_nodes).median())

    total_time = 0
    for iter in range(num_iter):
        # buckets.fill_(-1)
        bucket_counts.fill_(0)
        t1 = time.time()
        bucketsTable_cpp.fill_buckets_FIFO_2(hash_indices_nodes, buckets, bucket_counts, randperm_nodes)
        t2 = time.time()
        if iter>=num_iter/2:
            total_time += t2-t1
    print('\nfill_buckets_FIFO_2', (total_time/(num_iter/2)*1000))
    print('bucket_counts.sum(dim=1).true_divide(num_nodes)', bucket_counts.sum(dim=1).true_divide(num_nodes))
    print('bucket_counts.sum(dim=1).true_divide(num_nodes).median()', bucket_counts.sum(dim=1).true_divide(num_nodes).median())

    total_time = 0
    for iter in range(num_iter):
        # buckets.fill_(-1)
        bucket_counts.fill_(0)
        t1 = time.time()
        bucketsTable_cpp.fill_buckets_FIFO_3(hash_indices_nodes, buckets, bucket_counts)
        t2 = time.time()
        if iter>=num_iter/2:
            total_time += t2-t1
    print('\nfill_buckets_FIFO_3', (total_time/(num_iter/2)*1000))
    print('bucket_counts.sum(dim=1).true_divide(num_nodes)', bucket_counts.sum(dim=1).true_divide(num_nodes))
    print('bucket_counts.sum(dim=1).true_divide(num_nodes).median()', bucket_counts.sum(dim=1).true_divide(num_nodes).median())

    total_time = 0
    for iter in range(num_iter):
        # buckets.fill_(-1)
        bucket_counts.fill_(0)
        t1 = time.time()
        bucketsTable_cpp.fill_buckets_FIFO_4(hash_indices_nodes, buckets, bucket_counts)
        t2 = time.time()
        if iter>=num_iter/2:
            total_time += t2-t1
    print('\nfill_buckets_FIFO_4', (total_time/(num_iter/2)*1000))
    print('bucket_counts.sum(dim=1).true_divide(num_nodes)', bucket_counts.sum(dim=1).true_divide(num_nodes))
    print('bucket_counts.sum(dim=1).true_divide(num_nodes).median()', bucket_counts.sum(dim=1).true_divide(num_nodes).median())

    total_time = 0
    for iter in range(num_iter):
        # buckets.fill_(-1)
        bucket_counts.fill_(0)
        t1 = time.time()
        bucketsTable_cpp.fill_buckets_reservoir_sampling(hash_indices_nodes, buckets, bucket_counts)
        t2 = time.time()
        if iter>=num_iter/2:
            total_time += t2-t1
    print('\nfill_buckets_reservoir_sampling', (total_time/(num_iter/2)*1000))
    print('bucket_counts.sum(dim=1).true_divide(num_nodes)', bucket_counts.sum(dim=1).true_divide(num_nodes))
    print('bucket_counts.sum(dim=1).true_divide(num_nodes).median()', bucket_counts.sum(dim=1).true_divide(num_nodes).median())

    total_time = 0
    for iter in range(num_iter):
        # buckets.fill_(-1)
        bucket_counts.fill_(0)
        t1 = time.time()
        bucketsTable_cpp.fill_buckets_reservoir_sampling_2(hash_indices_nodes, buckets, bucket_counts)
        t2 = time.time()
        if iter>=num_iter/2:
            total_time += t2-t1
    print('\nfill_buckets_reservoir_sampling_2', (total_time/(num_iter/2)*1000))
    print('bucket_counts.sum(dim=1).true_divide(num_nodes)', bucket_counts.sum(dim=1).true_divide(num_nodes))
    print('bucket_counts.sum(dim=1).true_divide(num_nodes).median()', bucket_counts.sum(dim=1).true_divide(num_nodes).median())

    sampled_nodes = torch.randint(0, num_nodes, [batch_size, sample_size], dtype=torch.int32)
    presample_counts = torch.randint(0, max_presample_size, [batch_size], dtype=torch.int32)
    hash_indices_batch = torch.randint(0, num_buckets, [batch_size, L], dtype=torch.int32)
    
    total_time = 0
    for iter in range(num_iter):
        t1 = time.time()
        bucketsTable_cpp.sample_nodes_vanilla(sampled_nodes, presample_counts, hash_indices_batch, buckets, bucket_counts, randperm_nodes)
        t2 = time.time()
        if iter>=num_iter/2:
            total_time += t2-t1
    print('\nsample_nodes_vanilla', (total_time/(num_iter/2)*1000))

    total_time = 0
    for iter in range(num_iter):
        t1 = time.time()
        bucketsTable_cpp.sample_nodes_vanilla_2(sampled_nodes, presample_counts, hash_indices_batch, buckets, bucket_counts, randperm_nodes)
        t2 = time.time()
        if iter>=num_iter/2:
            total_time += t2-t1
    print('\nsample_nodes_vanilla_2', (total_time/(num_iter/2)*1000))

    total_time = 0
    for iter in range(num_iter):
        t1 = time.time()
        bucketsTable_cpp.sample_nodes_vanilla_3(sampled_nodes, presample_counts, hash_indices_batch, buckets, bucket_counts, randperm_nodes)
        t2 = time.time()
        if iter>=num_iter/2:
            total_time += t2-t1
    print('\nsample_nodes_vanilla_3', (total_time/(num_iter/2)*1000))


if __name__ == '__main__':

    run_experiments()    

    # sampled_nodes = torch.randint(0, num_nodes, [batch_size, sample_size], dtype=torch.int32)
    # presample_counts = torch.randint(0, max_presample_size, [batch_size], dtype=torch.int32)
    # hash_indices_nodes = torch.randint(0, num_buckets, [num_nodes, L], dtype=torch.int32)
    # hash_indices_batch = torch.randint(0, num_buckets, [batch_size, L], dtype=torch.int32)
    # buckets = torch.empty([L, num_buckets, bucket_size], dtype=torch.int32).fill_(-1)
    # bucket_counts = torch.zeros([L, num_buckets], dtype=torch.int32)
    # randperm_nodes = torch.randperm(num_nodes, dtype=torch.int32)

    # print('randperm_nodes', randperm_nodes)

    # buckets = torch.empty([L, num_buckets, bucket_size], dtype=torch.int32).fill_(-1)
    # bucket_counts = torch.zeros([L, num_buckets], dtype=torch.int32)
    # bucketsTable_cpp.fill_buckets_FIFO(hash_indices_nodes, buckets, bucket_counts)
    # print('\n\nafter fill_buckets_FIFO')
    # print('hash_indices_nodes', hash_indices_nodes)
    # print('buckets', buckets)
    # print('bucket_counts', bucket_counts)

    # buckets = torch.empty([L, num_buckets, bucket_size], dtype=torch.int32).fill_(-1)
    # bucket_counts = torch.zeros([L, num_buckets], dtype=torch.int32)
    # bucketsTable_cpp.fill_buckets_FIFO_2(hash_indices_nodes, buckets, bucket_counts, randperm_nodes)
    # print('\n\nafter fill_buckets_FIFO_2')
    # print('hash_indices_nodes', hash_indices_nodes)
    # print('buckets', buckets)
    # print('bucket_counts', bucket_counts)

    # buckets = torch.empty([L, num_buckets, bucket_size], dtype=torch.int32).fill_(-1)
    # bucket_counts = torch.zeros([L, num_buckets], dtype=torch.int32)
    # bucketsTable_cpp.fill_buckets_FIFO_3(hash_indices_nodes, buckets, bucket_counts)
    # print('\n\nafter fill_buckets_FIFO_3')
    # print('hash_indices_nodes', hash_indices_nodes)
    # print('buckets', buckets)
    # print('bucket_counts', bucket_counts)

    # buckets = torch.empty([L, num_buckets, bucket_size], dtype=torch.int32).fill_(-1)
    # bucket_counts = torch.zeros([L, num_buckets], dtype=torch.int32)
    # bucketsTable_cpp.fill_buckets_FIFO_4(hash_indices_nodes, buckets, bucket_counts)
    # print('\n\nafter fill_buckets_FIFO_4')
    # print('hash_indices_nodes', hash_indices_nodes)
    # print('buckets', buckets)
    # print('bucket_counts', bucket_counts)

    # # #######
    # buckets = torch.empty([L, num_buckets, bucket_size], dtype=torch.int32).fill_(-1)
    # bucket_counts = torch.zeros([L, num_buckets], dtype=torch.int32)
    # bucketsTable_cpp.fill_buckets_reservoir_sampling(hash_indices_nodes, buckets, bucket_counts)
    # print('\n\nafter fill_buckets_reservoir_sampling')
    # print('hash_indices_nodes', hash_indices_nodes)
    # print('buckets', buckets)
    # print('bucket_counts', bucket_counts)

    # buckets = torch.empty([L, num_buckets, bucket_size], dtype=torch.int32).fill_(-1)
    # bucket_counts = torch.zeros([L, num_buckets], dtype=torch.int32)
    # bucketsTable_cpp.fill_buckets_reservoir_sampling_2(hash_indices_nodes, buckets, bucket_counts)
    # print('\n\nafter fill_buckets_reservoir_sampling_2')
    # print('hash_indices_nodes', hash_indices_nodes)
    # print('buckets', buckets)
    # print('bucket_counts', bucket_counts)

    # # #######
    # sampled_nodes = torch.randint(0, num_nodes, [batch_size, sample_size], dtype=torch.int32)
    # presample_counts = torch.randint(0, max_presample_size, [batch_size], dtype=torch.int32)
    # print('\n\nsampled_nodes', sampled_nodes)
    # print('presample_counts', presample_counts)
    # bucketsTable_cpp.sample_nodes_vanilla(sampled_nodes, presample_counts, hash_indices_batch, buckets, bucket_counts, randperm_nodes)
    # print('after sample_nodes_vanilla')
    # print('sampled_nodes', sampled_nodes)
    # print('presample_counts', presample_counts)
    # print('hash_indices_batch', hash_indices_batch)

    # sampled_nodes = torch.randint(0, num_nodes, [batch_size, sample_size], dtype=torch.int32)
    # presample_counts = torch.randint(0, max_presample_size, [batch_size], dtype=torch.int32)
    # bucketsTable_cpp.sample_nodes_vanilla_2(sampled_nodes, presample_counts, hash_indices_batch, buckets, bucket_counts, randperm_nodes)
    # print('\n\nsampled_nodes', sampled_nodes)
    # print('presample_counts', presample_counts)
    # print('after sample_nodes_vanilla_2')
    # print('sampled_nodes', sampled_nodes)
    # print('presample_counts', presample_counts)
    # print('hash_indices_batch', hash_indices_batch)

    # sampled_nodes = torch.randint(0, num_nodes, [batch_size, sample_size], dtype=torch.int32)
    # presample_counts = torch.randint(0, max_presample_size, [batch_size], dtype=torch.int32)
    # bucketsTable_cpp.sample_nodes_vanilla_3(sampled_nodes, presample_counts, hash_indices_batch, buckets, bucket_counts, randperm_nodes)
    # print('\n\nsampled_nodes', sampled_nodes)
    # print('presample_counts', presample_counts)
    # print('after sample_nodes_vanilla_3')
    # print('sampled_nodes', sampled_nodes)
    # print('presample_counts', presample_counts)
    # print('hash_indices_batch', hash_indices_batch)

    





    # a = time.time()
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