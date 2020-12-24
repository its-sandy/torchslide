import torch
import bucketsTable_cpp

class bucketsTable():
    def __init__(self, L, num_buckets, bucket_size):
        self.L = L
        self.num_buckets = num_buckets
        self.bucket_size = bucket_size

        self.buckets = torch.zeros([self.L, self.num_buckets, self.bucket_size], dtype=torch.int32)
        # bucket_counts holds the total number of nodes assigned to that bucket (can exceed bucket_size)
        self.bucket_counts = torch.zeros([self.L, self.num_buckets], dtype=torch.int32)
        # num_nodes holds total number of nodes filled into the buckets (can exceed num_buckets x bucket_size)
        self.num_nodes = 0
    
    def fill_buckets_FIFO(self, hash_indices):
        self.bucket_counts.fill_(0)
        self.num_nodes = hash_indices.size(0)
        bucketsTable_cpp.fill_buckets_FIFO(hash_indices, self.buckets, self.bucket_counts)

    def fill_buckets_reservoir_sampling(self, hash_indices):
        self.bucket_counts.fill_(0)
        self.num_nodes = hash_indices.size(0)
        bucketsTable_cpp.fill_buckets_reservoir_sampling(hash_indices, self.buckets, self.bucket_counts)

    def generate_randperm_nodes(self):
        # This has considerable latency
        self.randperm_nodes = torch.randperm(self.num_nodes, dtype=torch.int32)
    
    def sample_nodes_vanilla(self, hash_indices, sampled_nodes, presample_counts=None):
        # hash_indices = [batch_size x L] (dtype=torch.int32)
        # sampled_nodes = [batch_size x sample_size] (dtype=torch.int32)
        # presample_counts = [batch_size] (dtype=torch.int32)
        # If some nodes are always to be included in the sample, they should be
        # in the prefix of sampled_nodes, and their counts in presample_counts
        assert sampled_nodes.size(1) <= self.num_nodes, "Trying to sample more nodes than available in bucket"
        if presample_counts is None:
            presample_counts = torch.zeros([sampled_nodes.size(0)], dtype=torch.int32)
        
        bucketsTable_cpp.sample_nodes_vanilla(sampled_nodes,
                                              presample_counts,
                                              hash_indices,
                                              self.buckets,
                                              self.bucket_counts,
                                              self.randperm_nodes)

    def clear_buckets(self):
        self.bucket_counts.fill_(0)
        self.num_nodes = 0
