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
        self.num_nodes = 0
    
    def fill_buckets(self, hash_indices):
        # Works best when L is larger than num_threads (to avoid inconsistencies because of race conditions)
        # Ideally L is a multiple of num_threadss
        self.bucket_counts.fill_(0)
        self.num_nodes = hash_indices.size(0)
        bucketsTable_cpp.fill_buckets(hash_indices, self.buckets, self.bucket_counts)

    def sample_nodes(self, hash_indices, active_nodes=None):
        # either active_nodes should be partially pre filled with the required label nodes
        # else, figure a way to send lists to pytorch c++ extensions
        # maybe look into nested tensor
        if required_nodes > self.num_nodes:
            raise Exception
        bucketsTable_cpp.sample_nodes_vanilla(self.buckets, self.bucket_counts, hash_indices)
