import torch
import torch.nn as nn
from bucketsTable import bucketsTable
from cppSparseMultiply import cppSparseMultiply
# from pySparseMultiply import pySparseMultiply
from srpHash import srpHashTable


class slideLayer(nn.Module):
    def __init__(
        self, in_dim, out_dim, K, L, bucket_size=128, fill_mode='FIFO', sample_mode='vanilla'):
        super(slideLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K
        self.L = L
        self.num_buckets = 2**K # for srpHash
        self.bucket_size = bucket_size
        self.fill_mode = fill_mode
        self.sample_mode = sample_mode

        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.sparse_multiplier = cppSparseMultiply.apply
        # self.sparse_multiplier = pySparseMultiply
        self.hash_table = srpHashTable(self.K, self.L, self.in_dim)
        self.buckets_table = bucketsTable(self.L, self.num_buckets, self.bucket_size)

        self.rehash_nodes(reset_hashes=False, reset_randperm_nodes=True)

    def rehash_nodes(self, reset_hashes=True, reset_randperm_nodes=False):
        if reset_hashes:
            self.hash_table.reset_hashes()

        hash_indices = self.hash_table.get_hash_indices(self.linear.weight)
        if self.fill_mode == 'FIFO':
            self.buckets_table.fill_buckets_FIFO(hash_indices)
        elif self.fill_mode == 'reservoir_sampling':
            self.buckets_table.fill_buckets_reservoir_sampling(hash_indices)
        else:
            raise ValueError('Invalid Buckets Table Fill Mode:', self.fill_mode)
        
        if reset_randperm_nodes:
            self.buckets_table.generate_randperm_nodes()

    def forward(self, in_values, active_in_indices=None, active_out_indices=None, presample_counts=None):
        # active_out_indices expected to be of size [batch_size x sample_size], dtype int32
        # If some nodes are always to be included in the sample, they should be
        # in the prefix of active_out_indices, and their counts in presample_counts
        # if presample_counts is None, means no pre samples
        # If active_out_indices is None, returns dense output

        if active_out_indices is None:
            # Dense Output
            out_values = self.sparse_multiplier(
                in_values, self.linear.weight, self.linear.bias, active_in_indices,None)
            return out_values, None
        
        else:
            # Sparse output
            hash_indices = self.hash_table.get_hash_indices(in_values, active_in_indices)
            if self.sample_mode == 'vanilla':
                self.buckets_table.sample_nodes_vanilla(hash_indices, active_out_indices, presample_counts)
            else:
                raise ValueError('Invalid Buckets Table Sampling Mode:', self.sample_mode)
            active_out_indices = active_out_indices.long()
            out_values = self.sparse_multiplier(
                in_values, self.linear.weight, self.linear.bias, active_in_indices,active_out_indices)
            return out_values, active_out_indices
