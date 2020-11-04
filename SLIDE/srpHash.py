import math
import torch
import srpHash_cpp


class srpHashTable():
    def __init__(self, K, L, dim, ratio):
        self.K = K
        self.L = L
        self.dim = dim
        self.nz_dim = math.ceil(dim / ratio)
        
        self.nz_indices = torch.empty([L, K, self.nz_dim], dtype=torch.int32)
        self.plus_mask = torch.empty([L, K, dim], dtype=torch.bool)
        self.minus_mask = torch.empty([L, K, dim], dtype=torch.bool)
        self.reset_hashes()

    def reset_hashes(self):
        srpHash_cpp.reset_hashes(self.nz_indices,
                                 self.plus_mask,
                                 self.minus_mask)

    def get_hash_indices(self, in_values, active_in_indices=None):
        hash_indices = torch.empty([in_values.size(0), self.L], dtype=torch.int32)
        if active_in_indices is None:
            srpHash_cpp.get_hash_indices_dense(hash_indices,
                                               in_values,
                                               self.nz_indices,
                                               self.plus_mask)
        else:
            srpHash_cpp.get_hash_indices_sparse(hash_indices,
                                                in_values,
                                                active_in_indices,
                                                self.plus_mask,
                                                self.minus_mask)
        return hash_indices
