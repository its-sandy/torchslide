import math
import torch
import wtaHash_cpp


class wtaHashTable():
    def __init__(self, K, L, dim, perm_size=8):
        self.K = K
        self.L = L
        self.dim = dim
        self.perm_size = perm_size
        # ideal if dim is a multiple of perm_size
        self.shift_len = math.ceil(math.log2(self.perm_size))

        self.num_hashes = K*L
        self.num_full_perms = math.ceil((self.num_hashes*self.perm_size)/self.dim)
        self.reset_hashes()

    def reset_hashes(self):
        # for large self.dim, this is better than using argsort for batch of randperm 
        # you can parallelize this using C++ extension if required
        self.perm_ind = torch.empty([self.num_full_perms, self.dim], dtype=torch.int32)
        for i in range(self.num_full_perms):
            torch.randperm(self.dim, out=self.perm_ind[i])

        (self.perm_ind).add_(
            (torch.arange(self.num_full_perms, dtype=torch.int32)*self.dim).unsqueeze(-1))

        self.perm_pos = self.perm_ind % self.perm_size
        (self.perm_ind).floor_divide_(self.perm_size)

    def get_hash_indices(self, in_values, active_in_indices=None):
        hash_indices = torch.empty([in_values.size(0), self.L], dtype=torch.int32)
        if active_in_indices is None:
            wtaHash_cpp.get_hash_indices_dense(
                hash_indices, in_values, self.perm_pos, self.perm_ind, self.K, self.L, self.shift_len)
        else:
            wtaHash_cpp.get_hash_indices_sparse(
                hash_indices, in_values, active_in_indices, self.perm_pos, self.perm_ind, self.K, self.L, self.shift_len)
        return hash_indices
