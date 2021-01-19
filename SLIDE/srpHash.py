import math
import torch
import torch.nn.functional as F
import srpHash_cpp


class srpHashTable():
    def __init__(self, K, L, dim):
        self.K = K
        self.L = L
        self.dim = dim
        self.reset_hashes()

    def reset_hashes(self):
        # Has minimal overhead compared to hashing and bucketing latency
        self.hash_vecs = torch.rand([self.L*self.K, self.dim], dtype=torch.get_default_dtype()).round() * 2 -1
        # self.hash_vecs = torch.rand([self.L*self.K, self.dim], dtype=torch.get_default_dtype()) - 0.5
        # self.hash_vecs = torch.randn([self.L*self.K, self.dim], dtype=torch.get_default_dtype())
        # self.hash_vecs = F.normalize(torch.randn([self.L*self.K, self.dim], dtype=torch.get_default_dtype()))

    def get_hash_indices(self, in_values, active_in_indices=None):
        hash_indices = torch.empty([in_values.size(0), self.L], dtype=torch.int32)
        if active_in_indices is None:
            with torch.no_grad():
                hash_values = torch.mm(in_values, self.hash_vecs.t())
                hash_values = hash_values.reshape(hash_values.size(0), self.L, self.K)
            srpHash_cpp.get_hash_indices_from_values(hash_indices, hash_values)
        else:
            srpHash_cpp.get_hash_indices_sparse(hash_indices,
                                                in_values,
                                                active_in_indices,
                                                self.hash_vecs.reshape(self.L, self.K, self.dim))
        return hash_indices
