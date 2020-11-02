import torch
import torch.nn as nn
import torch.multiprocessing as mp
import threading
import concurrent.futures

class pySlideLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(pySlideLayer, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)

        # the implementations in this module perform unnecessary data copy 
        # due to the advanced indexing required. This is not an issue in the
        # corresponding c++ implementations used in cppslide.py

    def forward(self, in_values, active_in_indices=None, active_out_indices=None):
        if active_in_indices is not None and active_out_indices is not None:
            return self.siso(in_values, active_in_indices, active_out_indices)
        elif active_out_indices is not None:
            return self.diso(in_values, active_out_indices)
        elif active_in_indices is not None:
            return self.sido(in_values, active_in_indices)
        else:
            return self.dido(in_values)

    def dido(self, inputs):
        # dense input dense output
        return self.linear(inputs)

    def diso(self, in_values, active_out_indices):
        # dense input sparse output
        # in_values = (batch x in_dim)
        # active_out_indices = (batch x active_out_dim)
        # output_values = (batch x active_out_dim)

        active_weights = self.linear.weight[[active_out_indices]]
        active_biases = self.linear.bias[[active_out_indices]]
        output_values = torch.bmm(active_weights, in_values.unsqueeze(-1)).squeeze(-1) + active_biases
        return output_values
    
    def sido(self, in_values, active_in_indices):
        # sparse input dense output
        # in_values = (batch x active_in_dim)
        # active_in_indices = (batch x active_in_dim)
        # output_values = (batch x out_dim)

        active_biases = self.linear.bias.expand(active_in_indices.size(0), -1)
        active_weight_rows = self.linear.weight.expand(active_in_indices.size(0), -1, -1)
        expanded_in_indices = active_in_indices.unsqueeze(1).expand(active_in_indices.size(0), active_weight_rows.size(1), active_in_indices.size(1))
        active_weights = torch.gather(active_weight_rows, 2, expanded_in_indices)
        output_values = torch.bmm(active_weights, in_values.unsqueeze(-1)).squeeze(-1) + active_biases
        return output_values

    def siso(self, in_values, active_in_indices, active_out_indices):
        # sparse input sparse output
        # in_values = (batch x active_in_dim)
        # active_in_indices = (batch x active_in_dim)
        # active_out_indices = (batch x active_out_dim)
        # output_values = (batch x active_out_dim)

        active_biases = self.linear.bias[[active_out_indices]]
        active_weight_rows = self.linear.weight[[active_out_indices]]
        expanded_in_indices = active_in_indices.unsqueeze(1).expand(active_in_indices.size(0), active_weight_rows.size(1), active_in_indices.size(1))
        active_weights = torch.gather(active_weight_rows, 2, expanded_in_indices)
        output_values = torch.bmm(active_weights, in_values.unsqueeze(-1)).squeeze(-1) + active_biases
        return output_values

    def threaded_diso(self, in_values, active_out_indices):
        # This attempt at a parallel implementation of slide multiplies in python, 
        # that also avoids data copies, was extremely inefficient. 
        # in_values = (batch x in_dim)
        # active_out_indices = (batch x active_out_dim)
        # output_values = (batch x active_out_dim)

        batch_size = active_out_indices.shape[0]

        def thread_fun(sample_num):
            torch.set_num_threads = 1
            cur_output_values= torch.stack([torch.dot(in_values[sample_num], self.linear.weight[i]) + self.linear.bias[i] for i in active_out_indices[sample_num]])
            output_values[sample_num] = cur_output_values

        # output_values = torch.empty(active_out_indices.shape, dtype=in_values.dtype, requires_grad=True)
        output_values = [0]*batch_size
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(64,batch_size)) as executor:
            executor.map(thread_fun, range(batch_size))
        # return torch.cat(output_values, dim=1)
        return output_values
