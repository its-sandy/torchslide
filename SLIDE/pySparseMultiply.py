import torch
import torch.nn as nn
import concurrent.futures

def dido(in_values, weights, bias):
    # dense input dense output
    out_values = in_values.mm(weights.t()) + bias.unsqueeze(0).expand_as(out_values)
    return out_values

def diso(in_values, active_out_indices, weights, bias):
    # dense input sparse output
    # in_values = (batch x in_dim)
    # active_out_indices = (batch x active_out_dim)
    # out_values = (batch x active_out_dim)

    active_weights = weights[[active_out_indices]]
    active_biases = bias[[active_out_indices]]
    out_values = torch.bmm(active_weights, in_values.unsqueeze(-1)).squeeze(-1) + active_biases
    return out_values

def sido(in_values, active_in_indices, weights, bias):
    # sparse input dense output
    # in_values = (batch x active_in_dim)
    # active_in_indices = (batch x active_in_dim)
    # out_values = (batch x out_dim)

    active_biases = bias.expand(active_in_indices.size(0), -1)
    active_weight_rows = weights.expand(active_in_indices.size(0), -1, -1)
    expanded_in_indices = active_in_indices.unsqueeze(1).expand(active_in_indices.size(0), active_weight_rows.size(1), active_in_indices.size(1))
    active_weights = torch.gather(active_weight_rows, 2, expanded_in_indices)
    out_values = torch.bmm(active_weights, in_values.unsqueeze(-1)).squeeze(-1) + active_biases
    return out_values

def siso(in_values, active_out_indices, active_in_indices, weights, bias):
    # sparse input sparse output
    # in_values = (batch x active_in_dim)
    # active_in_indices = (batch x active_in_dim)
    # active_out_indices = (batch x active_out_dim)
    # out_values = (batch x active_out_dim)

    active_biases = bias[[active_out_indices]]
    active_weight_rows = weights[[active_out_indices]]
    expanded_in_indices = active_in_indices.unsqueeze(1).expand(active_in_indices.size(0), active_weight_rows.size(1), active_in_indices.size(1))
    active_weights = torch.gather(active_weight_rows, 2, expanded_in_indices)
    out_values = torch.bmm(active_weights, in_values.unsqueeze(-1)).squeeze(-1) + active_biases
    return out_values

def threaded_diso(in_values, active_out_indices, weights, bias):
    # This attempts a parallel implementation of sparse multiplies in python, 
    # that also avoids data copies. Nevertheless, was extremely inefficient. 
    # in_values = (batch x in_dim)
    # active_out_indices = (batch x active_out_dim)
    # out_values = (batch x active_out_dim)

    batch_size = active_out_indices.shape[0]

    def thread_fun(sample_num):
        torch.set_num_threads = 1
        cur_out_values= torch.stack([torch.dot(in_values[sample_num], weights[i]) + bias[i] for i in active_out_indices[sample_num]])
        out_values[sample_num] = cur_out_values

    # out_values = torch.empty(active_out_indices.shape, dtype=in_values.dtype, requires_grad=True)
    out_values = [0]*batch_size
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(64,batch_size)) as executor:
        executor.map(thread_fun, range(batch_size))
    # return torch.cat(out_values, dim=1)
    return out_values


def pySparseMultiply(in_values, weights, bias, active_in_indices=None, active_out_indices=None):
    # the implementations in this module perform unnecessary data copy 
    # due to the advanced indexing required. This is not an issue in the
    # corresponding c++ implementations used in cppSparseMultiply.py

    if active_in_indices is not None and active_out_indices is not None:
        return siso(in_values, active_out_indices, active_in_indices, weights, bias)
    elif active_out_indices is not None:
        return diso(in_values, active_out_indices, weights, bias)
    elif active_in_indices is not None:
        return sido(in_values, active_in_indices, weights, bias)
    else:
        return dido(in_values, weights, bias)
