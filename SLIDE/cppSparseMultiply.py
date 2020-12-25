import torch
import torch.nn as nn
import cppSparseMultiply_cpp

class cppSparseMultiply(torch.autograd.Function):

    @staticmethod
    def forward(ctx, in_values, weights, bias, active_in_indices=None, active_out_indices=None):
        ctx.save_for_backward(in_values, weights, bias, active_in_indices, active_out_indices)
        
        if active_in_indices is not None and active_out_indices is not None:
            out_values = cppSparseMultiply_cpp.siso(in_values, active_out_indices, active_in_indices, weights, bias)
        elif active_out_indices is not None:
            out_values = cppSparseMultiply_cpp.diso(in_values, active_out_indices, weights, bias)
        elif active_in_indices is not None:
            out_values = cppSparseMultiply_cpp.sido(in_values, active_in_indices, weights, bias)
        else:
            # out_values = cppSparseMultiply_cpp.dido_naive(in_values, weights, bias)
            out_values = in_values.mm(weights.t())
            out_values += bias.unsqueeze(0).expand_as(out_values)
        return out_values

    @staticmethod
    def backward(ctx, grad_out_values):
        in_values, weights, bias, active_in_indices, active_out_indices = ctx.saved_tensors
        
        # try incorporating ctx.needs_input_grad to avoid unnecessary grad computation
        if active_in_indices is not None and active_out_indices is not None:
            grad_in_values, grad_weights, grad_bias = cppSparseMultiply_cpp.siso_backward(
                grad_out_values, in_values, active_out_indices, active_in_indices, weights, bias)

        elif active_out_indices is not None:
            grad_in_values, grad_weights, grad_bias = cppSparseMultiply_cpp.diso_backward(
                grad_out_values, in_values, active_out_indices, weights, bias)

        elif active_in_indices is not None:
            grad_in_values, grad_weights, grad_bias = cppSparseMultiply_cpp.sido_backward(
                grad_out_values, in_values, active_in_indices, weights, bias)

        else:
            grad_in_values = grad_out_values.mm(weights)
            # grad_in_values = cppSparseMultiply_cpp.dido_naive(
                # grad_out_values, weights.t(), torch.zeros(weights.size(0)))
            grad_weights = grad_out_values.t().mm(in_values)
            # grad_weights = cppSparseMultiply_cpp.dido_naive(
                # grad_out_values.t(), in_values.t(), torch.zeros(in_values.size(1)))
            grad_bias = grad_out_values.sum(0)
            
        return grad_in_values, grad_weights, grad_bias, None, None
