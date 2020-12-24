import torch

def softmax_cross_entropy_with_logits(logits, label_counts):
    # The first label_counts[i] logits in the ith example corresponds to the true labels
    
    # This implementation may be made slightly faster using a c++ extension to
    # perform the aggregation part in forward and backprop. Check if it is worth it.
    # (refer experiments branch for initial implementation)
    # refer https://gombru.github.io/2018/05/23/cross_entropy_loss/
    batch_size = logits.size(0)
    sample_size = logits.size(1)
    label_mask = torch.arange(sample_size).expand(batch_size,-1) < label_counts.unsqueeze(-1)
    return -(((torch.nn.LogSoftmax(dim=1)(logits) * label_mask).sum(1) / label_counts).mean())

def binary_cross_entropy_with_logits(logits, label_counts):
    # The first label_counts[i] logits in the ith example corresponds to the true labels
    batch_size = logits.size(0)
    sample_size = logits.size(1)
    targets = (torch.arange(sample_size).expand(batch_size,-1) < label_counts.unsqueeze(-1)).float()
    return torch.nn.BCEWithLogitsLoss()(logits, targets)
