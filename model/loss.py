import torch

def dummy_loss(data_dict, alpha=None, roi=None, options=()):
    return {'loss': torch.tensor([0])}