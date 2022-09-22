"""Utility file for CUDA and GPU-specific functions."""


import torch
import torch.backends.cudnn as cudnn


def setup_gpus(gpu_ids):
    """Set up the GPUs and return the device to be used.

    Args:
        gpu_ids (list): list of GPU IDs

    Returns:
        device (str): the device, either 'cuda' or 'cpu'

    """
    device = None
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        for i in range(len(gpu_ids)):
            torch.cuda.set_device(gpu_ids[i])
        cudnn.benchmark = True
        if len(gpu_ids) > 1:
            device = f'cuda:{gpu_ids[0]}'
        else:
            device='cuda'
    else:
        device = 'cpu'

    return device
