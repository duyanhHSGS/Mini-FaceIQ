
__author__ = 'cleardusk'

import os
import pickle

import numpy as np
import torch


def mkdir(d):
    os.makedirs(d, exist_ok=True)


def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        with open(fp, 'rb') as file:
            return pickle.load(file)


def _dump(wfp, obj):
    suffix = _get_suffix(wfp)
    if suffix == 'npy':
        np.save(wfp, obj)
    elif suffix == 'pkl':
        with open(wfp, 'wb') as file:
            pickle.dump(obj, file)
    else:
        raise ValueError(f'Unknown Type: {suffix}')


def _load_tensor(fp, mode='cpu'):
    if mode.lower() == 'cpu':
        return torch.from_numpy(_load(fp))
    elif mode.lower() == 'gpu':
        return torch.from_numpy(_load(fp)).cuda()


def _tensor_to_cuda(x):
    if x.is_cuda:
        return x
    else:
        return x.cuda()


def _load_gpu(fp):
    return torch.from_numpy(_load(fp)).cuda()


_load_cpu = _load
_numpy_to_tensor = lambda x: torch.from_numpy(x)
_tensor_to_numpy = lambda x: x.numpy()
_numpy_to_cuda = lambda x: _tensor_to_cuda(torch.from_numpy(x))
_cuda_to_tensor = lambda x: x.cpu()
_cuda_to_numpy = lambda x: x.cpu().numpy()
