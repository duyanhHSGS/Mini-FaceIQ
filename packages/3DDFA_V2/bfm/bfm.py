
__author__ = 'cleardusk'

import sys

sys.path.append('..')

import os.path as osp

import numpy as np
from utils.io import _load

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


class BFMModel:
    def __init__(self, bfm_fp, shape_dim=40, exp_dim=10):
        bfm = _load(bfm_fp)
        if not isinstance(bfm, dict):
            raise TypeError(f'Expected BFM mapping: {bfm_fp}')
        self.u = np.asarray(bfm['u'], dtype=np.float32)  # fix bug
        self.w_shp = np.asarray(bfm['w_shp'], dtype=np.float32)[..., :shape_dim]
        self.w_exp = np.asarray(bfm['w_exp'], dtype=np.float32)[..., :exp_dim]
        if osp.split(bfm_fp)[-1] == 'bfm_noneck_v3.pkl':
            self.tri = np.asarray(_load(make_abs_path('../configs/tri.pkl')))  # this tri/face is re-built for bfm_noneck_v3
        else:
            self.tri = np.asarray(bfm['tri'])

        self.tri = _to_ctype(self.tri.T).astype(np.int32)
        self.keypoints = np.asarray(bfm['keypoints'], dtype=np.int32)  # fix bug
        w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(w, axis=0)

        self.u_base = self.u[self.keypoints].reshape(-1, 1)
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]
