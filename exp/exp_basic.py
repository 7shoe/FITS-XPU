import os
import torch
import numpy as np
from utils.device import get_device, xpu_is_available


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and xpu_is_available():
            device = get_device(self.args.gpu)
            print('Rank {} uses XPU: xpu:{}'.format(
                getattr(self.args, 'rank', 0), self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Rank {} uses CPU'.format(getattr(self.args, 'rank', 0)))
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
