"""Optimizer
@author: FlyEgle
@datetime: 2022-01-20
"""
from torch.optim import SGD, Adam, AdamW


class BuildOptim:
    def __init__(self, optim_name, lr, weight_decay, momentum, betas=(0.9, 0.999), eps=1e-8):
        _names_ = ['sgd', 'adam', 'adamw']
        if not optim_name.lower() in _names_:
            raise NotImplementedError(f"{optim_name} have not been implemented")
        self.optim_name = optim_name
        self.lr = lr
        self.betas = betas  
        self.eps  = eps 
        self.weight_decay = weight_decay 
        self.momentum = momentum

    def _sgd(self, params):
        return SGD(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum
        )

    def _adam(self, params):
        return Adam(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
        )

    def _adamw(self, params):
        return AdamW(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
        )
    
    def __call__(self, *args):
        if self.optim_name.lower() == "sgd":
            return self._sgd(*args)
        elif self.optim_name.lower() == "adam":
            return self._adam(*args)
        elif self.optim_name.lower() == "adamw":
            return self._adamw(*args)
        else:
            raise NotImplementedError(f"{self.optim_name} have not been implementation")