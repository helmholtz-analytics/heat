import torch.optim.lr_scheduler as lr_scheduler


def __getattr__(name):
    try:
        return lr_scheduler.__getattribute__(name)
    except AttributeError:
        raise AttributeError(f"name {name} is not implemented in torch.optim.lr_scheduler")
