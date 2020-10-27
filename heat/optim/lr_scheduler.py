import sys
import torch.optim.lr_scheduler as lrs

if sys.version_info.minor >= 7:

    def __getattr__(name):
        try:
            return lrs.__getattribute__(name)
        except AttributeError:
            raise AttributeError(f"name {name} is not implemented in torch.optim.lr_scheduler")


else:
    LambdaLR = lrs.LambdaLR
    MultiplicativeLR = lrs.MultiplicativeLR
    StepLR = lrs.StepLR
    MultiStepLR = lrs.MultiStepLR
    ExponentialLR = lrs.ExponentialLR
    CosineAnnealingLR = lrs.CosineAnnealingLR
    ReduceLROnPlateau = lrs.ReduceLROnPlateau
    CyclicLR = lrs.CyclicLR
    CosineAnnealingWarmRestarts = lrs.CosineAnnealingWarmRestarts
    OneCycleLR = lrs.OneCycleLR
