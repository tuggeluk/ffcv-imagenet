"""
FFCV transform that randomly rotates images
"""

import torch as ch
from torch import Tensor

class AngleClassifierWrapper(ch.nn.Module):
    """Transform using the given torch.nn.Module

    Parameters
    ----------
    module: torch.nn.Module
        The module for transformation
    """
    def __init__(self, base_model, angle_class):
        super().__init__()
        self.base_model = base_model
        self.angle_class = angle_class

    def freeze_base(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False
        return None

    def unfreeze_base(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False
        return None

    def forward(self, x: Tensor) -> (Tensor, Tensor):

        extracted_tensors = dict()
        for name, mod in self.base_model._modules.items():
            if name == 'fc':
                x = ch.flatten(x, 1)
            x = mod(x)
            if name in self.angle_class.extract_list:
                extracted_tensors[name] = x

        out = x
        out_ang = self.angle_class(extracted_tensors)

        return out, out_ang