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
    def __init__(self, base_model, upright_class, ang_class):
        super().__init__()
        self.base_model = base_model
        self.up_class = upright_class
        self.ang_class = ang_class

    def freeze_base(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False
        return None

    def unfreeze_base(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False
        return None

    def forward(self, x: Tensor) -> (Tensor, Tensor):

        extracted_up_tensors = dict()
        extracted_ang_tensors = dict()
        for name, mod in self.base_model._modules.items():
            if name == 'fc':
                x = ch.flatten(x, 1)
            x = mod(x)

            if self.up_class is not None:
               if name in self.up_class.extract_list:
                    extracted_up_tensors[name] = x

            if self.ang_class is not None:
                if name in self.ang_class.extract_list:
                    extracted_ang_tensors[name] = x

        out = x
        up_ang = out_ang = None
        if self.up_class is not None:
            up_ang = self.up_class(extracted_up_tensors)
        if self.ang_class is not None:
            out_ang = self.ang_class(extracted_ang_tensors)

        return out, up_ang, out_ang