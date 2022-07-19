"""
FFCV transform that randomly rotates images
"""

import torch as ch
from torch import Tensor
from torchvision.models.efficientnet import EfficientNet
from collections import OrderedDict
from torch.nn.modules.batchnorm import BatchNorm2d

class AngleClassifierWrapper(ch.nn.Module):
    """Transform using the given torch.nn.Module

    Parameters
    ----------
    module: torch.nn.Module
        The module for transformation
    """
    def __init__(self, base_model, upright_class, ang_class):
        super().__init__()
        self.EfficientNetMode = False
        if isinstance(base_model, EfficientNet):
            self.EfficientNetMode = True
        self.base_model = base_model
        self.up_class = upright_class
        self.ang_class = ang_class

        self.forward_modules = self.base_model._modules
        if self.EfficientNetMode:
            self.unwrap_features()


    def unwrap_features(self):
        new_forward_modules = OrderedDict()
        for key, mod in self.forward_modules['features']._modules.items():
            new_forward_modules[key] = mod

        new_forward_modules['avgpool'] = self.forward_modules['avgpool']
        new_forward_modules['classifier'] = self.forward_modules['classifier']
        self.forward_modules = new_forward_modules

    def freeze_base(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.freeze_bn(self.base_model)
        return None

    def freeze_bn(self, module):
        if isinstance(module, BatchNorm2d):
            module.eval()
            print(module)
        for child in module.children():
            self.freeze_bn(child)

    def unfreeze_base(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False
        return None

    def forward(self, x: Tensor) -> (Tensor, Tensor):

        extracted_up_tensors = dict()
        extracted_ang_tensors = dict()
        for name, mod in self.forward_modules.items():
            if name == 'fc' or name == 'classifier':
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