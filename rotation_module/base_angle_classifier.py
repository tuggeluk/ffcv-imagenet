import torch as ch

class BaseAngleClassifier(ch.nn.Module):

    def _get_recursive_last_size(self, module):
        if len(module._modules) > 0:

            candidate = module._modules[list(module._modules.keys())[-1]]
            ret = self._get_recursive_last_size(candidate)

            if ret > 0:
                return ret
            else:
                for k, m in reversed(module._modules.items()):
                    if hasattr(m, "num_features"):
                        return m.num_features
                    if hasattr(m, "_modules") and len(m._modules)>0:
                        return self._get_recursive_last_size(m._modules[list(m._modules.keys())[-1]])
        else:
            return -1