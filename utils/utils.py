import re
import torch


def LoadCheckpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = {}
    # Modify the state dict since the checkpoint might be from pytorch lightning, which adds a 'model'. prefix
    for k, v in checkpoint['state_dict'].items():
        param_path = k.split('.')
        if param_path[0] == 'model':
            new_state_dict['.'.join(param_path[1:])] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    checkpoint['state_dict'] = new_state_dict
    return checkpoint


def GetLayerNumberWithPattern(param_name, pattern):
    m = re.match(pattern, param_name)
    if m:
        return int(m.groups()[0])
    else:
        return -1