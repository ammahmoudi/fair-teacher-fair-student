import torch


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def smart_load_state_dict(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Detect if keys have 'module.' prefix
    has_module_prefix = all(k.startswith('module.') for k in state_dict.keys())
    
    # If necessary, strip 'module.' prefix
    if has_module_prefix:
        new_state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
