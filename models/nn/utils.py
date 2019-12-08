import torch.nn as nn


class CloneMixin:
    def clone(self):
        clone = self.__class__(*self.__args__)
        clone.load_state_dict(self.state_dict())
        return clone


def get_param_update_fn(src_model, dest_model):
    def update_params(update_factor=1.0):
        nonlocal src_model, dest_model
        for src_param, dest_param in zip(src_model.parameters(), dest_model.parameters()):
            updated_param = src_param.data * update_factor + dest_param * (1 - update_factor)
            dest_param.data.copy_(updated_param)

    return update_params


def folded_init(model):
    for name, tensor in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_normal_(tensor)
            tensor.data = tensor.abs()


def xavier_init(model):
    for name, tensor in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_normal_(tensor)
