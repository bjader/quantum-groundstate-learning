from torch import nn
import torch
from qaml.ml.wanner_2025.util.transforms import Identity, UnitInterval
from qaml.ml.models.geometry import GridMap, HeavyHexGridMap
from typing import Union

class Sin(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.sin(x)

def get_activation(activation_string: str):
    if activation_string == "tanh":
        return nn.Tanh()
    elif activation_string == "relu":
        return nn.ReLU()
    elif activation_string == "gelu":
        return nn.GELU()
    else:
        raise NotImplementedError
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def get_transform(tf_string: str, **tf_args):
    if tf_string=="id":
        return Identity(**tf_args)
    elif tf_string=="unit":
        return UnitInterval(**tf_args)
    else:
        raise NotImplementedError
    
def get_n_terms(mode: str, gm: Union[GridMap, HeavyHexGridMap]):
    if mode == "edges": # terms correspond to edges in lattice
        return gm.m
    else:
        raise NotImplementedError