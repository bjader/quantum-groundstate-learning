"""Vectorized model: replaces the 196-iteration Python loop in SimpleFullDNN with
batched 3D weight tensors and a single einsum per layer.

The key change vs model.py:
- Old: nn.ModuleList of n_terms LocalDNNs with varying in_dim, iterated in a Python loop
- New: single set of [n_terms, in, out] weight tensors, forward is pure tensor ops

ModifiedCombinedFullDNN is identical to model.py except it uses the new SimpleFullDNN.
"""
import logging
import math
import copy

import torch
from torch import nn
from torch.func import stack_module_state, functional_call

logger = logging.getLogger(__name__)

from qaml.ml.models.geometry import GridMap, HeavyHexGridMap
from qaml.ml.wanner_2025.util.helper import get_activation, init_weights, get_n_terms


def _make_grid_map(geometry_parameters: dict):
    gp = dict(geometry_parameters)
    topology = gp.pop("topology", None)
    if topology == "heavy_hex":
        return HeavyHexGridMap(**gp)
    else:
        return GridMap(**gp)


class SimpleFullDNN(nn.Module):
    """Vectorized version: all n_terms local DNNs computed in one batched pass."""

    def __init__(self, n_terms, geometry_parameters={}, local_parameters={}) -> None:
        super().__init__()

        self.gm = _make_grid_map(geometry_parameters)
        self.local_map = self.gm.get_layer()
        self.n_terms = n_terms if isinstance(n_terms, int) else get_n_terms(n_terms, self.gm)

        width  = local_parameters.get("width", 10)
        depth  = local_parameters.get("depth", 2)
        act_fn = local_parameters.get("act_fun", "tanh")
        self.dropout_p = local_parameters.get("dropout", 0.0)
        self.act_fun = get_activation(act_fn)

        # Pad every parameter map to the same length so we can index in one shot.
        raw_pms = self.local_map.parameter_map          # list of CPU LongTensors
        max_pm  = max(len(pm) for pm in raw_pms)
        pm_cpu  = torch.zeros(self.n_terms, max_pm, dtype=torch.long)
        for i, pm in enumerate(raw_pms):
            pm_cpu[i, :len(pm)] = pm
            if len(pm) < max_pm:
                pm_cpu[i, len(pm):] = pm[-1]            # repeat last index to pad

        # Store as plain Python attribute (NOT a buffer) so base_model.to('meta')
        # does not clobber it, and functional_call does not need to substitute it.
        self._padded_pm     = pm_cpu
        self._pm_cache      = None
        self._pm_device     = None

        # One set of 3D weight tensors replaces n_terms separate nn.Linear layers.
        self.w_in     = nn.Parameter(torch.empty(self.n_terms, max_pm, width))
        self.b_in     = nn.Parameter(torch.zeros(self.n_terms, width))
        self.w_hidden = nn.ParameterList([
            nn.Parameter(torch.empty(self.n_terms, width, width)) for _ in range(depth - 1)
        ])
        self.b_hidden = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_terms, width)) for _ in range(depth - 1)
        ])
        self.w_out     = nn.Parameter(torch.empty(self.n_terms, width, 1))
        self.last_layer = nn.Linear(self.n_terms, 1, bias=False)

        logger.info(f"SimpleFullDNN: n_terms={self.n_terms}, max_pm={max_pm}, width={width}, depth={depth}")
        self._init_weights(max_pm, width)

    def _init_weights(self, fan_in, fan_hidden):
        std = math.sqrt(2.0 / fan_in)
        self.w_in.data.uniform_(-std, std)
        std_h = math.sqrt(2.0 / fan_hidden)
        for wh in self.w_hidden:
            wh.data.uniform_(-std_h, std_h)
        self.w_out.data.uniform_(-std_h, std_h)
        nn.init.xavier_uniform_(self.last_layer.weight)

    def forward(self, x):
        # Move padded index map to the same device as x (once, then cached).
        device = x.device
        if self._pm_device != device:
            self._pm_cache  = self._padded_pm.to(device)
            self._pm_device = device

        # Single gather: [batch, n_terms, max_pm]
        x_local = x[:, self._pm_cache]

        # Batched first layer: [B, T, I] x [T, I, W] -> [B, T, W]
        h = torch.einsum("bti,tiw->btw", x_local, self.w_in) + self.b_in
        h = self.act_fun(h)
        if self.training and self.dropout_p > 0:
            h = nn.functional.dropout(h, p=self.dropout_p)

        # Hidden layers: [B, T, W] x [T, W, W] -> [B, T, W]
        for w, b in zip(self.w_hidden, self.b_hidden):
            h = torch.einsum("btw,twv->btv", h, w) + b
            h = self.act_fun(h)
            if self.training and self.dropout_p > 0:
                h = nn.functional.dropout(h, p=self.dropout_p)

        # Output layer: [B, T, W] x [T, W, 1] -> [B, T]
        h = torch.einsum("btw,twv->btv", h, self.w_out).squeeze(-1)

        x = self.last_layer(h).flatten()
        return torch.stack((x, torch.ones_like(x) * self.last_layer.weight.abs().sum()))

    def init_xavier(self):
        self._init_weights(self.w_in.shape[1], self.w_in.shape[2])


class ModifiedCombinedFullDNNFast(nn.Module):
    """Fast vectorized version of ModifiedCombinedFullDNN using padded 3D tensors.
    
    This is architecturally incompatible with the original ModifiedCombinedFullDNN
    in model.py due to weight padding. Models trained with one cannot be evaluated
    with the other.
    """

    def __init__(self, n_terms, n_outputs, geometry_parameters={}, local_parameters={}, device="cpu") -> None:
        super().__init__()
        self.geometry_parameters = geometry_parameters
        self.local_parameters    = local_parameters

        base_model    = SimpleFullDNN(n_terms, self.geometry_parameters, self.local_parameters)
        self.n_terms  = base_model.n_terms
        self.n_outputs = n_outputs
        self.gm       = base_model.gm

        base_model = base_model.to("meta")

        def f_model(params, buffers, x):
            return functional_call(base_model, (params, buffers), (x,))
        self.f_model = f_model

        self.models = nn.ModuleList([
            SimpleFullDNN(n_terms, geometry_parameters, local_parameters)
            for _ in range(n_outputs)
        ]).to(torch.device(device))

        self.params, self.buffs = stack_module_state(self.models)
        self._parameters = self.params

    def forward(self, x):
        res_vmap = torch.vmap(self.f_model, in_dims=(0, 0, None))(self.params, self.buffs, x)
        xs, ws = res_vmap.permute(1, 2, 0)
        return xs, ws

    def init_xavier(self):
        for model in self.models:
            model.init_xavier()
