import torch
from torch import nn
from qaml.ml.models.geometry import GridMap, HeavyHexGridMap
from qaml.ml.wanner_2025.util.helper import get_activation, init_weights, get_n_terms
from torch.func import stack_module_state, functional_call
import copy


def _make_grid_map(geometry_parameters: dict):
    """
    Factory that returns either a GridMap or HeavyHexGridMap depending on
    whether ``geometry_parameters`` contains ``topology="heavy_hex"``.

    The ``topology`` key is consumed here and not forwarded to the underlying
    map constructors (which don't accept it).
    """
    gp = dict(geometry_parameters)  # shallow copy so we don't mutate the caller's dict
    topology = gp.pop("topology", None)

    if topology == "heavy_hex":
        # HeavyHexGridMap accepts: distance, pauli_qubits, delta1, mode,
        #                          use_qiskit, custom_edges, custom_coordinates
        return HeavyHexGridMap(**gp)
    else:
        return GridMap(**gp)

# standard fully connected deep network: [d_input] --> float
class LocalDNN(nn.Module):
    def __init__(self, in_dim, width=10, depth=2, act_fun="tanh", dropout=0.0) -> None:
        super().__init__()
        assert(depth >= 1)

        self.in_dim = in_dim
        self.depth = depth
        self.act_fun = get_activation(act_fun)
        self.width = width
        self.dropout = dropout

        # construct network
        self.layers = [nn.Linear(in_dim, width)]
        for _ in range(depth-1):
            self.layers.append(self.act_fun)
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.Dropout1d(dropout))
            # self.layers.append(nn.LayerNorm((width)))
        self.layers.append(nn.Linear(width, 1))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

# model according to paper: [N.o. parameters] --> float
class SimpleFullDNN(nn.Module):
    def __init__(self,
                 n_terms,
                 geometry_parameters={},
                 local_parameters={}) -> None:
        super().__init__()
        
        self.gm = _make_grid_map(geometry_parameters)
        self.local_map = self.gm.get_layer()
        self.n_terms = n_terms if isinstance(n_terms, int) else get_n_terms(n_terms, self.gm) 
        self.models = nn.ModuleList([LocalDNN(len(loc_ind), **local_parameters) 
                                     for loc_ind in self.local_map.parameter_map])
        self.last_layer = nn.Linear(self.n_terms, 1, bias=False)
    

    def forward(self, x):
        # can be sped up if necessary using vmap
        # use more efficient map in more sophisticated version
        x = self.local_map(x)
        x = torch.cat([model(x_P) for model, x_P in zip(self.models, x)], dim=-1)
        x = self.last_layer(x).flatten()

        # ones_like because of potential batching
        return torch.stack((x, torch.ones_like(x) * self.last_layer.weight.abs().sum()))
    
    def init_xavier(self):
        self.apply(init_weights)


# Stacked SimpleFullDNNs using torch.vmap
# used to predict several observables in parallel
# [N.o. parameters] (Set by n_terms) --> [N.o. observables] (Always n_spins-1 due to the if statement in __init__)
# If gp["mode"] == "nonlocal", the n_terms is equal to number of edges with all-to-all connectivity, i.e. n_spins * (n_spins - 1)/2
class CombinedFullDNN(nn.Module):
    def __init__(self, n_terms, geometry_parameters={}, local_parameters={}, device="cpu") -> None:
        super().__init__()
        self.geometry_parameters = geometry_parameters
        self.local_parameters = local_parameters
        base_model = SimpleFullDNN(n_terms, self.geometry_parameters, self.local_parameters)
        self.n_terms = base_model.n_terms
        self.gm = base_model.gm
        # for performance
        base_model = base_model.to('meta')
        def f_model(params, buffers, x):
            return functional_call(base_model, (params, buffers), (x,))
        self.f_model = f_model

        # adjust number of combined DNNs for adjacent correlations
        if geometry_parameters["mode"] == "nonlocal":
            gp = copy.deepcopy(geometry_parameters)
            gp["mode"] = "local"
            self.gm = GridMap(**gp)
            self.n_terms = self.gm.m
       
        #self.f_model = lambda params, buffers, x: functional_call(base_model, (params, buffers), (x,))                                     
        self.models = nn.ModuleList([SimpleFullDNN(n_terms, geometry_parameters, local_parameters) for _ in range(self.n_terms)]).to(torch.device(device))
        self.params, self.buffs = stack_module_state(self.models)
        # seems a bit hacky, but works
        self._parameters = self.params
        """self.params = nn.ParameterDict(params)
        self.register_buffer('buffs', buffs, persistent=False)"""
  

    def forward(self, x):
        # can be sped up if necessary using vmap/or by making it a Linear layer
        # use more efficient map in more sophisticated version
        """xs = []
        ws = []
        for model in self.models:
            pred, w = model(x)
            xs.append(pred)
            ws.append(w)
        xs = torch.stack(xs, dim=-1)
        ws = torch.stack(ws, dim=-1)"""
        
        res_vmap = torch.vmap(self.f_model, in_dims=(0, 0, None))(self.params, self.buffs, x)
        xs, ws = res_vmap.permute(1, 2, 0)
        #print(torch.allclose(xs_v, xs_v, atol=1e-3, rtol=1e-5), torch.allclose(ws_v, ws_v, atol=1e-3, rtol=1e-5))
        return xs, ws
    
    def init_xavier(self):
        for model in self.models:
            model.init_xavier()
            
###############################################################            
#                 Modification of Legacy Code                 #
###############################################################
# Modified CombinedFullDNN to make the number of observables a free parameter.
# Previously, the number of observables was fixed to the number of nearest neighbour correlation functions.
# Stacked SimpleFullDNNs using torch.vmap
# used to predict several observables in parallel
# [N.o. parameters] (Set by n_terms) --> [N.o. observables] (Set by n_output)
class ModifiedCombinedFullDNN(nn.Module):
    def __init__(self, n_terms, n_outputs, geometry_parameters={}, local_parameters={}, device="cpu") -> None:
        super().__init__()
        self.geometry_parameters = geometry_parameters
        self.local_parameters = local_parameters
        base_model = SimpleFullDNN(n_terms, self.geometry_parameters, self.local_parameters)
        self.n_terms = base_model.n_terms
        self.n_outputs = n_outputs
        self.gm = base_model.gm
        # for performance
        base_model = base_model.to('meta')
        def f_model(params, buffers, x):
            return functional_call(base_model, (params, buffers), (x,))
        self.f_model = f_model
       
        #self.f_model = lambda params, buffers, x: functional_call(base_model, (params, buffers), (x,))                                     
        self.models = nn.ModuleList([SimpleFullDNN(n_terms, geometry_parameters, local_parameters) for _ in range(n_outputs)]).to(torch.device(device))
        self.params, self.buffs = stack_module_state(self.models)
        # seems a bit hacky, but works
        self._parameters = self.params
        """self.params = nn.ParameterDict(params)
        self.register_buffer('buffs', buffs, persistent=False)"""
  

    def forward(self, x):
        # can be sped up if necessary using vmap/or by making it a Linear layer
        # use more efficient map in more sophisticated version
        """xs = []
        ws = []
        for model in self.models:
            pred, w = model(x)
            xs.append(pred)
            ws.append(w)
        xs = torch.stack(xs, dim=-1)
        ws = torch.stack(ws, dim=-1)"""
        
        res_vmap = torch.vmap(self.f_model, in_dims=(0, 0, None))(self.params, self.buffs, x)
        xs, ws = res_vmap.permute(1, 2, 0)
        #print(torch.allclose(xs_v, xs_v, atol=1e-3, rtol=1e-5), torch.allclose(ws_v, ws_v, atol=1e-3, rtol=1e-5))
        return xs, ws
    
    def init_xavier(self):
        for model in self.models:
            model.init_xavier()


class TinyMLP(nn.Module):
    """
    A very small MLP: R -> R for one observable.
    """
    def __init__(self, in_dim=1, width=32, depth=2, act_fun="tanh", dropout=0.0):
        super().__init__()
        act = get_activation(act_fun)
        layers = []
        last = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(last, width))
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            last = width
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [B, 1]
        return self.net(x)  # [B, 1]


class LightweightPerObsDNN(nn.Module):
    """
    One tiny per-observable MLP; input is the scalar jz (extracted from [B, n_spins] input).
    Output: preds [B, n_outputs], ws=None (kept for compatibility).
    """
    def __init__(self,
                 n_outputs: int,
                 width: int = 32,
                 depth: int = 2,
                 act_fun: str = "tanh",
                 dropout: float = 0.0,
                 device: str = "cpu"):
        super().__init__()
        self.n_outputs = n_outputs
        self.models = nn.ModuleList([
            TinyMLP(in_dim=1, width=width, depth=depth, act_fun=act_fun, dropout=dropout)
            for _ in range(n_outputs)
        ]).to(torch.device(device))

    def forward(self, x):
        # x is [B, n_spins] where every entry is the same scalar jz; just take the first feature
        if x.dim() != 2:
            raise ValueError(f"Expected x of shape [B, n_features], got {tuple(x.shape)}")
        jz = x[:, :1]  # [B, 1]

        outs = [m(jz) for m in self.models]  # list of [B,1]
        preds = torch.cat(outs, dim=1)       # [B, n_outputs]

        # Keep return signature compatible with legacy code
        ws = None
        return preds, ws

    

