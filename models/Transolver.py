import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
from layers.Physics_Attention import Physics_Attention_Irregular_Mesh
from layers.Physics_Attention import Physics_Attention_Structured_Mesh_1D
from layers.Physics_Attention import Physics_Attention_Structured_Mesh_2D
from layers.Physics_Attention import Physics_Attention_Structured_Mesh_3D
import torch.utils.checkpoint as checkpoint

PHYSICS_ATTENTION = {
    'unstructured': Physics_Attention_Irregular_Mesh,
    'structured_1D': Physics_Attention_Structured_Mesh_1D,
    'structured_2D': Physics_Attention_Structured_Mesh_2D,
    'structured_3D': Physics_Attention_Structured_Mesh_3D
}

ACTIVATION = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU(0.1),
    'softplus': nn.Softplus,
    'ELU': nn.ELU,
    'silu': nn.SiLU
}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class GeometryContextEncoder(nn.Module):
    def __init__(self, space_dim, fun_dim, hidden_dim, context_dim, act='gelu'):
        super().__init__()
        if act in ACTIVATION.keys():
            act_layer = ACTIVATION[act]
        else:
            raise NotImplementedError

        self.space_dim = space_dim
        self.fun_dim = fun_dim
        in_dim = 4 * space_dim + 2 * fun_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, context_dim),
            act_layer(),
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x, fx):
        x_mean = x.mean(dim=1)
        x_std = x.std(dim=1, unbiased=False)
        x_min = x.amin(dim=1)
        x_max = x.amax(dim=1)
        stats = [x_mean, x_std, x_min, x_max]

        if self.fun_dim > 0:
            if fx is None:
                fx_stats = x.new_zeros(x.shape[0], 2 * self.fun_dim)
            else:
                fx_stats = torch.cat([fx.mean(dim=1), fx.std(dim=1, unbiased=False)], dim=-1)
            stats.append(fx_stats)

        return self.net(torch.cat(stats, dim=-1))


class GeometryFiLM(nn.Module):
    def __init__(self, hidden_dim, strength=0.1):
        super().__init__()
        self.strength = nn.Parameter(torch.tensor(float(strength)))
        self.to_gamma_beta = nn.Linear(hidden_dim, hidden_dim * 2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x, context):
        if context is None:
            return x
        gamma, beta = self.to_gamma_beta(context).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return x * (1.0 + self.strength * gamma) + self.strength * beta


class Transolver_block(nn.Module):
    """Transolver encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            geotype='unstructured',
            shapelist=None,
            use_geo_film=False,
            geo_film_strength=0.1,
            use_local_adaptive_slice=False,
            local_slice_strength=1.0,
            physics_mixer='transolver',
            use_eidetic_slice=False,
            eidetic_min_temp=0.01,
            eidetic_gumbel=False,
            eidetic_hard=False
    ):
        super().__init__()
        self.last_layer = last_layer
        self.use_geo_film = bool(use_geo_film)
        self.ln_1 = nn.LayerNorm(hidden_dim)

        attn_kwargs = {}
        if geotype == 'unstructured':
            attn_kwargs.update(dict(use_local_adaptive_slice=use_local_adaptive_slice,
                                    local_slice_strength=local_slice_strength,
                                    physics_mixer=physics_mixer,
                                    use_eidetic_slice=use_eidetic_slice,
                                    eidetic_min_temp=eidetic_min_temp,
                                    eidetic_gumbel=eidetic_gumbel,
                                    eidetic_hard=eidetic_hard))
        self.Attn = PHYSICS_ATTENTION[geotype](hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                               dropout=dropout, slice_num=slice_num, shapelist=shapelist,
                                               **attn_kwargs)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.use_geo_film:
            self.attn_film = GeometryFiLM(hidden_dim, strength=geo_film_strength)
            self.mlp_film = GeometryFiLM(hidden_dim, strength=geo_film_strength)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, geo_context=None):
        attn_out = self.Attn(self.ln_1(fx), context=geo_context)
        if self.use_geo_film:
            attn_out = self.attn_film(attn_out, geo_context)
        fx = attn_out + fx

        mlp_out = self.mlp(self.ln_2(fx))
        if self.use_geo_film:
            mlp_out = self.mlp_film(mlp_out, geo_context)
        fx = mlp_out + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'Transolver'
        self.args = args
        ## embedding
        self.preprocess = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
                              n_layers=0, res=False, act=args.act)
        self.use_geo_film = bool(getattr(args, 'use_geo_film', 0))
        self.use_local_adaptive_slice = bool(getattr(args, 'use_local_adaptive_slice', 0))
        self.use_eidetic_slice = bool(getattr(args, 'use_eidetic_slice', 0))
        if self.use_geo_film or self.use_local_adaptive_slice:
            self.geo_context = GeometryContextEncoder(args.space_dim, args.fun_dim, args.n_hidden,
                                                      getattr(args, 'geo_film_hidden', args.n_hidden),
                                                      act=args.act)
        else:
            self.geo_context = None

        ## models
        self.blocks = nn.ModuleList([Transolver_block(num_heads=args.n_heads, hidden_dim=args.n_hidden,
                                                      dropout=args.dropout,
                                                      act=args.act,
                                                      mlp_ratio=args.mlp_ratio,
                                                      out_dim=args.out_dim,
                                                      slice_num=args.slice_num,
                                                      last_layer=(_ == args.n_layers - 1),
                                                      geotype=args.geotype,
                                                      shapelist=args.shapelist,
                                                      use_geo_film=self.use_geo_film,
                                                      geo_film_strength=getattr(args, 'geo_film_strength', 0.1),
                                                      use_local_adaptive_slice=self.use_local_adaptive_slice,
                                                      local_slice_strength=getattr(args, 'local_slice_strength', 1.0),
                                                      physics_mixer=getattr(args, 'physics_mixer', 'transolver'),
                                                      use_eidetic_slice=self.use_eidetic_slice,
                                                      eidetic_min_temp=getattr(args, 'eidetic_min_temp', 0.01),
                                                      eidetic_gumbel=bool(getattr(args, 'eidetic_gumbel', 0)),
                                                      eidetic_hard=bool(getattr(args, 'eidetic_hard', 0)))
                                     for _ in range(args.n_layers)])
        self.placeholder = nn.Parameter((1 / (args.n_hidden)) * torch.rand(args.n_hidden, dtype=torch.float))
        self.initialize_weights()
        self.reset_experiment_adapters()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def reset_experiment_adapters(self):
        for module in self.modules():
            if isinstance(module, GeometryFiLM):
                module.reset_parameters()
            if hasattr(module, 'local_slice_bias'):
                nn.init.zeros_(module.local_slice_bias[-1].weight)
                nn.init.zeros_(module.local_slice_bias[-1].bias)
            if hasattr(module, 'context_to_slice'):
                nn.init.zeros_(module.context_to_slice[-1].weight)
                nn.init.zeros_(module.context_to_slice[-1].bias)
            if hasattr(module, 'proj_temperature'):
                nn.init.zeros_(module.proj_temperature[-1].weight)
                nn.init.zeros_(module.proj_temperature[-1].bias)
            if hasattr(module, 'token_gate'):
                nn.init.zeros_(module.token_gate[-2].weight)
                nn.init.zeros_(module.token_gate[-2].bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def structured_geo(self, x, fx):
        if self.args.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1)
        geo_context = self.geo_context(x, fx) if self.geo_context is not None else None
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            if self.args.checkpoint:
                if geo_context is None:
                    fx = checkpoint.checkpoint(block, fx)
                else:
                    fx = checkpoint.checkpoint(block, fx, geo_context)
            else:
                fx = block(fx, geo_context)
        return fx

    def unstructured_geo(self, x, fx):
        geo_context = self.geo_context(x, fx) if self.geo_context is not None else None
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            if self.args.checkpoint:
                if geo_context is None:
                    fx = checkpoint.checkpoint(block, fx)
                else:
                    fx = checkpoint.checkpoint(block, fx, geo_context)
            else:
                fx = block(fx, geo_context)
        return fx

    def forward(self, x, fx):
        if self.args.geotype == 'unstructured':
            return self.unstructured_geo(x, fx)
        else:
            return self.structured_geo(x, fx)
