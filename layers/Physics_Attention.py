import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np


def gumbel_softmax(logits, tau, hard=False, dim=-1):
    noise = torch.rand_like(logits)
    noise = -torch.log(-torch.log(noise + 1e-8) + 1e-8)
    y = F.softmax((logits + noise) / tau, dim=dim)
    if hard:
        index = y.max(dim=dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        y = (y_hard - y).detach() + y
    return y


class Physics_Attention_Irregular_Mesh(nn.Module):
    ## for irregular meshes in 1D, 2D or 3D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, shapelist=None,
                 use_local_adaptive_slice=False, local_slice_strength=1.0,
                 physics_mixer='transolver', use_eidetic_slice=False,
                 eidetic_min_temp=0.01, eidetic_gumbel=False, eidetic_hard=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.physics_mixer = physics_mixer
        self.use_eidetic_slice = bool(use_eidetic_slice)
        self.eidetic_min_temp = float(eidetic_min_temp)
        self.eidetic_gumbel = bool(eidetic_gumbel)
        self.eidetic_hard = bool(eidetic_hard)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.use_local_adaptive_slice = bool(use_local_adaptive_slice)
        self.local_slice_strength = nn.Parameter(torch.tensor(float(local_slice_strength)))

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        if self.use_eidetic_slice:
            self.proj_temperature = nn.Sequential(
                nn.Linear(dim_head, slice_num),
                nn.GELU(),
                nn.Linear(slice_num, 1)
            )
        if self.use_local_adaptive_slice:
            self.local_slice_bias = nn.Sequential(
                nn.LayerNorm(dim_head),
                nn.Linear(dim_head, dim_head),
                nn.GELU(),
                nn.Linear(dim_head, slice_num)
            )
            self.context_to_slice = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, inner_dim),
                nn.GELU(),
                nn.Linear(inner_dim, heads * slice_num)
            )
            nn.init.zeros_(self.local_slice_bias[-1].weight)
            nn.init.zeros_(self.local_slice_bias[-1].bias)
            nn.init.zeros_(self.context_to_slice[-1].weight)
            nn.init.zeros_(self.context_to_slice[-1].bias)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        if self.physics_mixer == 'gated_linear':
            self.token_gate = nn.Sequential(
                nn.LayerNorm(dim_head),
                nn.Linear(dim_head, dim_head),
                nn.GELU(),
                nn.Linear(dim_head, dim_head),
                nn.Sigmoid()
            )
            nn.init.zeros_(self.token_gate[-2].weight)
            nn.init.zeros_(self.token_gate[-2].bias)
        elif self.physics_mixer not in ['transolver', 'linearno']:
            raise ValueError(f"Unknown physics_mixer='{self.physics_mixer}'")
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, return_feature=False, vis=False, context=None):
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_logits = self.in_project_slice(x_mid)
        if self.use_local_adaptive_slice:
            local_bias = self.local_slice_bias(x_mid)
            slice_logits = slice_logits + self.local_slice_strength * local_bias
            if context is not None:
                context_bias = self.context_to_slice(context).reshape(B, self.heads, 1, -1)
                slice_logits = slice_logits + self.local_slice_strength * context_bias
        if self.use_eidetic_slice:
            temperature = torch.clamp(self.proj_temperature(x_mid) + self.temperature,
                                      min=self.eidetic_min_temp)
        else:
            temperature = self.temperature

        if self.use_eidetic_slice and self.eidetic_gumbel and self.training:
            slice_weights = gumbel_softmax(slice_logits, temperature, hard=self.eidetic_hard, dim=-1)
        else:
            slice_weights = self.softmax(slice_logits / temperature)  # B H N G
        if vis:
            np.save("slice_weights.npy", slice_weights.detach().cpu().numpy())
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Mixing among physical states
        if self.physics_mixer == 'transolver':
            q_slice_token = self.to_q(slice_token)
            k_slice_token = self.to_k(slice_token)
            v_slice_token = self.to_v(slice_token)
            if hasattr(F, 'scaled_dot_product_attention'):
                out_slice_token = F.scaled_dot_product_attention(
                    q_slice_token, k_slice_token, v_slice_token,
                    dropout_p=self.dropout.p if self.training else 0.0
                )
            else:
                dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
                attn = self.softmax(dots)
                attn = self.dropout(attn)
                out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D
        elif self.physics_mixer == 'linearno':
            out_slice_token = self.to_v(slice_token)
        else:
            value = self.to_v(slice_token)
            gate = self.token_gate(slice_token)
            out_slice_token = gate * value + (1.0 - gate) * slice_token

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        if return_feature:
            return self.to_out(out_x), slice_token
        else:
            return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_1D(nn.Module):
    ## for structured mesh in 1D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, shapelist=None, kernel=3):  # kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.length = shapelist[0]

        self.in_project_x = nn.Conv1d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv1d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, return_feature=False, context=None):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.length, C).contiguous().permute(0, 2, 1).contiguous()  # B C N

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        if return_feature:
            return self.to_out(out_x), slice_token
        else:
            return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_2D(nn.Module):
    ## for structured mesh in 2D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, shapelist=None, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = shapelist[0]
        self.W = shapelist[1]

        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_3D(nn.Module):
    ## for structured mesh in 3D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=32, shapelist=None, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = shapelist[0]
        self.W = shapelist[1]
        self.D = shapelist[2]

        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, self.D, C).contiguous().permute(0, 4, 1, 2, 3).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)
