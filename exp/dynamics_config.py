from __future__ import annotations
import math
from typing import Callable, Dict, Type
import torch
import torch.nn as nn
from layers.Downstream_Adapters import TokenNonLocalBlock

# ==========================================
# 🚀 基础组件：融合 PointNet 思想的 Advanced SEBlock
# ==========================================
class AdvancedSEBlock(nn.Module):
    def __init__(self, channel, cond_dim=3, reduction=4):
        super(AdvancedSEBlock, self).__init__()
        se_in_dim = channel * 2 + cond_dim
        self.fc = nn.Sequential(
            nn.Linear(se_in_dim, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, cond):
        x_mean = x.mean(dim=1, keepdim=True)
        x_max = x.max(dim=1, keepdim=True)[0]
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        global_desc = torch.cat([x_mean, x_max, cond], dim=-1)
        y = self.fc(global_desc)

        self.saved_attention = y.detach()

        return x * y


# ==========================================
# 🚀 基础组件：通用的动态提示词微扰预测器
# ==========================================
class GenericLocalDynamicsMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, cond_dim=3, use_nonlocal=False,
                 nonlocal_context=128, nonlocal_reduction=2):
        super().__init__()
        self.use_nonlocal = bool(use_nonlocal)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()

        self.se = AdvancedSEBlock(channel=hidden_dim, cond_dim=cond_dim, reduction=4)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        if self.use_nonlocal:
            self.nonlocal_mixer = TokenNonLocalBlock(
                hidden_dim=hidden_dim,
                reduction=nonlocal_reduction,
                context_size=nonlocal_context,
            )

        # 核心：输出维度不再硬编码为 3，而是和不同任务的 cond_dim 保持一致
        self.out = nn.Linear(hidden_dim, cond_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, cond):
        B, N, _ = x.shape
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)

        cond_expanded = cond.expand(-1, N, -1)
        features = torch.cat([x, cond_expanded], dim=-1)

        h = self.fc1(features)
        h = self.norm1(h)
        h = self.act1(h)

        h = self.se(h, cond)

        h = self.fc2(h)
        h = self.norm2(h)
        h = self.act2(h)
        if self.use_nonlocal:
            h = self.nonlocal_mixer(h)

        deltas = self.out(h)  # [B, N, cond_dim]
        return deltas


# ================================================================
# 🚀 任务特化模块 (封装为 nn.Module)
# ================================================================

class CraftDynamics(nn.Module):
    def __init__(self, use_nonlocal=False, nonlocal_context=128, nonlocal_reduction=2):
        super().__init__()
        # x: 7维, cond: 3维 -> in_dim = 10
        self.mlp = GenericLocalDynamicsMLP(
            in_dim=10, hidden_dim=64, cond_dim=3,
            use_nonlocal=use_nonlocal,
            nonlocal_context=nonlocal_context,
            nonlocal_reduction=nonlocal_reduction,
        )

        # ✅ 把 1/3.0 变成可学习参数（初始值就是 1/3.0，和原来一致）
        self.mach_scale = nn.Parameter(torch.tensor(1.0 / 3.0))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        device, dtype = x.device, x.dtype
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)

        # 1. 获取微扰
        deltas = self.mlp(x, cond)

        # 2. 局部物理量 = 全局 Baseline + 微扰
        cond_exp = cond.expand(-1, n, -1)
        local_mach = cond_exp[:, :, 0:1] + deltas[:, :, 0:1]
        local_aoa = cond_exp[:, :, 1:2] + deltas[:, :, 1:2]
        local_beta = cond_exp[:, :, 2:3] + deltas[:, :, 2:3]

        # 3. 物理公式转换
        vx = torch.cos(math.pi * local_aoa / 180.0) * torch.cos(math.pi * local_beta / 180.0)
        vy = torch.sin(math.pi * local_aoa / 180.0)
        vz = torch.cos(math.pi * local_aoa / 180.0) * torch.sin(math.pi * local_beta / 180.0)

        v = torch.cat([vx, vy, vz], dim=-1).to(device=device, dtype=dtype)

        # ✅ 使用可学习参数代替固定 1/3.0
        extra = (local_mach * self.mach_scale).to(device=device, dtype=dtype)

        return torch.cat([v, extra], dim=-1)

class NasaDynamics(nn.Module):
    def __init__(self, use_nonlocal=False, nonlocal_context=128, nonlocal_reduction=2):
        super().__init__()
        # x: 7维, cond: 2维 (mach, aoa) -> in_dim = 9
        self.mlp = GenericLocalDynamicsMLP(
            in_dim=9, hidden_dim=64, cond_dim=2,
            use_nonlocal=use_nonlocal,
            nonlocal_context=nonlocal_context,
            nonlocal_reduction=nonlocal_reduction,
        )

        # 可学习缩放系数，初始值 1.6
        self.mach_scale = nn.Parameter(torch.tensor(1.6))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        device, dtype = x.device, x.dtype
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)

        deltas = self.mlp(x, cond)
        cond_exp = cond.expand(-1, n, -1)

        local_mach = cond_exp[:, :, 0:1] + deltas[:, :, 0:1]
        local_aoa = cond_exp[:, :, 1:2] + deltas[:, :, 1:2]

        vx = torch.cos(math.pi * local_aoa / 180.0)
        vy = torch.sin(math.pi * local_aoa / 180.0)
        vz = torch.zeros(b, n, 1, device=device, dtype=dtype)

        v = torch.cat([vx, vy, vz], dim=-1)
        # 用可学习系数代替固定 1.6
        extra = (local_mach * self.mach_scale).to(dtype)
        return torch.cat([v, extra], dim=-1)


class CrashDynamics(nn.Module):
    def __init__(self, use_nonlocal=False, nonlocal_context=128, nonlocal_reduction=2):
        super().__init__()
        # x: 7维, cond: 1维 (angle) -> in_dim = 8
        self.mlp = GenericLocalDynamicsMLP(
            in_dim=8, hidden_dim=64, cond_dim=1,
            use_nonlocal=use_nonlocal,
            nonlocal_context=nonlocal_context,
            nonlocal_reduction=nonlocal_reduction,
        )

        # 🌟 核心修改 1：将硬编码的 0.5 注册为全局可学习参数
        # 初始值设为 0.5 以保证与你之前的 Baseline 热启动状态完全一致
        self.learnable_base_speed = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        device, dtype = x.device, x.dtype
        if cond.dim() == 2: cond = cond.unsqueeze(1)

        deltas = self.mlp(x, cond)
        cond_exp = cond.expand(-1, n, -1)

        local_angle = cond_exp[..., :1] + deltas[..., :1]

        vx = torch.cos(math.pi * local_angle / 180.0)
        vy = torch.zeros(b, n, 1, device=device, dtype=dtype)
        vz = torch.sin(math.pi * local_angle / 180.0)
        v = torch.cat([vx, vy, vz], dim=-1).to(dtype)

        x_max = torch.max(x[:, :, 0:1], dim=1, keepdim=True)[0]
        x_min = torch.min(x[:, :, 0:1], dim=1, keepdim=True)[0]
        speed = (x[:, :, 0:1] - x_min) / (x_max - x_min + 1e-8)

        # 🌟 核心修改 2：用 self.learnable_base_speed 替换掉硬编码的 0.5
        extra = (speed * self.learnable_base_speed).to(dtype)

        return torch.cat([v, extra], dim=-1)



# class HullDynamics(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # x: 7维, cond: 1维 (angle) -> in_dim = 8
#         self.mlp = GenericLocalDynamicsMLP(in_dim=8, hidden_dim=64, cond_dim=1)
#
#     def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
#         b, n, _ = x.shape
#         device, dtype = x.device, x.dtype
#         if cond.dim() == 2: cond = cond.unsqueeze(1)
#
#         deltas = self.mlp(x, cond)
#         cond_exp = cond.expand(-1, n, -1)
#
#         local_angle = cond_exp[..., :1] + deltas[..., :1]
#
#         vx = torch.cos(math.pi * local_angle / 180.0)
#         vy = torch.zeros(b, n, 1, device=device, dtype=dtype)
#         vz = torch.sin(math.pi * local_angle / 180.0)
#         v = torch.cat([vx, vy, vz], dim=-1).to(dtype)
#
#         thr = 0.17428
#         mask = (x[:, :, 1] > thr).unsqueeze(-1)
#         extra = (0.3 * (~mask).to(dtype)).to(device=device)
#         return torch.cat([v, extra], dim=-1)

# class HullDynamics(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mlp = GenericLocalDynamicsMLP(in_dim=8, hidden_dim=64, cond_dim=1)
#
#         # 可学习基础速度（你已经做得很好）
#         self.learnable_base_speed = nn.Parameter(torch.tensor(0.3))
#
#         # 🌟 可学习水面阈值（强烈推荐！）
#         self.water_threshold = nn.Parameter(torch.tensor(0.17428))
#
#         # 🌟 可选：可学习软过渡宽度（让水面更平滑）
#         self.transition_width = nn.Parameter(torch.tensor(0.02))
#
#     def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
#         b, n, _ = x.shape
#         device, dtype = x.device, x.dtype
#         if cond.dim() == 2:
#             cond = cond.unsqueeze(1)
#
#         deltas = self.mlp(x, cond)
#         cond_exp = cond.expand(-1, n, -1)
#         local_angle = cond_exp[..., :1] + deltas[..., :1]
#
#         vx = torch.cos(math.pi * local_angle / 180.0)
#         vy = torch.zeros(b, n, 1, device=device, dtype=dtype)
#         vz = torch.sin(math.pi * local_angle / 180.0)
#         v = torch.cat([vx, vy, vz], dim=-1).to(dtype)
#
#         # ========================
#         # 🌟 可学习软阈值掩码（核心改进）
#         # ========================
#         y = x[:, :, 1:2]  # 水面高度维度
#
#         # 软掩码：从水下 → 水上平滑过渡，不是硬阶梯
#         mask = torch.sigmoid(
#             (y - self.water_threshold) / self.transition_width.abs()
#         )
#
#         # 水下 = 1，水上 = 0
#         water_mask = 1.0 - mask
#
#         # 可学习速度 * 掩码
#         extra = self.learnable_base_speed * water_mask
#         extra = extra.to(device=device, dtype=dtype)
#
#         return torch.cat([v, extra], dim=-1)

class HullDynamics(nn.Module):
    def __init__(self, use_nonlocal=False, nonlocal_context=128, nonlocal_reduction=2):
        super().__init__()
        self.mlp = GenericLocalDynamicsMLP(
            in_dim=8, hidden_dim=64, cond_dim=1,
            use_nonlocal=use_nonlocal,
            nonlocal_context=nonlocal_context,
            nonlocal_reduction=nonlocal_reduction,
        )
        self.learnable_base_speed = nn.Parameter(torch.tensor(0.3))
        # 根据预处理中的 scale 和 shift，0.17428 确实是对应的真实吃水线！
        self.water_threshold = nn.Parameter(torch.tensor(0.17428))
        self.transition_width = nn.Parameter(torch.tensor(0.02))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        device, dtype = x.device, x.dtype
        if cond.dim() == 2: cond = cond.unsqueeze(1)

        deltas = self.mlp(x, cond)
        cond_exp = cond.expand(-1, n, -1)
        local_angle = cond_exp[..., :1] + deltas[..., :1]

        # 完美适配这套被原作者“魔改”的坐标系
        vx = torch.cos(math.pi * local_angle / 180.0)
        vy = torch.zeros(b, n, 1, device=device, dtype=dtype)  # y 是高度，设为 0
        vz = torch.sin(math.pi * local_angle / 180.0)  # z 是横向，设为 sin
        v = torch.cat([vx, vy, vz], dim=-1).to(dtype)

        y_height = x[:, :, 1:2]  # index 1 确实是高度！

        # 软掩码
        width = torch.clamp(self.transition_width.abs(), min=1e-4, max=0.05)
        mask = torch.sigmoid((y_height - self.water_threshold) / width)

        water_mask = 1.0 - mask
        extra = self.learnable_base_speed * water_mask

        return torch.cat([v, extra.to(dtype)], dim=-1)


class DrivAerMLDynamics(nn.Module):
    def __init__(self, use_nonlocal=False, nonlocal_context=128, nonlocal_reduction=2):
        super().__init__()
        # x: 7维 (几何特征)
        # cond: 2维 [weight, angle_degrees]
        # in_dim = 7 + 2 = 9
        self.mlp = GenericLocalDynamicsMLP(
            in_dim=9, hidden_dim=64, cond_dim=2,
            use_nonlocal=use_nonlocal,
            nonlocal_context=nonlocal_context,
            nonlocal_reduction=nonlocal_reduction,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        device, dtype = x.device, x.dtype

        # 统一 cond 的维度为 [B, 1, 2]
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)

        # 1. 生成局部微扰
        # deltas 形状: [b, n, 2] -> 分别对应 [weight的微扰, angle的微扰]
        deltas = self.mlp(x, cond)
        cond_exp = cond.expand(-1, n, -1)

        # 2. 物理残差叠加
        # 原版直接硬编码了 base_speed = 0.3。我们用 deltas[..., 0:1] 去修正这个速度
        # 代表：局部真实风速 = 默认风洞风速(0.3) + MLP预测的尾流衰减
        local_speed = 0.3 + deltas[..., 0:1]

        # 用 deltas[..., 1:2] 去修正来流风向角 (偏航角)
        # 代表：局部真实风向 = 全局风向角 + 车身几何造成的局部气流偏转
        local_angle = cond_exp[..., 1:2] + deltas[..., 1:2]

        # 3. 严谨的物理公式转换 (三角函数展开)
        # 汽车通常没有仰角(Pitch)，所以 vz 保持 0，只在 XY 平面做三角分解
        vx = torch.cos(math.pi * local_angle / 180.0)
        vy = torch.sin(math.pi * local_angle / 180.0)
        vz = torch.zeros(b, n, 1, device=device, dtype=dtype)

        v = torch.cat([vx, vy, vz], dim=-1).to(dtype)
        w = local_speed.to(dtype)

        # 返回 4 维动态特化提示词: [B, N, 4]
        return torch.cat([v, w], dim=-1)


# -------- registry --------
_REGISTRY: Dict[str, Type[nn.Module]] = {
    "craft": CraftDynamics,
    "nasa": NasaDynamics,
    "crash": CrashDynamics,
    "hull": HullDynamics,
    "drivAerML": DrivAerMLDynamics,
}

_ALIASES: Dict[str, str] = {
    "Craft": "craft",
    "NASA": "nasa",
    "Hull": "hull",
    "Car": "drivAerML",
    "drivAerml": "drivAerML",
}


def get_direction(dynamics_config: str, use_nonlocal=False, nonlocal_context=128, nonlocal_reduction=2) -> nn.Module:
    """
    现在返回的是一个实例化的 nn.Module，内部自带参数和特化逻辑
    """
    if dynamics_config in _ALIASES:
        dynamics_config = _ALIASES[dynamics_config]

    if dynamics_config not in _REGISTRY:
        raise ValueError(f"Unknown dynamics_config='{dynamics_config}'. Supported: {list(_REGISTRY.keys())}")

    return _REGISTRY[dynamics_config](
        use_nonlocal=use_nonlocal,
        nonlocal_context=nonlocal_context,
        nonlocal_reduction=nonlocal_reduction,
    )
