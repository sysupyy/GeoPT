import os
import torch
import torch.nn as nn
from exp.exp_basic import Exp_Basic
from exp.dynamics_config import get_direction  # 这里的 get_direction 现在返回包含 MLP 参数的 nn.Module
from models.model_factory import get_model
from data_provider.data_factory import get_data
from utils.loss import L2Loss
from utils.visual import visual
import matplotlib.pyplot as plt
import numpy as np
import math


class Exp_Steady(Exp_Basic):

    def __init__(self, args):
        super(Exp_Steady, self).__init__(args)

        if not hasattr(self.args, "dynamics"):
            raise ValueError("args.dynamics is required.")

        # ==========================================
        # 🚀 极其优雅的初始化：
        # 无论是什么任务，统一从 registry 获取特化的物理驱动器（内含 MLP 和 SEBlock 参数）
        # ==========================================
        self.direction = get_direction(self.args.dynamics).cuda()
        print(f"[Exp_Steady] Dynamics module '{self.args.dynamics}' loaded with unified prompt generator.")

    def vali(self):
        myloss = L2Loss(size_average=False)
        self.model.eval()
        self.direction.eval()  # 统一开启 eval 模式

        rel_err = 0.0
        with torch.no_grad():
            for pos, fx, cond, y in self.test_loader:
                x, fx, cond, y = pos.cuda(), fx.cuda(), cond.cuda(), y.cuda()

                # 统一调用 direction 生成局部提示词
                v = self.direction(x, cond)

                fx = torch.cat((fx, v), dim=-1)
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x[:, :, :3], fx)
                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)

                tl = myloss(out, y).item()
                rel_err += tl

        rel_err /= self.args.ntest
        return rel_err

    def load_pretrained_with_filter(self, model, pretrained_path, exclude_layers=None, map_location="cpu"):
        if exclude_layers is None:
            exclude_layers = ("mlp2", "ln_3")  # exclude last layer

        pretrained = torch.load(pretrained_path, map_location=map_location)

        # 兼容处理：防止加载的是包含了 direction/local_dynamics 的检查点字典
        if 'model' in pretrained:
            pretrained = pretrained['model']

        model_state = model.state_dict()

        filtered = {
            k: v
            for k, v in pretrained.items()
            if k in model_state
               and model_state[k].shape == v.shape
               and not any(excl in k for excl in exclude_layers)
        }

        model_state.update(filtered)
        model.load_state_dict(model_state)

        print(f"[Pretrain] Loaded {len(filtered)}/{len(pretrained)} parameters")
        return model

    def train(self):
        ### load GeoPT pre-trained model
        if self.args.finetune:
            self.model = self.load_pretrained_with_filter(self.model,
                                                          "./checkpoints/" + self.args.finetune_name + ".pt")

        # ==========================================
        # 🚀 统一将 GeoPT 主模型和动态物理提示词生成器的参数送入优化器
        # ==========================================
        trainable_params = list(self.model.parameters()) + list(self.direction.parameters())

        if self.args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(trainable_params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(trainable_params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise ValueError('Optimizer only AdamW or Adam')

        ### adopt learning rate scheduler
        if self.args.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr, epochs=self.args.epochs,
                                                            steps_per_epoch=len(self.train_loader),
                                                            pct_start=self.args.pct_start)
        elif self.args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        elif self.args.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        myloss = L2Loss(size_average=False)

        train_loss_list = []
        test_loss_list = []

        for ep in range(self.args.epochs):

            self.model.train()
            self.direction.train()  # 统一开启 train 模式

            train_loss = 0
            train_dir_loss = 0

            # 🌟 保留你设计的动态权重机制框架，供未来使用
            # initial_dir_weight = 1.6
            # final_dir_weight = 0.2
            # progress = ep / self.args.epochs
            # current_dir_weight = final_dir_weight + 0.5 * (initial_dir_weight - final_dir_weight) * (1 + math.cos(math.pi * progress))

            for pos, fx, cond, y in self.train_loader:
                x, fx, cond, y = pos.cuda(), fx.cuda(), cond.cuda(), y.cuda()

                # 1. 统一获取动态风场特征
                v_dyn = self.direction(x, cond)
                # 提取纯单位风场方向 [vx, vy, vz]，用于辅助监督
                v_pred_dir = v_dyn[:, :, :3]

                fx = torch.cat((fx, v_dyn), dim=-1)
                if self.args.fun_dim == 0:
                    fx = None

                # 2. 主模型前向传播
                out = self.model(x[:, :, :3], fx)

                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)
                    y = self.dataset.y_normalizer.decode(y)

                # 3. 计算主模型的预测 Loss
                main_loss = myloss(out, y)

                # ==========================================
                # 🚀 物理风场方向辅助监督 (精准适配各数据集坐标系)
                # ==========================================
                if self.args.dynamics in ['craft', 'hull', 'drivAerml']:
                    # 1. 提取真实速度分量
                    if self.args.dynamics == 'craft':
                        # AirCraft: 速度通道 2:5 [u, v, w], 无需变换
                        v_true = y[:, :, 2:5]

                    elif self.args.dynamics == 'drivAerml':
                        # DrivAerML: 速度通道 1:4 [Ux, Uy, Uz], 无需变换
                        v_true = y[:, :, 1:4]

                    elif self.args.dynamics == 'hull':
                        # 🌟 DTCHull 特化逻辑：同步执行预处理中的 Transform 变换
                        # 原始速度通道 1:4 [Ux, Uy, Uz]
                        v_raw = y[:, :, 1:4]

                        # 按照预处理脚本: new_x = -old_x, new_y = old_z, new_z = old_y
                        v_true = torch.zeros_like(v_raw)
                        v_true[:, :, 0] = -v_raw[:, :, 0]  # 航向取反
                        v_true[:, :, 1] = v_raw[:, :, 2]  # y轴取自原始z
                        v_true[:, :, 2] = v_raw[:, :, 1]  # z轴取自原始y

                    # 2. 计算真实速度的单位方向向量
                    v_true_unit = torch.nn.functional.normalize(v_true, p=2, dim=-1, eps=1e-8)

                    # 3. 计算预测方向 (v_pred_dir) 与 真实方向 (v_true_unit) 的余弦相似度
                    cos_sim = torch.sum(v_pred_dir * v_true_unit, dim=-1)

                    # 4. 计算辅助损失 (1 - cos_sim)
                    direction_loss = torch.mean(1.0 - cos_sim)

                    # 5. 损失合并 (保持 1.0 权重)
                    # loss = main_loss + 1.0 * direction_loss
                    if self.args.dynamics == 'craft':
                        loss = main_loss + 1.0 * direction_loss
                    elif self.args.dynamics == 'drivAerml':
                        loss = main_loss + 1.0 * direction_loss
                    elif self.args.dynamics == 'hull':
                        loss = main_loss + 1.5 * direction_loss

                    train_dir_loss += direction_loss.item()

                else:
                    # 其他数据集（nasa, crash）维持原样
                    loss = main_loss

                train_loss += main_loss.item()

                optimizer.zero_grad()
                loss.backward()

                if self.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()

                if self.args.scheduler == 'OneCycleLR':
                    scheduler.step()
            if self.args.scheduler == 'CosineAnnealingLR' or self.args.scheduler == 'StepLR':
                scheduler.step()

            train_loss = train_loss / self.args.ntrain
            train_dir_loss = train_dir_loss / self.args.ntrain

            print("Epoch {} Train loss : {:.5f} | Dir loss : {:.5f}".format(ep, train_loss, train_dir_loss))
            train_loss_list.append(train_loss)

            rel_err = self.vali()
            print("rel_err:{}".format(rel_err))
            test_loss_list.append(rel_err)

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save models')
                # 统一保存 GeoPT 和 特化驱动器 的参数
                state_to_save = {
                    'model': self.model.state_dict(),
                    'direction': self.direction.state_dict()
                }
                torch.save(state_to_save, os.path.join('./checkpoints', self.args.save_name + '.pt'))

            if ep % 10 == 0:
                if not os.path.exists('./training_logs'):
                    os.makedirs('./training_logs')
                print('save logs')
                np.save(os.path.join('./training_logs', self.args.save_name + '_train_loss.npy'),
                        np.array(train_loss_list))
                np.save(os.path.join('./training_logs', self.args.save_name + '_test_loss.npy'),
                        np.array(test_loss_list))

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('final save models')
        state_to_save = {
            'model': self.model.state_dict(),
            'direction': self.direction.state_dict()
        }
        torch.save(state_to_save, os.path.join('./checkpoints', self.args.save_name + '.pt'))

        if not os.path.exists('./training_logs'):
            os.makedirs('./training_logs')
        print('final training logs')
        np.save(os.path.join('./training_logs', self.args.save_name + '_train_loss.npy'), np.array(train_loss_list))
        np.save(os.path.join('./training_logs', self.args.save_name + '_test_loss.npy'), np.array(test_loss_list))

    def test(self):
        import os
        import math
        import torch
        import numpy as np
        from utils.visual import visual, visual_deltas, visual_cosine_similarity, visual_cos_sim_diff, \
            visual_se_attention
        from utils.loss import L2Loss  # 假设 L2Loss 在这里，请按你原有的 import 调整

        checkpoint = torch.load("./checkpoints/" + self.args.save_name + ".pt")
        # ⚠️ 加上 strict=False，防止加载带有额外门控/LoRA结构的模型时报错
        self.model.load_state_dict(checkpoint['model'], strict=False)

        # 兼容性设计：支持加载旧版名为 local_dynamics 的权重，也支持新版 direction 权重
        if 'direction' in checkpoint:
            self.direction.load_state_dict(checkpoint['direction'])
        elif 'local_dynamics' in checkpoint:
            self.direction.load_state_dict(checkpoint['local_dynamics'])

        self.model.eval()
        self.direction.eval()

        if not os.path.exists('./results/' + self.args.save_name + '/'):
            os.makedirs('./results/' + self.args.save_name + '/')

        rel_err = 0.0
        rel_err_split = 0.0
        rel_err_split_max = 0.0
        id = 0
        mse = 0.0
        mae = 0.0
        myloss = L2Loss(size_average=False)

        # 🌟 1. 初始化收集列表，用于分析 SE Attention 随物理条件的变化
        all_attns = []
        all_conds = []

        with torch.no_grad():
            for pos, fx, cond, y in self.test_loader:
                id += 1
                x, fx, cond, y = pos.cuda(), fx.cuda(), cond.cuda(), y.cuda()

                # ---------------------------------------------------------
                # 可视化前置准备：获取最纯净的 Baseline 和 Ours
                # 1. 模拟 Baseline：完全不依赖网络，用物理公式算出初始风向
                b, n, _ = x.shape
                cond_exp_base = cond.expand(-1, n, -1) if cond.dim() == 3 else cond.unsqueeze(1).expand(-1, n, -1)

                if self.args.dynamics == 'craft':
                    global_aoa = cond_exp_base[:, :, 1:2]
                    global_beta = cond_exp_base[:, :, 2:3]
                    # 算出纯物理的初始方向向量
                    vx_b = torch.cos(math.pi * global_aoa / 180.0) * torch.cos(math.pi * global_beta / 180.0)
                    vy_b = torch.sin(math.pi * global_aoa / 180.0)
                    vz_b = torch.cos(math.pi * global_aoa / 180.0) * torch.sin(math.pi * global_beta / 180.0)
                    v_baseline = torch.cat([vx_b, vy_b, vz_b], dim=-1).to(device=x.device, dtype=x.dtype)
                else:
                    global_aoa = cond_exp_base[:, :, 1:2] if cond_exp_base.shape[-1] >= 2 else cond_exp_base[:, :, 0:1]
                    vx_b = torch.cos(math.pi * global_aoa / 180.0)
                    vy_b = torch.sin(math.pi * global_aoa / 180.0)
                    vz_b = torch.zeros(b, n, 1, device=x.device, dtype=x.dtype)
                    v_baseline = torch.cat([vx_b, vy_b, vz_b], dim=-1).to(device=x.device, dtype=x.dtype)

                # 2. 获取你优化后的最终提示词基底
                v_full_ours = self.direction(x, cond)
                v_ours = v_full_ours[..., :3]

                # 3. 算出真正意义上的物理微扰 (Physical Deltas)
                true_physical_deltas = v_ours - v_baseline

                # ---------------------------------------------------------
                # 可视化 1: 画出真正的物理微扰分布图
                if id <= self.args.vis_num:
                    print(f'\nvisual true physical deltas: {id}')
                    visual_deltas(x[:, :, :3], true_physical_deltas, self.args, id)

                # ---------------------------------------------------------
                # 可视化 2: 综合绘制余弦相似度的绝对大小与提升量
                if id <= self.args.vis_num:
                    # 依然使用我们验证过正确的、归一化后的 y 空间
                    if self.args.dynamics == 'craft':
                        gt_v = y[..., 2:5]
                    else:
                        gt_v = y[..., 1:4]

                    from utils.visual import visual_cosine_similarity, visual_cos_sim_diff

                    print(f'\nvisual absolute cosine similarity: {id}')
                    # 画出 Baseline 的绝对余弦相似度 (PDF)
                    visual_cosine_similarity(x[:, :, :3], v_baseline, gt_v, self.args, id, suffix="baseline")

                    # 画出 Ours (引入SE-MLP后) 的绝对余弦相似度 (PDF)
                    visual_cosine_similarity(x[:, :, :3], v_ours, gt_v, self.args, id, suffix="ours")

                    print(f'visual cos sim diff: {id}')
                    # 画出 Ours 相比 Baseline 的差值提升图 (PDF)
                    visual_cos_sim_diff(x[:, :, :3], v_baseline, v_ours, gt_v, self.args, id)
                # ---------------------------------------------------------

                # 🌟 2. 收集 SE 探针数据
                if hasattr(self.direction, 'mlp') and hasattr(self.direction.mlp, 'se'):
                    if hasattr(self.direction.mlp.se, 'saved_attention'):
                        # saved_attention 的 shape 是 [B, 1, channel]，去掉多余的维度
                        attn_val = self.direction.mlp.se.saved_attention.squeeze(1).cpu().numpy()
                        cond_val = cond.squeeze(1).cpu().numpy()
                        all_attns.append(attn_val)
                        all_conds.append(cond_val)

                fx = torch.cat((fx, v_full_ours), dim=-1)
                if self.args.fun_dim == 0:
                    fx = None

                # 传入主干网络预测
                out = self.model(x[:, :, :3], fx)

                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)

                # 计算误差
                tl = myloss(out, y).item()
                mse += (out - y).pow(2).mean(dim=1).mean(dim=1).sum().item()
                mae += torch.abs(out - y).mean(dim=1).mean(dim=1).sum().item()
                rel_err += tl
                rel_err_split += torch.mean(torch.mean(torch.abs(out - y) / torch.abs(y), dim=0), dim=0)
                rel_err_split_max += torch.max(torch.max(torch.abs(out - y) / torch.abs(y), dim=0)[0], dim=0)[0]

                if id < self.args.vis_num:
                    print('visual original out: ', id)
                    visual(x, y, out, self.args, id)

        rel_err /= self.args.ntest
        mse /= self.args.ntest
        mae /= self.args.ntest
        rel_err_split /= self.args.ntest
        rel_err_split_max /= self.args.ntest
        print("test rel_err:{}".format(rel_err))
        print("test mse:{}".format(mse))
        print("test mae:{}".format(mae))
        print("test rel_err split:{}".format(rel_err_split))
        print("test rel_err split max:{}".format(rel_err_split_max))

        # 🌟 3. 在所有样本测试完毕后，触发统计画图
        if len(all_attns) > 0:
            all_attns_np = np.concatenate(all_attns, axis=0)  # [Total_Samples, 64]
            all_conds_np = np.concatenate(all_conds, axis=0)  # [Total_Samples, cond_dim]

            try:
                visual_se_attention(all_attns_np, all_conds_np, self.args)
            except ImportError:
                print("[Warning] 未找到 visual_se_attention 函数，请确保已在 utils/visual.py 中添加！")

    def test_full_mesh(self):
        checkpoint = torch.load("./checkpoints/" + self.args.save_name + ".pt")
        self.model.load_state_dict(checkpoint['model'])

        if 'direction' in checkpoint:
            self.direction.load_state_dict(checkpoint['direction'])
        elif 'local_dynamics' in checkpoint:
            self.direction.load_state_dict(checkpoint['local_dynamics'])

        self.model.eval()
        self.direction.eval()

        if not os.path.exists('./results/' + self.args.save_name + '/'):
            os.makedirs('./results/' + self.args.save_name + '/')

        rel_err = 0.0
        rel_err_split = 0.0
        rel_err_split_max = 0.0
        id = 0
        mse = 0.0
        mae = 0.0
        myloss = L2Loss(size_average=False)

        with torch.no_grad():
            for pos, fx, cond, y in self.test_loader_full:
                id += 1
                x, fx, cond, y = pos.cuda(), fx.cuda(), cond.cuda(), y.cuda()

                v = self.direction(x, cond)

                fx = torch.cat((fx, v), dim=-1)
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x[:, :, :3], fx)
                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)
 
                tl = myloss(out, y).item()
                mse += (out - y).pow(2).mean(dim=1).mean(dim=1).sum().item()
                mae += torch.abs(out - y).mean(dim=1).mean(dim=1).sum().item()
                rel_err += tl
                rel_err_split += torch.mean(torch.mean(torch.abs(out - y) / torch.abs(y), dim=0), dim=0)
                rel_err_split_max += torch.max(torch.max(torch.abs(out - y) / torch.abs(y), dim=0)[0], dim=0)[0]

                if id < self.args.vis_num:
                    print('visual: ', id)
                    visual(x, y, out, self.args, id)

        rel_err /= self.args.ntest
        mse /= self.args.ntest
        mae /= self.args.ntest
        rel_err_split /= self.args.ntest
        rel_err_split_max /= self.args.ntest
        print("test rel_err:{}".format(rel_err))
        print("test mse:{}".format(mse))
        print("test mae:{}".format(mae))
        print("test rel_err split:{}".format(rel_err_split))
        print("test rel_err split max:{}".format(rel_err_split_max))