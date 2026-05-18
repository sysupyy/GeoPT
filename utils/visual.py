import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')

def visual(x, y, out, args, id):
    if args.geotype == 'structured_2D':
        visual_structured_2d(x, y, out, args, id)
    # 修复：使用 args.space_dim 进行判断，而不是 x 的特征维度
    if args.geotype == 'unstructured' and args.space_dim == 2:
        visual_unstructured_2d(x, y, out, args, id)
    if args.geotype == 'unstructured' and args.space_dim == 3:
        visual_unstructured_3d(x, y, out, args, id)


def visual_unstructured_3d(x, y, out, args, id, channel=0):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter3D(x[0, :, 0].detach().cpu().numpy(), x[0, :, 1].detach().cpu().numpy(),
                           x[0, :, 2].detach().cpu().numpy(), c=y[0, :, channel].detach().cpu().numpy(),
                           cmap='coolwarm',
                           s=50)#, vmin=0.0, vmax=0.06)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Value')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "gt_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter3D(x[0, :, 0].detach().cpu().numpy(), x[0, :, 1].detach().cpu().numpy(),
                           x[0, :, 2].detach().cpu().numpy(), c=out[0, :, channel].detach().cpu().numpy(),
                           cmap='coolwarm',
                           s=50)#, vmin=0.0, vmax=0.06)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Value')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "pred_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter3D(x[0, :, 0].detach().cpu().numpy(), x[0, :, 1].detach().cpu().numpy(),
                           x[0, :, 2].detach().cpu().numpy(),
                           c=(y[0, :, channel] - out[0, :, channel]).detach().cpu().numpy(),
                           cmap='coolwarm', s=50)#, vmin=-0.02, vmax=0.02)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Value')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "error_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()


def visual_unstructured_2d(x, y, out, args, id):
    plt.axis('off')
    plt.scatter(x=x[0, :, 0].detach().cpu().numpy(), y=x[0, :, 1].detach().cpu().numpy(),
                c=y[0, :].detach().cpu().numpy(), cmap='coolwarm')
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "gt_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.axis('off')
    plt.scatter(x=x[0, :, 0].detach().cpu().numpy(), y=x[0, :, 1].detach().cpu().numpy(),
                c=out[0, :].detach().cpu().numpy(), cmap='coolwarm')
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "pred_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.axis('off')
    plt.scatter(x=x[0, :, 0].detach().cpu().numpy(), y=x[0, :, 1].detach().cpu().numpy(),
                c=((y[0, :] - out[0, :])).detach().cpu().numpy(), cmap='coolwarm')
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "error_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()


def visual_structured_1d(x, y, out, args, id):
    pass


def visual_structured_2d(x, y, out, args, id):
    if args.vis_bound is not None:
        space_x_min = args.vis_bound[0]
        space_x_max = args.vis_bound[1]
        space_y_min = args.vis_bound[2]
        space_y_max = args.vis_bound[3]
    else:
        space_x_min = 0
        space_x_max = args.shapelist[0]
        space_y_min = 0
        space_y_max = args.shapelist[1]
    plt.axis('off')
    plt.pcolormesh(x[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   x[0, :, 1].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   np.zeros([args.shapelist[0], args.shapelist[1]])[space_x_min: space_x_max, space_y_min: space_y_max],
                   shading='auto',
                   edgecolors='black', linewidths=0.1)
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "input_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.axis('off')
    plt.pcolormesh(x[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   x[0, :, 1].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   out[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   shading='auto', cmap='coolwarm')
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "pred_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.axis('off')
    plt.pcolormesh(x[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   x[0, :, 1].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   y[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   shading='auto', cmap='coolwarm')
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "gt_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.axis('off')
    plt.pcolormesh(x[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   x[0, :, 1].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   out[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy() - \
                   y[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   shading='auto', cmap='coolwarm')
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "error_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()


def visual_structured_3d(x, y, out, args, id):
    pass


def visual_se_attention(all_attns, all_conds, args):
    """
    可视化 SE 模块在不同物理条件下的通道注意力分布
    all_attns: [N_samples, Channel]
    all_conds: [N_samples, Cond_dim] (对于 craft 任务，cond[:, 0] 通常是马赫数)
    """
    print("开始绘制 SE Attention 对比图...")
    # 1. 提取马赫数作为分类依据 (假设 cond 的第 0 维是 Mach)
    mach_values = all_conds[:, 0]

    # 2. 找到低速和高速的样本索引 (取最低 10% 和最高 10%)
    sort_idx = np.argsort(mach_values)
    num_samples = len(sort_idx)
    top_k = max(1, int(num_samples * 0.1))

    low_mach_idx = sort_idx[:top_k]
    high_mach_idx = sort_idx[-top_k:]

    # 3. 计算低速和高速情况下的平均通道激活值
    attn_low = np.mean(all_attns[low_mach_idx], axis=0)  # Shape: [64]
    attn_high = np.mean(all_attns[high_mach_idx], axis=0)  # Shape: [64]

    # 4. 绘制对比柱状图 (Bar Chart)
    channels = np.arange(len(attn_low))
    width = 0.4

    fig, ax = plt.subplots(figsize=(15, 6))

    # 使用对比强烈的颜色
    ax.bar(channels - width / 2, attn_low, width, label=f'Low Mach (avg: {mach_values[low_mach_idx].mean():.2f})',
           color='cornflowerblue', alpha=0.9)
    ax.bar(channels + width / 2, attn_high, width, label=f'High Mach (avg: {mach_values[high_mach_idx].mean():.2f})',
           color='tomato', alpha=0.9)

    ax.set_xlabel('SE-MLP Channel Index', fontsize=14)
    ax.set_ylabel('Attention Weight (Sigmoid Output)', fontsize=14)
    ax.set_title('Channel Attention Shift: Low Speed vs High Speed', fontsize=16)

    # 画一条 y=0.5 的参考线，说明哪些通道是被“激活”的，哪些是被“抑制”的
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 自动调整 x 轴刻度，防止太拥挤
    ax.set_xticks(channels[::2])
    ax.set_xticklabels(channels[::2])

    save_dir = os.path.join('./results/', args.save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, "se_attention_shift.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"SE Attention 对比图已保存至: {save_path}")


def visual_cosine_similarity(x, pred_v, gt_v, args, id, suffix=""):
    """
    可视化基底向量场与真实物理场的余弦相似度 (保存为高清 PDF)
    x: [B, N, 3] 几何坐标
    pred_v: [B, N, 3] 模型预测的提示词基底速度向量 (Vx, Vy, Vz)
    gt_v: [B, N, 3] 真实的速度场标签
    suffix: 保存文件名的后缀，用于区分原版和优化版 (如 "baseline", "ours")
    """
    import torch.nn.functional as F

    # 1. 计算余弦相似度
    cos_sim = F.cosine_similarity(pred_v, gt_v, dim=-1)  # [B, N]

    # 2. 安全检查：处理可能的 NaN
    if torch.isnan(cos_sim).any():
        print(f"[Warning] ID {id} ({suffix}) 的余弦相似度包含 NaN，将用 0 填充。")
        cos_sim = torch.nan_to_num(cos_sim, nan=0.0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_np = x[0, :, :3].detach().cpu().numpy()
    c_np = cos_sim[0, :].detach().cpu().numpy()

    # 3. 绘制 3D 散点图
    # coolwarm: 1.0(深红), 0.5(白), 0.0(深蓝)
    scatter = ax.scatter3D(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                           c=c_np, cmap='coolwarm', s=5, alpha=0.9, vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Absolute Cosine Similarity', fontsize=12)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 4. 调整视角
    ax.view_init(elev=30, azim=-60)

    # 清理背景网格
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    save_dir = os.path.join('./results/', args.save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 5. 保存为 PDF
    save_path = os.path.join(save_dir, f"cos_sim_{suffix}_{id}.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"成功保存余弦相似度绝对值图至: {save_path} (Mean Sim: {c_np.mean():.4f})")

def visual_deltas(x, deltas, args, id):
    """可视化局部微扰 (Deltas) 的空间分布 (保存为 PDF)"""
    delta_mag = torch.norm(deltas, dim=-1)

    if torch.isnan(delta_mag).any():
        print(f"[Warning] ID {id} 的 Deltas 包含 NaN，跳过画图。")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_np = x[0, :, :3].detach().cpu().numpy()
    c_np = delta_mag[0, :].detach().cpu().numpy()

    scatter = ax.scatter3D(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                           c=c_np, cmap='magma', s=5, alpha=0.9)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Delta Magnitude (Local Perturbation)', fontsize=12)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    save_dir = os.path.join('./results/', args.save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 恢复为 PDF 格式
    save_path = os.path.join(save_dir, f"deltas_{id}.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


def visual_cos_sim_diff(x, v_baseline, v_ours, gt_v, args, id):
    """绘制基底方向的相似度提升量 (保存为 PDF)"""
    sim_base = F.cosine_similarity(v_baseline, gt_v, dim=-1)
    sim_ours = F.cosine_similarity(v_ours, gt_v, dim=-1)
    diff = sim_ours - sim_base

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_np = x[0, :, :3].detach().cpu().numpy()
    c_np = diff[0, :].detach().cpu().numpy()

    # RdBu_r: 红色代表提升，蓝色代表变差。vmax=0.05 保证色彩饱和度足够。
    scatter = ax.scatter3D(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                           c=c_np, cmap='RdBu_r', s=5, alpha=0.9, vmin=-0.05, vmax=0.05)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Cosine Similarity Improvement (Ours - Baseline)', fontsize=12)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=-60)

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    save_dir = os.path.join('./results/', args.save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 恢复为 PDF 格式
    save_path = os.path.join(save_dir, f"cos_sim_diff_{id}.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"成功保存相似度提升图(PDF)至: {save_path} (Max Impr: {c_np.max():.4f})")