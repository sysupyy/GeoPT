import numpy as np
import matplotlib.pyplot as plt


def inspect_velocity_normal_constraint(x_file_path, y_file_path):
    print(f"正在加载几何文件: {x_file_path}")
    print(f"正在加载标签文件: {y_file_path}")

    # 1. 加载数据
    try:
        data_x = np.load(x_file_path)
        data_y = np.load(y_file_path)
    except FileNotFoundError as e:
        print(f"找不到文件，请检查路径是否正确！\n报错信息: {e}")
        return

    # 数据形状通常为 [N, C] 或 [B, N, C]，如果是后者取第一个 Batch
    if data_x.ndim == 3:
        data_x = data_x[0]
    if data_y.ndim == 3:
        data_y = data_y[0]

    assert data_x.shape[0] == data_y.shape[0], "几何节点数与标签节点数不匹配！"

    print("\n" + "=" * 50)
    print(f"几何数据维度: {data_x.shape}, 标签数据维度: {data_y.shape}")

    # 2. 提取各个特征列
    coords = data_x[:, 0:3]
    sdf = data_x[:, 3]
    normals = data_x[:, 4:7]  # Nx, Ny, Nz
    velocities = data_y[:, 2:5]  # 真实速度场 Vx, Vy, Vz

    # 3. 统计表面节点 (SDF 接近 0 的点)
    surface_mask = np.abs(sdf) < 1e-5
    surface_idx = np.where(surface_mask)[0]

    if len(surface_idx) == 0:
        print("未检测到表面节点！请检查 SDF 列是否正确。")
        return

    # 提取表面节点的速度和法向量
    v_surf = velocities[surface_mask]
    n_surf = normals[surface_mask]

    # 4. 计算法向速度 (V · n)
    v_dot_n = np.sum(v_surf * n_surf, axis=-1)

    # 获取表面真实速度的平均大小 (用于对比量级)
    v_magnitude = np.linalg.norm(v_surf, axis=-1)

    print(f"\n--- 物理约束 ($V \cdot n = 0$) 验证统计 ---")
    print(f"表面节点总数: {len(surface_idx)}")
    print(f"表面气流真实平均速度 |V|: {np.mean(v_magnitude):.6f}")
    print("-" * 30)
    print(f"V·n 平均绝对误差 (MAE): {np.mean(np.abs(v_dot_n)):.6e}")
    print(f"V·n 最大绝对误差 (Max): {np.max(np.abs(v_dot_n)):.6e}")
    print(f"V·n 均方误差 (MSE): {np.mean(v_dot_n ** 2):.6e}")
    print("=" * 50 + "\n")

    # 5. 可视化分析
    print("正在生成诊断可视化图表...")
    fig = plt.figure(figsize=(16, 6))

    # --- 子图 1：表面节点 3D 误差热力图 ---
    ax1 = fig.add_subplot(121, projection='3d')
    surf_step = max(1, len(surface_idx) // 8000)  # 降采样防卡顿

    # 用绝对误差大小给点上色
    v_dot_n_sampled = np.abs(v_dot_n[::surf_step])
    sc = ax1.scatter(coords[surface_idx[::surf_step], 0],
                     coords[surface_idx[::surf_step], 1],
                     coords[surface_idx[::surf_step], 2],
                     c=v_dot_n_sampled, cmap='Reds', s=2, alpha=0.8)

    plt.colorbar(sc, ax=ax1, label='Absolute Normal Velocity |V · n|', shrink=0.7)
    ax1.set_title("3D Distribution of |V · n| Errors")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_box_aspect([1, 1, 1])  # 保持比例不拉伸

    # --- 子图 2：V·n 值分布直方图 ---
    ax2 = fig.add_subplot(122)
    ax2.hist(v_dot_n, bins=150, color='royalblue', alpha=0.75, edgecolor='black', linewidth=0.5)
    ax2.axvline(0, color='red', linestyle='dashed', linewidth=2, label='Ideal V·n = 0')
    ax2.set_title("Histogram of V · n Values on Surface")
    ax2.set_xlabel("V · n Value")
    ax2.set_ylabel("Node Count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    save_path = "vn_constraint_inspection.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化完成！图片已保存至当前目录: {save_path}")


if __name__ == "__main__":
    # 请确保 x_0.npy 和 y_0.npy 都在该目录下
    x_file = "./aircraft_npys/x_0.npy"
    y_file = "./aircraft_npys/y_0.npy"

    inspect_velocity_normal_constraint(x_file, y_file)