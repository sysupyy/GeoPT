import numpy as np
import matplotlib.pyplot as plt

train_loss = np.load("training_logs/craft_geopt_8layers_SENet_newattention_v3_train_loss.npy")
test_loss = np.load("training_logs/craft_geopt_8layers_SENet_newattention_v3_test_loss.npy")

plt.figure(figsize=(9, 5))
plt.plot(train_loss, label='Train Loss', color='blue')
plt.plot(test_loss, label='Test Loss (rel_err)', color='red')

plt.title("hull : Train vs Test")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ===================== 我帮你加的部分：逐 epoch 打印 rel_err =====================
# print("\n===== 每个 Epoch 的 Test Rel_err 详细数值 =====")
# for epoch, err in enumerate(test_loss):
#     print(f"Epoch [{epoch+1:3d}] | Test Rel_err = {err:.8f}")

# 最终结果
print(f"\n=====  hull 最终结果汇总 =====")
print(f"最终训练集 Loss: {train_loss[-1]:.8f}")
print(f"最终测试集 Error: {test_loss[-1]:.8f}")