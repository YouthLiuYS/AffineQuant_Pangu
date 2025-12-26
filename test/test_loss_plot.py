"""
测试loss曲线绘制逻辑
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 模拟layer_loss_data数据结构
# {layer_id: [(epoch, loss), ...]}
layer_loss_data = {}

# 创建测试数据：假设有5层，每层训练10个epoch
num_test_layers = 5
epochs_per_layer = 10

for layer_id in range(num_test_layers):
    loss_list = []
    # 为每层生成不同的loss曲线模式
    base_loss = 1.0 + layer_id * 0.1  # 不同层起始loss不同
    for epoch in range(epochs_per_layer):
        # 模拟loss下降趋势，加入一些随机波动
        loss = base_loss * np.exp(-0.2 * epoch) + np.random.random() * 0.05
        loss_list.append((epoch, loss))
    layer_loss_data[layer_id] = loss_list

print("测试数据:")
for layer_id, loss_list in layer_loss_data.items():
    print(f"Layer {layer_id}: {len(loss_list)} epochs, loss range [{min(l for _, l in loss_list):.4f}, {max(l for _, l in loss_list):.4f}]")

# 使用与affinequant.py相同的绘图逻辑
fig, ax = plt.subplots(figsize=(18, 6))

# Use a colormap to assign different colors to each layer
num_layers = len(layer_loss_data)
colors = cm.tab20(np.linspace(0, 1, num_layers)) if num_layers <= 20 else cm.viridis(np.linspace(0, 1, num_layers))

# Track global step counter
current_global_step = 0
layer_boundaries = []

for idx, (layer_id, loss_list) in enumerate(sorted(layer_loss_data.items())):
    # Assign global steps for this layer
    global_steps = []
    losses = []
    
    for epoch, loss in loss_list:
        global_steps.append(current_global_step)
        losses.append(loss)
        current_global_step += 1
    
    # Plot this layer's curve (points within same layer are connected)
    ax.plot(global_steps, losses, marker='o', markersize=4, linewidth=1.5, 
           color=colors[idx], label=f'Layer {layer_id}', alpha=0.8)
    
    # Record layer boundary for visualization
    if global_steps:
        layer_boundaries.append((global_steps[0], layer_id))
    
    print(f"Layer {layer_id}: global steps {global_steps[0]} to {global_steps[-1]}")

# Add vertical lines at layer boundaries (except the first one)
for i in range(1, len(layer_boundaries)):
    boundary_x, layer_id = layer_boundaries[i]
    ax.axvline(x=boundary_x - 0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

ax.set_xlabel('Global Step', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Loss Curves by Layer - Test (逐层量化Loss变化测试)', fontsize=14)
ax.grid(True, alpha=0.3)

# Create legend
if num_layers > 10:
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=max(1, num_layers // 15))
    plt.tight_layout(rect=[0, 0, 0.88, 1])
else:
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()

# 保存图片
output_file = '/home/lys/pangu/AffineQuant/test_loss_curves.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n图片已保存到: {output_file}")
print("\n验证要点:")
print("1. 每层的曲线应该是独立的，不同层之间不应该有连线")
print("2. 横坐标是连续的global step (0到49)")
print("3. 每层用不同颜色，有10个点(epoch 0-9)")
print("4. 层边界处有灰色虚线分隔")
print("5. 每层的loss应该呈现下降趋势")

plt.show()
