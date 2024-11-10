import numpy as np

# 假设输入数组为 1x17x2 形状
data = np.random.rand(1, 17, 2)  # 示例数据
d = np.random.rand(480,640)
# 获取中心（shape），可以使用数组的中间点，或者手动指定值
center_x, center_y = data.shape[1] / 2, data.shape[2] / 2

# 计算偏移，normalize 使其中心为 shape
normalized_data = data - [center_x, center_y]

print("Normalized data:", normalized_data)
