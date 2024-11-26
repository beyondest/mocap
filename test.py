import matplotlib.pyplot as plt
import numpy as np

# 设置 matplotlib 使用的后端
import matplotlib
#matplotlib.use('TkAgg')

# 创建数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 创建一个图形窗口并设置初始大小
fig = plt.figure(figsize=(8, 6))  # 初始宽为8英寸，高为6英寸

# 绘制图形
plt.plot(x, y)
plt.title("Adjustable Window Size Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# 显示图形，启动交互模式
plt.show()
