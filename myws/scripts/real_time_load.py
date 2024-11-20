import pickle
import numpy as np
np.set_printoptions(precision=3, suppress=True)
# 逐帧读取数据
trajectory_list = []
with open("D:/VS_ws/python/mocap/myws/data/taiji_yunshou_speedup.pkl", "rb") as f:
    while True:
        try:
            frame_data = pickle.load(f)
            trajectory_list.append(frame_data)
        except EOFError:
            break

print(f"已加载 {len(trajectory_list)} 帧数据")
for i in range(len(trajectory_list)):
    print(f"第 {i+1} 帧数据：")
    data = trajectory_list[i]
    print(trajectory_list[i])