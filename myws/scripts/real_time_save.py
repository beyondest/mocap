# import numpy as np

# # 假设每帧的数据是 1x17x3 的 ndarray
# for frame_idx in range(100):  # 模拟 100 帧
#     frame_data = np.random.rand(1, 17, 3)  # 示例数据

#     # 实时保存每帧数据到独立文件
#     np.save(f"frame_{frame_idx:04d}.npy", frame_data)
#     print(f"帧 {frame_idx} 已保存")


# # 定期合并所有帧为一个 npz 文件
# frame_files = [f"frame_{i:04d}.npy" for i in range(100)]
# frames = [np.load(f) for f in frame_files]
# np.savez("trajectory_data.npz", *frames)
# print("所有帧已合并到 trajectory_data.npz")

import numpy as np
import pickle

# 实时写入每帧数据
with open("./../data/trajectory_data.pkl", "ab") as f:  # 使用 'ab' 模式追加写入
    for frame_idx in range(100):  # 模拟 100 帧
        frame_data = np.random.rand(1, 17, 3)  # 示例数据
        pickle.dump(frame_data, f)  # 写入一帧数据
        print(f"帧 {frame_idx} 已保存")
