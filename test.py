from myws.tools import *
import numpy as np
pose_yolo = np.array([[i,i] for i in range(17)])
pose_yolo = pose_yolo.reshape(1,17,2)

print(pose_yolo)
pose_h36m = Kpt.tran_yolo_to_h36m(pose_yolo)
print(pose_h36m)
pose_yolo = Kpt.tran_h36m_to_yolo(pose_h36m)
print(pose_yolo)

