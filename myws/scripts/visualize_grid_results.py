import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from visualize import plot_pose_3d, plot_pose_2d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
from tools import *
from matplotlib.gridspec import GridSpec
from tools import Onnx_Engine
from network import *

datapath = "D:/Datasets/h36m/gt/test_custom_3d_unnorm.pth.tar"
data2dpath = "D:/Datasets/h36m/gt/test_custom_2d_unnorm.pth.tar"
onnx_path = 'D:/VS_ws/python/mocap/weights/grid20.onnx'

key = ('S11', 'Walking', 'Walking.60457274')
key = ('S11', 'Posing', 'Posing 1.54138969')
njoints = 17

data3d = torch.load(datapath,encoding = 'latin1')
data2d = torch.load(data2dpath,encoding = 'latin1')

s11walking3d = data3d[key]
s11walking2d = data2d[key]

joint_3d = s11walking3d['joint_3d']
pelvis = s11walking3d['pelvis']
camera = s11walking3d['camera']

pose_2d = s11walking2d
tar_3d = joint_3d.reshape(len(joint_3d), njoints, 3)
tar_2d = pose_2d.reshape(len(pose_2d), njoints, 2)
tar_2d[:,:,1] = -tar_2d[:,:,1] # flip y axis for visualization

fig = plt.figure(figsize=(15, 5), dpi=100)
gs = GridSpec(1, 2, figure=fig)  
ax = fig.add_subplot(gs[0], projection='3d')  
ax2 = fig.add_subplot(gs[1])  

ax.view_init(elev=-75, azim=-90)

plt.subplots_adjust(wspace=0.3)
t = 0

bones = Kpt.H36M.skeleton
limb_color = [Palettes.RGB.RED for i in range(len(bones))]

onnx_engine = Onnx_Engine(onnx_path)

def update(frame):
    global t
    ax.clear()
    ax2.clear()
    p2d = (tar_2d[t] - 500) / 500
    p2d = p2d.reshape(1, 17, 2)
    inp = semantic_grid_trans(p2d).astype(np.float32)
    output = onnx_engine.run(None,{'input':inp })[0]
    output = output 
    p3d = inverse_semantic_grid_trans(output)
    plot_pose_3d(ax=ax, tar=None, 
        pred = p3d[0], 
        bones=bones, 
        limb_color= limb_color, 
        legend=True)
    
    plot_pose_2d(ax=ax2, tar=tar_2d[t], 
        bones=bones, 
        limb_color= limb_color, 
        normalize=False)
    t += 1
    if t >= len(tar_3d):
        t = 0

ani = FuncAnimation(fig, update, frames=len(tar_3d), interval=1000/30, blit=False)
plt.show()
    














        

    
    
    




    
    



