import sys
import os
import multiprocessing
from liftpose.main import train as lp3d_train
import numpy as np
rand =np.random.rand

n_points, n_joints = 100, 5
train_2d, test_2d = rand(n_points, n_joints, 2), rand(n_points, n_joints, 2)
train_3d, test_3d = rand(n_points, n_joints, 3), rand(n_points, n_joints, 3)

train_2d = {"experiment_1": train_2d}
train_3d = {"experiment_1": train_3d}
test_2d = {"experiment_2": test_2d}
test_3d = {"experiment_2": test_3d}

roots = [0]
target_sets = [[1,2,3,4]]

if __name__ == "__main__":
    multiprocessing.freeze_support()
    lp3d_train(train_2d, test_2d, train_3d, test_3d, roots, target_sets)

