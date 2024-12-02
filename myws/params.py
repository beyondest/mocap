import numpy as np
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")).replace("\\", "/")

input_size = 640

yaml_file = os.path.join(root_path, 'utils/args.yaml').replace("\\", "/")
weights_file = os.path.join(root_path, 'weights/best.pt').replace("\\", "/")
onnx_file = os.path.join(root_path, 'weights/best20.onnx').replace("\\", "/")
trt_file = 'path/to/trt_engine.engine'  # .trt or .engine file path
trt_max_batch_size = 1              # max batch size for TRT engine, this is set during engine creation
trt_input_dtype = np.float16        # fp16 or fp32
video_path = os.path.join(root_path, 'myws/notes/taiji_yunshou_speedup.mp4').replace("\\", "/")
onnx_file2 = os.path.join(root_path, 'weights/grid20.onnx').replace("\\", "/")
save_data_path = os.path.join(root_path, 'myws/data/taiji_yunshou_speedup.pkl').replace("\\", "/")
trt_file2 = os.path.join(root_path, 'weights/grid20.engine').replace("\\", "/")


class MODEL_TYPE():
    ONNX = "onnx"
    PT = "pt"
    TRT = "trt"
model_type = MODEL_TYPE.ONNX

VISUALIZE_DRAW = True       # if True, draw detections on frame
VISUALIZE_PLOT = True    # if True, plot the skeleton using matplotlib
USE_CAMERA = False   # if True, use camera to capture video, otherwise use video file
TRANS_H36M = True    # if True, use H36M skeleton, otherwise use COCO skeleton



TRANS_TO_3D = True
SAVE_DATA = True