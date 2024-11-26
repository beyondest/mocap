import numpy as np
input_size = 640
yaml_file = 'D:/VS_ws/python/mocap/utils/args.yaml'
weights_file = 'D:/VS_ws/python/mocap/weights/best.pt'
onnx_file = 'D:/VS_ws/python/mocap/weights/best20.onnx'
trt_file = 'path/to/trt_engine.engine'  # .trt or .engine file path
trt_max_batch_size = 1              # max batch size for TRT engine, this is set during engine creation
trt_input_dtype = np.float16        # fp16 or fp32
video_path = 'D:/VS_ws/python/mocap/myws/data/taiji_yunshou_speedup.mp4'
onnx_file2 = 'D:/VS_ws/python/mocap/weights/grid20.onnx'
save_data_path = 'D:/VS_ws/python/mocap/myws/data/taiji_yunshou_speedup.pkl'
trt_file2 = 'D:/VS_ws/python/mocap/weights/grid20.engine'  # .trt or .engine file path


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
SAVE_DATA = False