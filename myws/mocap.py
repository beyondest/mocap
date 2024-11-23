import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from myws.params import *
import argparse
import os
import warnings
from visualize import visualize_init,visualize_2d_pose,visualize_3d_pose
import numpy
import torch
import yaml
from torch.utils import data
import cv2
import time
from nets import nn
from utils import util
from myws.tools import *
import matplotlib.pyplot as plt
import pickle

from network import semantic_grid_trans, inverse_semantic_grid_trans
warnings.filterwarnings("ignore")
numpy.set_printoptions(precision=3)



@torch.no_grad()
def main():

    if VISUALIZE_PLOT:
        fig, ax, ax2 = visualize_init()
        plt.show(block=False)

    if model_type == MODEL_TYPE.ONNX:
        onnx_engine = Onnx_Engine(onnx_file,if_offline=False)
    elif model_type == MODEL_TYPE.TRT:
        trt_engine = TRT_Engine_2(trt_file,trt_max_batch_size)
    else:
        pass
    if TRANS_TO_3D:
        if model_type == MODEL_TYPE.ONNX:
            onnx_engine2 = Onnx_Engine(onnx_file2,if_offline=False)
        elif model_type == MODEL_TYPE.TRT:
            trt_engine2 = TRT_Engine_2(trt_file2,max_batchsize=1)
        else:
            pass
        
    if SAVE_DATA:
        if os.path.exists(save_data_path):
            raise ValueError(f"Save data path {save_data_path} already exists, please delete it first")
        f = open(save_data_path,"ab")
        
    skeleton = Kpt.Yolov8.skeleton if not TRANS_H36M else Kpt.H36M.skeleton
    kpt_color = Kpt.Yolov8.kpt_color if not TRANS_H36M else Kpt.H36M.kpt_color
    limb_color = Kpt.Yolov8.limb_color if not TRANS_H36M else Kpt.H36M.limb_color
    model = torch.load(weights_file, map_location='cpu')['model'].float()
    stride = int(max(model.stride.cpu().numpy()))

    model.half()
    model.eval()
    if USE_CAMERA:
        vd = cv2.VideoCapture(0)
    else:
        vd = cv2.VideoCapture(video_path)
    if not vd.isOpened():
        info = "camera" if USE_CAMERA else f"video : {video_path}"
        print(f"Error opening {info}")
        
    try:
        while vd.isOpened():
            
            # Capture frame-by-frame
            t1 = time.perf_counter()
            success, frame = vd.read()
            if success:
                image = frame.copy()
                center_x, center_y = image.shape[1]//2, image.shape[0]//2
                image = resize_image(image,target_size=(480,640),stride=stride,if_use_stride=False) # Resize to (3,480,640)
                # Inference
                if model_type == MODEL_TYPE.ONNX:
                    image = numpy.expand_dims(image,axis=0)
                    image = (image / 255).astype(numpy.float32)
                    outputs = onnx_engine.run(None,{'input':image})[0]

                elif model_type == MODEL_TYPE.TRT:
                    image = numpy.expand_dims(image,axis=0)
                    image = (image / 255).astype(trt_input_dtype)
                    outputs = trt_engine.run({0:image})[0]
                    outputs = outputs.reshape(1, 56, -1)
                    outputs = torch.from_numpy(outputs)
                else:
                    raise NotImplementedError(f"Model type {model_type} is not supported, only support onnx, pt and trt")
                # NMS
                outputs = non_max_suppression(outputs, 0.25, 0.7, model.head.nc)
                box_output, kps_output = pose_estimation_postprocess(outputs,image,frame,model)
                box_output = box_output.numpy() # (box_num, )
                kps_output = kps_output.numpy() # (box_num, 17, 3)
                if len(box_output) == 0:
                    print("No Person Detected")
                    continue
                else:
                    box_output = box_output[0].reshape(1, -1)
                    kps_output = kps_output[0].reshape(1, 17, 3)
                
                if TRANS_H36M:
                    kps_output = Kpt.tran_yolo_to_h36m(kps_output)
                if VISUALIZE_DRAW:
                    visualize_detections(frame,box_output,kps_output,kpt_color,skeleton,limb_color)
                if TRANS_TO_3D:
                    norm_kps_output = kps_output[:,:,:2]
                    norm_kps_output = (norm_kps_output - [center_x,center_y]) / [center_x,center_y]
                    norm_kps_output = semantic_grid_trans(norm_kps_output)
                    if model_type == MODEL_TYPE.ONNX:
                    
                        p3d = onnx_engine2.run(None,{'input':norm_kps_output.astype(numpy.float32)})[0]
                    elif model_type == MODEL_TYPE.TRT:
                        p3d = trt_engine2.run({0:norm_kps_output.astype(trt_input_dtype)})[0]
                        p3d = p3d.reshape(1, 17, 3)
                    
                    p3d = inverse_semantic_grid_trans(p3d)
                    p3d = p3d.reshape(1, 17, 3)
                    p3d = p3d * 1000 # mm to m
                    
                if VISUALIZE_PLOT:
                    visualize_2d_pose(kps_output, ax2, skeleton, limb_color)
                    if TRANS_TO_3D:
                        visualize_3d_pose(p3d[0], ax, skeleton,limb_color)
                if SAVE_DATA:
                    if not TRANS_TO_3D:
                        pickle.dump(kps_output[0],f)
                    else:
                        pickle.dump(p3d[0],f)
                t2 = time.perf_counter()
                fps = round(1/(t2 - t1))
                print(f'FPS : {fps}')
            else:
                print("End of Video")
                break
            
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
        
    except Exception as e:
        print(f"Get Error : {e}")  
        
    finally:
      
        vd.release()
        cv2.destroyAllWindows()
        if SAVE_DATA:
            f.close()
        
        print("All Resources released")


if __name__ == "__main__":
    
    util.setup_seed()
    #util.setup_multi_processes()
    maxmium_performance()
    if TRANS_TO_3D:
        if not TRANS_H36M:
            raise ValueError("if you want to transfer to 3D, please set TRANS_H36M to True")
    main()
    
    