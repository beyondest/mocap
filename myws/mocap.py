import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from myws.params import *
import argparse
import os
import warnings

import numpy
import torch
import yaml
from torch.utils import data
import cv2
import time
from nets import nn
from utils import util
from myws.tools import *

warnings.filterwarnings("ignore")
numpy.set_printoptions(precision=3)



@torch.no_grad()
def demo():
    if model_type == MODEL_TYPE.ONNX:
        onnx_engine = Onnx_Engine(onnx_file,if_offline=False)
    elif model_type == MODEL_TYPE.TRT:
        trt_engine = TRT_Engine_2(trt_file,trt_max_batch_size)
    else:
        pass

    skeleton = Kpt.Yolov8.skeleton
    kpt_color = Kpt.Yolov8.kpt_color
    limb_color = Kpt.Yolov8.limb_color
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
                shape = image.shape[:2]  # current shape [height, width]
                image = resize_image(image,target_size=(480,640),stride=stride,if_use_stride=False) # Resize to (3,480,640)
                # Inference
                
                if model_type == MODEL_TYPE.ONNX:
                    image = numpy.expand_dims(image,axis=0)
                    image = (image / 255).astype(numpy.float32)
                    outputs = onnx_engine.run(None,{'input':image})[0]
                    outputs = torch.from_numpy(outputs)
                    
                elif model_type == MODEL_TYPE.PT:
                    image = torch.from_numpy(image)
                    image = image.unsqueeze(0)
                    image = image.half()
                    image = image / 255
                    outputs = model(image)
                    
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
                if VISUALIZE:
                    visualize_detections(frame,box_output,kps_output,kpt_color,skeleton,limb_color)
                else:
                    print(f"KPS_OUTPUT : {kps_output}")

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
      
    vd.release()
    cv2.destroyAllWindows()
    print("All Resources released")


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input-size', default=input_size, type=int)
    # parser.add_argument('--local_rank', default=0, type=int)
    # parser.add_argument('--demo', action='store_true')

    # args = parser.parse_args()

    # args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    # args.world_size = int(os.getenv('WORLD_SIZE', 1))

    # if args.world_size > 1:
    #     torch.cuda.set_device(device=-1)
    #     torch.distributed.init_process_group(backend='nccl', init_method='env://')
    util.setup_seed()
    #util.setup_multi_processes()
    maxmium_performance()
    demo()


if __name__ == "__main__":
    main()
