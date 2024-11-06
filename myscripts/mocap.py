import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from myscripts.params import *
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
from myscripts.tools import Onnx_Engine
warnings.filterwarnings("ignore")




@torch.no_grad()
def demo(args):
    onnx_engine = Onnx_Engine(onnx_file,if_offline=False)
    
    
    palette = numpy.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                           [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                           [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                           [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                          dtype=numpy.uint8)
    # which 2 keypoint will bind together to form a limb
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    # keypoint color
    # 17 keypoints
    kpt_color = palette[[16, 16, 16,
                         16, 16,  0,
                         0,   0,  0, 
                         0,   0,  9,
                         9,   9,  9, 
                         9,   9]]
    # limb color
    limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    model = torch.load(weights_file, map_location='cpu')['model'].float()
    stride = int(max(model.stride.cpu().numpy()))

    model.half()
    model.eval()

    camera = cv2.VideoCapture(0)
    # Check if camera opened successfully
    if not camera.isOpened():
        print("Error opening video stream or file")
    # Read until video is completed
    try:
        while camera.isOpened():
            
            # Capture frame-by-frame
            t1 = time.perf_counter()
            success, frame = camera.read()
            if success:
                print(frame.shape)
                image = frame.copy()
                shape = image.shape[:2]  # current shape [height, width]
                r = min(1.0, args.input_size / shape[0], args.input_size / shape[1])
                pad = int(round(shape[1] * r)), int(round(shape[0] * r))
                w = args.input_size - pad[0]
                h = args.input_size - pad[1]
                w = numpy.mod(w, stride)
                h = numpy.mod(h, stride)
                w /= 2
                h /= 2
                if shape[::-1] != pad:  # resize
                    image = cv2.resize(image,
                                    dsize=pad,
                                    interpolation=cv2.INTER_LINEAR)
                top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
                left, right = int(round(w - 0.1)), int(round(w + 0.1))
                image = cv2.copyMakeBorder(image,
                                        top, bottom,
                                        left, right,
                                        cv2.BORDER_CONSTANT)  # add border
                # Convert HWC to CHW, BGR to RGB
                image = image.transpose((2, 0, 1))[::-1]
                image = numpy.ascontiguousarray(image)
                print(image.shape)
                # Inference
                
                if USE_ONNX:
                    image = numpy.expand_dims(image,axis=0)
                    image = (image / 255).astype(numpy.float32)
                    outputs = onnx_engine.run(None,{'input':image})[0]
                    outputs = torch.from_numpy(outputs)
                else:
                    image = torch.from_numpy(image)
                    image = image.unsqueeze(0)
                    image = image.half()
                    image = image / 255
                    outputs = model(image)

                    
                # NMS
                outputs = util.non_max_suppression(outputs, 0.25, 0.7, model.head.nc)
                for output in outputs:
                    output = output.clone()
                    if len(output):
                        box_output = output[:, :6]
                        kps_output = output[:, 6:].view(len(output), *model.head.kpt_shape)
                    else:
                        box_output = output[:, :6]
                        kps_output = output[:, 6:]

                    r = min(image.shape[2] / shape[0], image.shape[3] / shape[1])

                    box_output[:, [0, 2]] -= (image.shape[3] - shape[1] * r) / 2  # x padding
                    box_output[:, [1, 3]] -= (image.shape[2] - shape[0] * r) / 2  # y padding
                    box_output[:, :4] /= r

                    box_output[:, 0].clamp_(0, shape[1])  # x
                    box_output[:, 1].clamp_(0, shape[0])  # y
                    box_output[:, 2].clamp_(0, shape[1])  # x
                    box_output[:, 3].clamp_(0, shape[0])  # y

                    kps_output[..., 0] -= (image.shape[3] - shape[1] * r) / 2  # x padding
                    kps_output[..., 1] -= (image.shape[2] - shape[0] * r) / 2  # y padding
                    kps_output[..., 0] /= r
                    kps_output[..., 1] /= r
                    kps_output[..., 0].clamp_(0, shape[1])  # x
                    kps_output[..., 1].clamp_(0, shape[0])  # y
                    if VISUALIZE:
                        for box in box_output:
                            box = box.cpu().numpy()
                            x1, y1, x2, y2, score, index = box
                            cv2.rectangle(frame,
                                        (int(x1), int(y1)),
                                        (int(x2), int(y2)),
                                        (0, 255, 0), 2)
                            
                        for kpt in reversed(kps_output):
                            for i, k in enumerate(kpt):
                                color_k = [int(x) for x in kpt_color[i]]
                                x_coord, y_coord = k[0], k[1]
                                if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                                    if len(k) == 3:
                                        conf = k[2]
                                        if conf < 0.5:
                                            continue
                                    cv2.circle(frame,
                                            (int(x_coord), int(y_coord)),
                                            5, color_k, -1, lineType=cv2.LINE_AA)
                                    
                            for i, sk in enumerate(skeleton):
                                pos1 = (int(kpt[(sk[0] - 1), 0]), int(kpt[(sk[0] - 1), 1]))
                                pos2 = (int(kpt[(sk[1] - 1), 0]), int(kpt[(sk[1] - 1), 1]))
                                if kpt.shape[-1] == 3:
                                    conf1 = kpt[(sk[0] - 1), 2]
                                    conf2 = kpt[(sk[1] - 1), 2]
                                    if conf1 < 0.5 or conf2 < 0.5:
                                        continue
                                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                                    continue
                                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                                    continue
                                cv2.line(frame,
                                        pos1, pos2,
                                        [int(x) for x in limb_color[i]],
                                        thickness=2, lineType=cv2.LINE_AA)
                        cv2.imshow('Frame', frame)
                        cv2.waitKey(1)
                        
                    else:
                        print(f"KPS_OUTPUT : {kps_output}")
                # # Press Q on keyboard to  exit
                # if cv2.waitKey(25) & 0xFF == ord('q'):
                #     break
            
                t2 = time.perf_counter()
                fps = round(1/(t2 - t1))
                print(f'FPS : {fps}')
                    
            # Break the loop
            else:
                break
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    except Exception as e:
        print(f"Get Error : {e}")  
      
    camera.release()
    cv2.destroyAllWindows()
    print("All Resources released")


def profile(args, params):
    model = nn.yolo_v8_n(len(params['names']))
    shape = (1, 3, args.input_size, args.input_size)

    model.eval()
    model(torch.zeros(shape))
    params = sum(p.numel() for p in model.parameters())
    if args.local_rank == 0:
        print(f'Number of parameters: {int(params)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=input_size, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--demo', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))

    if args.world_size > 1:
        torch.cuda.set_device(device=-1)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')


    util.setup_seed()
    util.setup_multi_processes()

    # with open(yaml_file, errors='ignore') as f:
    #     params = yaml.safe_load(f)
    #profile(args, params)

    demo(args)


if __name__ == "__main__":
    main()
