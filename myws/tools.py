import numpy as np
import onnx
import onnxruntime
import os
import cv2
import numpy as np
import torchvision
import time
import torch
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
except ImportError as e:
    print(e)
    print("Please install tensorrt and pycuda")


class TRT_Engine_2:
    class HostDeviceMem(object):
    
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

        
    def __init__(self,
                 trt_path:str,
                 max_batchsize:int = 10) -> None:
        """Only support single input binding and single output binding

        Warning: idx_to_max_batchsize must include all binding index of engine, or will raise error\n
                        include input and output binding index of engine\n
        """
        assert os.path.exists(trt_path), f"Tensorrt engine file not found: {trt_path}"
        
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        self.stream = cuda.Stream()
        with open(trt_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.input_idx_to_binding_idx = {}
        
        for i in range(max_batchsize):
            inputs,outputs,bindings = self.allocate_buffers(i+1)
            self.inputs.append(inputs)
            self.outputs.append(outputs)
            self.bindings.append(bindings)
        
        
        
    def allocate_buffers(self,batchsize:int=1):
        """
        Args:
            batchsize (int, optional):  Defaults to 1.

        Returns:
            list: [inputs,outputs,bindings]
        """
        inputs = []
        outputs = []
        bindings = []
        
        for index in range(len(self.engine)):
            
            shape = self.engine.get_binding_shape(self.engine[index])
            
            if shape[0] == -1:
                shape = (batchsize, *shape[1:])
            
            dtype = trt.nptype(self.engine.get_binding_dtype(self.engine[index]))
            size = trt.volume(shape) 
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(self.engine[index]):
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
                self.input_idx_to_binding_idx.update({len(inputs)-1 : index})
                
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))
        
        return [inputs,outputs,bindings]
                
    def run(self,input_idx_o_npvalue:dict)->list[np.ndarray]:
        """
        Only support all nodes are in same batchsize  
        Warning: input_idx_o_npvalue must be {input_index:np_array,...}
                 e.g.:
                    {0:np.array([1,2,3,4,5,6,7,8,9,10]),
                     3:np.array([1,2,3,4,5,6,7,8,9,10])}\n
                     and 0,3 must be INPUT binding index of engine
        Returns:
            list: [output_node0_output,output_node1_output,..]
        """
        outlist = []
        
        batchsize = input_idx_o_npvalue[0].shape[0]
        
        for input_index in input_idx_o_npvalue:
            self.context.set_binding_shape(self.input_idx_to_binding_idx[input_index], (batchsize, *self.engine.get_binding_shape(self.engine[input_index])[1:]))
            np.copyto(self.inputs[batchsize-1][input_index].host, input_idx_o_npvalue[input_index].ravel())
            cuda.memcpy_htod_async(self.inputs[batchsize-1][input_index].device, self.inputs[batchsize-1][input_index].host, self.stream)
        
        self.context.execute_async_v2(bindings=self.bindings[batchsize-1], stream_handle=self.stream.handle)
        
        for output_index in range(len(self.outputs[batchsize-1])):
            cuda.memcpy_dtoh_async(self.outputs[batchsize-1][output_index].host, self.outputs[batchsize-1][output_index].device, self.stream)

        self.stream.synchronize()
        
        for output_index in  range(len(self.outputs[batchsize-1])):
            outlist.append(self.outputs[batchsize-1][output_index].host)
       
        return outlist
    
class Onnx_Engine:
    class Standard_Data:
        def __init__(self) -> None:
            self.result = 0
            
        def save_results(self, results: np.ndarray):
            self.result = results
        
        
    def __init__(self,
                 filename: str,
                 if_offline: bool = False,
                 ) -> None:
        """Config here if you want more

        Args:
            filename (_type_): _description_
        """
        if os.path.splitext(filename)[-1] != '.onnx':
            raise TypeError(f"onnx file load failed, path not end with onnx :{filename}")

        custom_session_options = onnxruntime.SessionOptions()
        
        if if_offline:
            custom_session_options.optimized_model_filepath = filename
            
        custom_session_options.enable_profiling = False  # enable or disable profiling of model
        # custom_session_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL       # ORT_PARALLEL | ORT_SEQUENTIAL
        # custom_session_options.inter_op_num_threads = 2                                     # default is 0
        # custom_session_options.intra_op_num_threads = 2                                     # default is 0
        custom_session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL  # DISABLE_ALL | ENABLE_BASIC | ENABLE_EXTENDED | ENABLE_ALL
        custom_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # if gpu, cuda first, or will use cpu
        self.user_data = self.Standard_Data()  
        self.ort_session = onnxruntime.InferenceSession(filename,
                                                        sess_options=custom_session_options,
                                                        providers=custom_providers)

    def standard_callback(results: np.ndarray, user_data: Standard_Data, error: str):
        if error:
            print(error)
        else:
            user_data.save_results(results)
    
    
    def run(self, output_nodes_name_list: list | None, input_nodes_name_to_npvalue: dict) -> list:
        """@timing\n
        Notice that input value must be numpy value
        Args:
            output_nodes_name_list (list | None): _description_
            input_nodes_name_to_npvalue (dict): _description_

        Returns:
            list: [node1_output, node2_output, ...]
        """
        
        return self.ort_session.run(output_nodes_name_list, input_nodes_name_to_npvalue)
    
    
 
            
    def run_async(self, output_nodes_name_list: list | None, input_nodes_name_to_npvalue: dict):
        """Will process output of model in callback, config callback here

        Args:
            output_nodes_name_list (list | None): _description_
            input_nodes_name_to_npvalue (dict): _description_

        Returns:
            None
        """
        self.ort_session.run_async(output_nodes_name_list,
                                    input_nodes_name_to_npvalue,
                                    callback=self.standard_callback,
                                    user_data=self.user_data)
        return None

def resize_image(image:np.ndarray, target_size=(480, 640), stride=32, if_use_stride = False)->np.ndarray:
    
    target_height, target_width = target_size
    original_shape = image.shape[:2]  # Current shape [height, width]

    # Calculate scaling ratio to fit the target size while maintaining aspect ratio
    r = min(target_width / original_shape[1], target_height / original_shape[0])
    new_size = (int(original_shape[1] * r), int(original_shape[0] * r))  # Scaled size (width, height)

    # Scale the image
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    # Calculate the size of the border to achieve the target size
    delta_w = target_width - new_size[0]
    delta_h = target_height - new_size[1]

    # If the target size is not aligned with the stride, adjust the fixed target size accordingly
    if if_use_stride:
        if target_width % stride != 0:
            delta_w += stride - (target_width % stride)
        if target_height % stride != 0:
            delta_h += stride - (target_height % stride)

    # Calculate borders for even distribution on left/right and top/bottom
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2

    # Add borders
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Convert to CHW format and adjust color order BGR -> RGB
    image = image.transpose((2, 0, 1))[::-1]
    image = np.ascontiguousarray(image)

    return image



def visualize_detections(frame:np.ndarray,
                         box_output:np.ndarray, 
                         kps_output:np.ndarray,
                         kpt_color:list,
                         skeleton:list,
                         limb_color:list):
    """
    Visualizes conf > 0.5 bounding boxes and keypoints on the given frame.

    Parameters:
    - frame: The image frame on which to draw the detections.
    - box_output: List of bounding boxes (x1, y1, x2, y2, score, index).
    - kps_output: List of keypoints for each detected object.
    - shape: The shape of the input frame, used for boundary checks.
    - kpt_color: Colors for keypoints.
    - skeleton: Connections between keypoints to draw skeleton lines.
    - limb_color: Colors for the limbs.
    - visualize: A flag to enable or disable visualization.

    Returns:
    - None
    """
    
    shape = frame.shape[:2]
    # Draw bounding boxes
    for box in box_output:
        x1, y1, x2, y2, score, index = box
        cv2.rectangle(frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0), 2)

    # Draw keypoints
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

        # Draw skeleton lines
        for i, sk in enumerate(skeleton):
            pos1 = (int(kpt[(sk[0]), 0]), int(kpt[(sk[0]), 1]))
            pos2 = (int(kpt[(sk[1]), 0]), int(kpt[(sk[1]), 1]))
            if kpt.shape[-1] == 3:
                conf1 = kpt[(sk[0]), 2]
                conf2 = kpt[(sk[1]), 2]
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

    # Display the frame
    cv2.imshow('Frame', frame)
    cv2.waitKey(1)
def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(outputs, conf_threshold, iou_threshold, nc)->list:
    """
    Perform Non-Maximum Suppression (NMS) on inference results.
    # Arguments
        outputs:  Model output [1, 5 + 51, 6300] for mocap.
        conf_threshold:  Object confidence threshold (float).
        iou_threshold:  IoU threshold for NMS (float).
        nc:  Number of classes (int), 1 for mocap, only human class.
    # Returns
        List: [Batch0, Batch1,..., BatchN],Tensor Batchi.shape = (bbox_num, 6 + 51) for mocap, 6 = 4 + 1 + 1, 4 is bbox, 1 is conf, 1 is cls human, 51 is mask
    """
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    bs = outputs.shape[0]  # batch size , 1 for mocap
    nc = nc or (outputs.shape[1] - 4)  # number of classes , 1 for mocap, refer to human
    nm = outputs.shape[1] - nc - 4 # number of masks, 56 - 1 - 4 = 51 for mocap
    mi = 4 + nc  # mask start index, 4 + 1 = 5 for mocap
    xc = outputs[:, 4:mi].amax(1) > conf_threshold  # candidates, xc.shape = (bs, results), (1, 6300) for mocap, full of True or False

    # Settings
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=outputs.device)] * bs # blank output for nms_results, [A] * 2 = [A, A], A.shape = (0, 6 + 51) for mocap
    for index, x in enumerate(outputs):  # image index, image inference, index is batchindex for mocap , which means 0
        x = x.transpose(0, -1)[xc[index]]  # x.transpose(0, -1) is (6300, 56) for mocap, this will select the candidates which is true in xc[index]

        # If none remain process next image, means all candidates are false
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1) # split the tensor into 3 parts in axis 1, box, cls, mask, box.shape = (candidates_num, 4), cls.shape = (candidates_num, 1), mask.shape = (candidates_num, 51)
        box = wh2xy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2), xy1 is top-left, xy2 is bottom-right
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)   # conf.shape = (candidates_num, 1), j.shape = (candidates_num, 1), j is the index of the best class, 0 for mocap
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_threshold] # dulpicated filter by conf_threshold, it has already been filtered by conf_threshold in xc[index]
                                                                                           # x.shape = (candidates_num, 6 + 51) for mocap, 4 for box, 1 for conf, 1 for cls, 51 for mask

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by conf of class (human for mocap) and remove excess boxes(only keep max_nms candidates)

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes, for mocap, human class is 0, so cls * max_wh = 0, offset is 0. for other situation, offset makes iou wont remove different class boxes in same position
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # remove overlap boxes, i is the index of the no_overlap boxes in different classes
        i = i[:max_det]  # limit detections

        output[index] = x[i] # save results in to batch index of output, for mocap, index is 0
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def pose_estimation_postprocess(outputs:torch.Tensor, image:np.ndarray|torch.Tensor, frame:np.ndarray, model)->tuple[torch.Tensor, torch.Tensor]:
    """
    Get Box and Kpts, will remap xy to original frame size.
    # Arguments
        outputs:  Model output [1, 5 + 51, 6300] for mocap.
        image:  The input image for inference.
        frame:  The original image frame, such as the video frame.
        model:  The model used for inference.
    # Returns
        box_output:  torch.Tensor, shape = (bbox_num, 6), (x1, y1, x2, y2, score, index).
        kps_output:  torch.Tensor, shape = (bbox_num, 17, 3), (x, y, score).
    # Note
        May change orignal outputs, because of removing `output = output.clone()`
    """
    shape = frame.shape[:2]
    for output in outputs:
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


    return box_output, kps_output

def maxmium_performance():
    """
    Setup multi-processing environment variables for maximum performance.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method; adjust based on your system
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('spawn', force=True) # Experiment with 'spawn' or 'forkserver'

    # optionally enable OpenCV multithreading (default is to use all available cores)
    # cv2.setNumThreads() can be called here with a specific value or left out for default behavior

    # remove the limits on OMP and MKL threads to use all available cores
    environ.pop('OMP_NUM_THREADS', None)  # Remove if it exists
    environ.pop('MKL_NUM_THREADS', None)  # Remove if it exists



class Palettes():
    class BGR:
        RED = (0, 0, 255)
        GREEN = (0, 255, 0)
        BLUE = (255, 0, 0)
        YELLOW = (0, 255, 255)
        PURPLE = (255, 0, 255)
    class RGB:
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        YELLOW = (255, 255, 0)
        PURPLE = (255, 0, 255)
            
class Kpt():

    class Yolov8():
        # Keypoint indices
        NOSE = 0
        LEFT_EYE = 1            # No mapping in H36M
        RIGHT_EYE = 2           # No mapping in H36M
        LEFT_EAR = 3            # No mapping in H36M
        RIGHT_EAR = 4           # No mapping in H36M
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_ELBOW = 7
        RIGHT_ELBOW = 8
        LEFT_WRIST = 9
        RIGHT_WRIST = 10
        LEFT_HIP = 11
        RIGHT_HIP = 12
        LEFT_KNEE = 13
        RIGHT_KNEE = 14
        LEFT_ANKLE = 15
        RIGHT_ANKLE = 16
        
        
        # Yolov8 19 bones
        skeleton = [
        [LEFT_ANKLE, LEFT_KNEE],
        [LEFT_KNEE, LEFT_HIP],
        [RIGHT_ANKLE, RIGHT_KNEE],
        [RIGHT_KNEE, RIGHT_HIP],
        [LEFT_HIP, RIGHT_HIP],
        [LEFT_SHOULDER, LEFT_HIP],
        [RIGHT_SHOULDER, RIGHT_HIP],
        [LEFT_SHOULDER, RIGHT_SHOULDER],
        [LEFT_SHOULDER, LEFT_ELBOW],
        [RIGHT_SHOULDER, RIGHT_ELBOW],
        [LEFT_ELBOW, LEFT_WRIST],
        [RIGHT_ELBOW, RIGHT_WRIST],
        [LEFT_EYE, RIGHT_EYE],
        [NOSE, RIGHT_EYE],
        [NOSE, LEFT_EYE],
        [RIGHT_EYE, RIGHT_EAR],
        [LEFT_EYE, LEFT_EAR],
        [LEFT_EAR, LEFT_SHOULDER],
        [RIGHT_EAR, RIGHT_SHOULDER]]
        
        kpt_color = np.array([Palettes.BGR.RED for _ in range(17)])
                             
        limb_color = np.array([Palettes.BGR.RED for _ in range(19)])

    class H36M():
        # Keypoint indices
        PELVIS_EXTRA = 0    # No mapping in Yolov8
        LEFT_HIP_EXTRA = 1
        LEFT_KNEE = 2
        LEFT_ANKLE = 3
        RIGHT_HIP_EXTRA = 4
        RIGHT_KNEE = 5
        RIGHT_ANKLE = 6
        SPINE_EXTRA = 7     # No mapping in Yolov8
        NECK_EXTRA = 8      # No mapping in Yolov8
        HEAD_EXTRA = 9
        HEADTOP = 10        # No mapping in Yolov8
        LEFT_SHOULDER = 11
        LEFT_ELBOW = 12
        LEFT_WRIST = 13
        RIGHT_SHOULDER = 14
        RIGHT_ELBOW = 15
        RIGHT_WRIST = 16
        # H36M 16 bones
        skeleton = [
        [PELVIS_EXTRA, LEFT_HIP_EXTRA],
        [LEFT_HIP_EXTRA, LEFT_KNEE],
        [LEFT_KNEE, LEFT_ANKLE],
        [PELVIS_EXTRA, RIGHT_HIP_EXTRA],
        [RIGHT_HIP_EXTRA, RIGHT_KNEE],
        [RIGHT_KNEE, RIGHT_ANKLE],
        [PELVIS_EXTRA, SPINE_EXTRA],
        [SPINE_EXTRA, NECK_EXTRA],
        [NECK_EXTRA, HEAD_EXTRA],
        [HEAD_EXTRA, HEADTOP],
        [NECK_EXTRA, LEFT_SHOULDER],
        [LEFT_SHOULDER, LEFT_ELBOW],
        [LEFT_ELBOW, LEFT_WRIST],
        [NECK_EXTRA, RIGHT_SHOULDER],
        [RIGHT_SHOULDER, RIGHT_ELBOW],
        [RIGHT_ELBOW, RIGHT_WRIST]]

        kpt_color = np.array([Palettes.BGR.RED for _ in range(17)])

        limb_color = np.array([Palettes.BGR.RED for _ in range(16)])


    yolo_to_h36m = {
        Yolov8.NOSE: H36M.HEAD_EXTRA,    # NOSE -> head_extra
        Yolov8.LEFT_SHOULDER: H36M.LEFT_SHOULDER,  # LEFT_SHOULDER -> left_shoulder
        Yolov8.RIGHT_SHOULDER: H36M.RIGHT_SHOULDER,  # RIGHT_SHOULDER -> right_shoulder
        Yolov8.LEFT_ELBOW: H36M.LEFT_ELBOW,  # LEFT_ELBOW -> left_elbow
        Yolov8.RIGHT_ELBOW: H36M.RIGHT_ELBOW,  # RIGHT_ELBOW -> right_elbow
        Yolov8.LEFT_WRIST: H36M.LEFT_WRIST,  # LEFT_WRIST -> left_wrist
        Yolov8.RIGHT_WRIST: H36M.RIGHT_WRIST,  # RIGHT_WRIST -> right_wrist
        Yolov8.LEFT_HIP: H36M.LEFT_HIP_EXTRA,  # LEFT_HIP -> left_hip_extra
        Yolov8.RIGHT_HIP: H36M.RIGHT_HIP_EXTRA,  # RIGHT_HIP -> right_hip_extra
        Yolov8.LEFT_KNEE: H36M.LEFT_KNEE,  # LEFT_KNEE -> left_knee
        Yolov8.RIGHT_KNEE: H36M.RIGHT_KNEE,  # RIGHT_KNEE -> right_knee
        Yolov8.LEFT_ANKLE: H36M.LEFT_ANKLE,  # LEFT_ANKLE -> left_ankle
        Yolov8.RIGHT_ANKLE: H36M.RIGHT_ANKLE   # RIGHT_ANKLE -> right_ankle
    }
    h36m_to_yolo = {v: k for k, v in yolo_to_h36m.items()}
 
    @staticmethod
    def tran_h36m_to_yolo(keypoints: np.ndarray):
        """
        Map Human3.6M keypoints to yolo8 keypoints.  
        Notice that dim 3 is confidence
        # Arguments
            keypoints:  Shape: (B, 17, 3), 3 is (x, y, conf)
        # Returns
            yolo_keypoints:  List of keypoints for each detected object in yolo8 format.
        """
        yolo_keypoints = []
        shape = keypoints.shape[1:]
        for kpt in keypoints:
            mapped_keypoints = np.zeros(shape)  # Initialize mapped keypoints array
            for h36m_id, yolo_id in Kpt.h36m_to_yolo.items():
                mapped_keypoints[yolo_id] = kpt[h36m_id]
            mapped_keypoints[Kpt.Yolov8.LEFT_EYE] = mapped_keypoints[Kpt.Yolov8.NOSE]
            mapped_keypoints[Kpt.Yolov8.RIGHT_EYE] = mapped_keypoints[Kpt.Yolov8.NOSE]
            mapped_keypoints[Kpt.Yolov8.LEFT_EAR] = mapped_keypoints[Kpt.Yolov8.NOSE]
            mapped_keypoints[Kpt.Yolov8.RIGHT_EAR] = mapped_keypoints[Kpt.Yolov8.NOSE]
            yolo_keypoints.append(mapped_keypoints)  # Append the mapped keypoints for the current batch
        return np.array(yolo_keypoints)
            
    @staticmethod
    def tran_yolo_to_h36m(keypoints: np.ndarray):
        """
        Map yolo8 keypoints to Human3.6M keypoints.  
        Notice that dim 3 is confidence
        # Arguments
            keypoints:  Shape: (B, 17, 3), 3 is (x, y, conf)
        # Returns
            h36m_keypoints:  List of keypoints for each detected object in h36m format.
        """
        h36m_keypoints = []
        shape = keypoints.shape[1:]
        for kpt in keypoints:
            mapped_keypoints = np.zeros(shape)  # Initialize mapped keypoints array
            for yolo_id, h36m_id in Kpt.yolo_to_h36m.items():
                mapped_keypoints[h36m_id] = kpt[yolo_id]
            mapped_keypoints[Kpt.H36M.PELVIS_EXTRA] = (mapped_keypoints[Kpt.H36M.LEFT_HIP_EXTRA] + mapped_keypoints[Kpt.H36M.RIGHT_HIP_EXTRA]) / 2 
            mid_shoulder = (mapped_keypoints[Kpt.H36M.LEFT_SHOULDER] + mapped_keypoints[Kpt.H36M.RIGHT_SHOULDER]) / 2
            mapped_keypoints[Kpt.H36M.NECK_EXTRA] = (mapped_keypoints[Kpt.H36M.HEAD_EXTRA] + mid_shoulder) / 2  
            mapped_keypoints[Kpt.H36M.SPINE_EXTRA] = (mapped_keypoints[Kpt.H36M.PELVIS_EXTRA] + mid_shoulder) / 2
            mapped_keypoints[Kpt.H36M.HEADTOP] = mapped_keypoints[Kpt.H36M.HEAD_EXTRA] + mapped_keypoints[Kpt.H36M.HEAD_EXTRA] - mapped_keypoints[Kpt.H36M.NECK_EXTRA]
            h36m_keypoints.append(mapped_keypoints)  # Append the mapped keypoints for the current batch
        return np.array(h36m_keypoints)
    
    
    
    
    







