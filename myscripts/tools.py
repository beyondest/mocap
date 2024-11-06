import numpy as np
import onnx
import onnxruntime
import os
import cv2


class Onnx_Engine:
    class Standard_Data:
        def __init__(self) -> None:
            self.result = 0
            
        def save_results(self,results:np.ndarray):
            self.result = results
        
        
    def __init__(self,
                 filename:str,
                 if_offline:bool = False,
                 ) -> None:
        """Config here if you wang more

        Args:
            filename (_type_): _description_
        """
        if os.path.splitext(filename)[-1] !='.onnx':
            raise TypeError(f"onnx file load failed, path not end with onnx :{filename}")

        custom_session_options = onnxruntime.SessionOptions()
        
        if if_offline:
            custom_session_options.optimized_model_filepath = filename
            
        custom_session_options.enable_profiling = False          #enable or disable profiling of model
        #custom_session_options.execution_mode =onnxruntime.ExecutionMode.ORT_PARALLEL       #ORT_PARALLEL | ORT_SEQUENTIAL
        #custom_session_options.inter_op_num_threads = 2                                     #default is 0
        #custom_session_options.intra_op_num_threads = 2                                     #default is 0
        custom_session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL # DISABLE_ALL |ENABLE_BASIC |ENABLE_EXTENDED |ENABLE_ALL
        custom_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']       # if gpu, cuda first, or will use cpu
        self.user_data = self.Standard_Data()  
        self.ort_session = onnxruntime.InferenceSession(filename,
                                                        sess_options=custom_session_options,
                                                        providers=custom_providers)

    def standard_callback(results:np.ndarray, user_data:Standard_Data,error:str):
        if error:
            print(error)
        else:
            user_data.save_results(results)
    
    
    def run(self,output_nodes_name_list:list|None,input_nodes_name_to_npvalue:dict)->list:
        """@timing\n
        Notice that input value must be numpy value
        Args:
            output_nodes_name_list (list | None): _description_
            input_nodes_name_to_npvalue (dict): _description_

        Returns:
            list: [node1_output,node2_output,...]
        """
        
        return self.ort_session.run(output_nodes_name_list,input_nodes_name_to_npvalue)
    
    
 
            
    def run_asyc(self,output_nodes_name_list:list|None,input_nodes_name_to_npvalue:dict):
        """Will process output of model in callback , config callback here

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

def resize_image(frame, target_size=(480, 640), stride=32):
    image = frame.copy()
    target_height, target_width = target_size
    original_shape = image.shape[:2]  # 当前形状 [高度, 宽度]

    # 计算缩放比例，使图像缩放后适配目标大小，同时保持比例
    r = min(target_width / original_shape[1], target_height / original_shape[0])
    new_size = (int(original_shape[1] * r), int(original_shape[0] * r))  # 缩放后的尺寸 (宽度, 高度)

    # 缩放图像
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    # 计算添加边框的尺寸，以达到目标大小
    delta_w = target_width - new_size[0]
    delta_h = target_height - new_size[1]

    # 如果目标尺寸与步长不对齐，对固定的目标尺寸进行步长约束调整
    if target_width % stride != 0:
        delta_w += stride - (target_width % stride)
    if target_height % stride != 0:
        delta_h += stride - (target_height % stride)

    # 计算边框，使其在左右、上下均匀分布
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2

    # 添加边框
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 转换为 CHW 格式并调整颜色顺序 BGR -> RGB
    image = image.transpose((2, 0, 1))[::-1]
    image = np.ascontiguousarray(image)

    return image



class GetKeypoint():
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16

