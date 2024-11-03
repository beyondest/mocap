import numpy as np
import onnx
import onnxruntime
import os

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

