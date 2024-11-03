import onnx
import torch
import sys
sys.path.append('../')



def save_model_to_onnx(
                           model:torch.nn.Module,
                           output_abs_path:str,
                           dummy_input:torch.Tensor,
                           trained_weights_path:str|None = None,
                           input_names = ['input'],
                           output_names = ['output'],
                           if_dynamic_batch_size:bool = True,
                           dynamic_axe_name = 'batch_size',
                           dynamic_axe_input_id = 0,
                           dynamic_axe_output_id = 0,
                           opt_version:int=10
                           ):


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if if_dynamic_batch_size == False:
        dynamic_axes = None
    else:
        dynamic_axes = {input_names[0]: {dynamic_axe_input_id: dynamic_axe_name}, 
                        output_names[0]: {dynamic_axe_output_id: dynamic_axe_name}}

    if trained_weights_path is not None:
        model.load_state_dict(torch.load(trained_weights_path,map_location=device))

    model.eval()
    # quatized model, but notice not all platforms onnx run will support this, so you need to add ATEN_FALLBACK 
    #q_model = quantize_dynamic(model,dtype=torch.qint8)

    torch.onnx.export(model=model,                          #model to trans
                    args=dummy_input,                       #dummy input to infer shape
                    f=output_abs_path,                      #output onnx name 
                    verbose=True,                           #if show verbose information in console
                    export_params=True,                     #if save present weights to onnx
                    input_names=input_names,                #input names list,its length depends on how many input your model have
                    output_names=output_names,              #output names list
                    training=torch.onnx.TrainingMode.EVAL,  #EVAL or TRAINING or Preserve(depends on if you specify model.eval or model.train)
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,   #ONNX or ONNX_FALLTROUGH or ONNX_ATEN_FALLBACK  or ONNX_ATEN, ATEN means array tensor library of Pytorch
                                                                                        # fallback to onnx or fallthrough to aten, use aten as default, aten better for torch, but onnx is more compat
                    opset_version=opt_version,                       #7<thix<17
                    do_constant_folding=True,               #True
                    dynamic_axes = dynamic_axes,            #specify which axe is dynamic
                    keep_initializers_as_inputs=False,      #if True, then you can change weights of onnx if model is same   
                    custom_opsets=None                 #custom operation, such as lambda x:abs(x),of course not so simple, you have to register to pytorch if you want to use custom op
                    )                 

    print(f'ori_onnx model saved to {output_abs_path}')

weight_path = '../weights/best.pt'
output_path = '../weights/best.onnx'
dummy_input = torch.randn(1, 3,480, 640)
model = torch.load(weight_path, map_location='cpu')['model'].float()

save_model_to_onnx(model,
                        output_path,
                        dummy_input,
                        trained_weights_path = None,
                        input_names=['input'],
                        output_names=['output'],
                        if_dynamic_batch_size=False,
                        opt_version=11
                        )





