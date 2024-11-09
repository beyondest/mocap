import onnx
import torch
import sys
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from myws.tools import save_model_to_onnx


weight_path = 'D:/VS_ws/python/mocap/weights/best.pt'
output_path = 'best.onnx'
dummy_input = torch.randn(1, 3,480, 640)
if torch.cuda.is_available():
    model = torch.load(weight_path, map_location='gpu')['model'].float()
else:
    model = torch.load(weight_path, map_location='cpu')['model'].float()

save_model_to_onnx(model,
                        output_path,
                        dummy_input,
                        trained_weights_path = None,
                        input_names=['input'],
                        output_names=['output'],
                        if_dynamic_batch_size=False,
                        opt_version=20
                        )





