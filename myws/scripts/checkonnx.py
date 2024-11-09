import onnx
onnx_model = onnx.load( 'D:/VS_ws/python/mocap/weights/autogrid20.onnx'
)
onnx.checker.check_model(onnx_model)
