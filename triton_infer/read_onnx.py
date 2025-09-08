import onnx, onnx.shape_inference as si

m = onnx.load("/workspace/OCR_Triton_Paddle/triton_infer/model_repository/text_detection/1/model.onnx")
try:
    m = si.infer_shapes(m)
except Exception:
    pass
print("DETECTION outputs:", [(o.name, [d.dim_value or d.dim_param
       for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])

m2 = onnx.load("/workspace/OCR_Triton_Paddle/triton_infer/onnx/model.onnx")
try:
    m2 = si.infer_shapes(m2)
except Exception:
    pass
print("RECOG outputs:", [(o.name, [d.dim_value or d.dim_param
       for d in o.type.tensor_type.shape.dim]) for o in m2.graph.output])

