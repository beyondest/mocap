# Model Information

- **ONNX Version**: v6
- **Producer**: pytorch 2.4.1
- **Version**: 0
- **Imports**: ai.onnx v11

## Graph
- **Name**: main_graph

## Inputs
- **Name**: input
  - **Tensor**: float32[1, 3, 480, 640]

## Outputs
- **Name**: output
  - **Tensor**: float32[1, 56, 6300]
  - ** Dimensions**:
    - 1: batch size
    - 56: 5 + 17 * 3, 5 is bbox (x, y, w, h, score), 17 is 17 keypoints (x, y, score)
    - 6300: number of results
