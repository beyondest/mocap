import cv2
import numpy as np

import cv2
import numpy as np

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



frame = cv2.imread("bus.jpg")
frame = cv2.resize(frame,(97,1000))
cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
image = resize_image(frame,(753,231))
print(image.shape)
