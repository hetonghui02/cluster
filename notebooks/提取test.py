from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2
import os
import numpy as np

image = cv2.imread("")
if image is None:
    print("Failed to read the image.")
height, width, channels = image.shape
if width <= 0 or height <= 0:
    print("Invalid image size.")
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(image.shape[0])
product_segmentation = pipeline(Tasks.product_segmentation, model='damo/cv_F3Net_product-segmentation')
result_status = product_segmentation({'input_path': "1.jpg"})
contour_mask = result_status[OutputKeys.MASKS]
if contour_mask is None:
    print("Contour mask is None.")
else:
    contour_mask = cv2.resize(contour_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    contour_mask = (contour_mask > 0).astype(np.uint8) * 255
    # 将轮廓掩码应用于原图像上
    contour_image = cv2.bitwise_and(image, image, mask=contour_mask)
    cv2.imshow("Contour Mask",contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
