from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2
import numpy as np
import os

output_folder = "picture_contour"
os.makedirs(output_folder, exist_ok=True)

input_folder = "picture"
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print("Failed to read the image.")
            continue
        height, width, channels = image.shape
        if width <= 0 or height <= 0:
            print("Invalid image size.")
            continue
        product_segmentation = pipeline(Tasks.product_segmentation, model='damo/cv_F3Net_product-segmentation')
        result_status = product_segmentation({'input_path': image_path})
        contour_mask = result_status[OutputKeys.MASKS]
        # 调整轮廓掩码的大小和格式
        if contour_mask is None:
            print("Contour mask is None.")
        else :
            contour_mask = cv2.resize(contour_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            contour_mask = (contour_mask > 0).astype(np.uint8) * 255
        # 将轮廓掩码应用于原图像上
            contour_image = cv2.bitwise_and(image, image, mask=contour_mask)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_contour.jpg")
            cv2.imwrite(output_path, contour_image)
