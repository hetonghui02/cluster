import shutil
import os

source_folder = "picture_spec_contour/clusters"
destination_folder = "picture_spec_contour"

# 遍历 source_folder 下的所有文件夹
for folder_name in os.listdir(source_folder):
    folder_path = os.path.join(source_folder, folder_name)

    # 检查路径是否为文件夹
    if os.path.isdir(folder_path):
        # 遍历文件夹下的所有文件
        for filename in os.listdir(folder_path):
            source_path = os.path.join(folder_path, filename)
            destination_path = os.path.join(destination_folder, filename)

            # 复制文件到目标文件夹
            shutil.copy2(source_path, destination_path)
