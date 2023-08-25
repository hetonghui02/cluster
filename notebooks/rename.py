import os

# 指定文件夹路径
folder_path = "picture"

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, filename)):
        # 获取旧文件名的完整路径
        old_file_path = os.path.join(folder_path, filename)

        # 删除文件名中的空格
        new_filename = filename.replace(" ", "")

        # 获取新文件名的完整路径
        new_file_path = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
