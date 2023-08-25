import os
import pandas as pd
import requests

# 读取 Excel 文件
df = pd.read_excel("item_pool_classification.xlsx")

# 创建保存图片的文件夹
picture_folder = "picture"
if not os.path.exists(picture_folder):
    os.makedirs(picture_folder)

# 遍历每一行，下载并保存图片
for index, row in df.iterrows():
    img_url = row['img_head']
    item_name = row['spec']

    # 获取图片的扩展名
    ext = os.path.splitext(img_url)[1]
    if ext.lower() not in ['.jpg', '.jpeg']:
        ext = '.jpg'

    # 拼接保存图片的文件名
    filename = os.path.join(picture_folder, f"{item_name}{ext}")

    # 下载并保存图片
    try:
        response = requests.get(img_url)
        response.raise_for_status()

        with open(filename, 'wb') as f:
            f.write(response.content)

        print(f"Saved image: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {img_url}, {e}")
