import pandas as pd
import re

# 读取Excel文件
df = pd.read_excel("item_pool.xlsx")

# 清除特殊符号并规范化文本
cleaned_titles = []
for title in df['item_name']:
    # 清除特殊符号，仅保留文字、数字、英文
    cleaned_title = re.sub(r"[^\w\d\s]", "", title)
    # 将英文字母改为小写
    cleaned_titles.append(cleaned_title)

# 创建新的DataFrame保存清理后的文本
cleaned_df = pd.DataFrame({'item_name': cleaned_titles})

# 输出清理后的结果
# print(cleaned_df)
cleaned_df.to_excel(excel_writer= r"item_pool_update.xlsx")