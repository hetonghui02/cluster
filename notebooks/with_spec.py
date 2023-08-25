import pandas as pd
import re

# 读取 Excel 文件
df = pd.read_excel("item_pool_classification.xlsx")

# 将缺失值替换为空字符串
df.fillna('', inplace=True)

# 生成新的 spec 列
df['spec_from_name'] = df['model'] + df['series'] + df['effectivity'] + df['description'] + df['size'] + df['color'] + df['style']

# 删除除了数字、汉字和字母以外的符号
df['spec_from_name' \
   ''] = df['spec'].apply(lambda x: re.sub(r'[^\w\u4e00-\u9fa5]', '', x))

# 保存到新的 Excel 文件
df.to_excel("item_pool_classification.xlsx", index=False)
