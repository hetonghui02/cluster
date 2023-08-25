import pandas as pd

# 读取 Excel 文件
df = pd.read_excel("item_classification.xlsx")

# 删除单元格中的方括号、双引号和单引号
df = df.replace({'\[': '', '\]': '', '\"': '', "\'": ''}, regex=True)

# 保存到新的 Excel 文件
df.to_excel("item_pool_classification.xlsx", index=False)
