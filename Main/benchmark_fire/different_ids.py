import pandas as pd

# 读取csv1和csv2
csv1 = pd.read_csv('output_ssd.csv')
csv2 = pd.read_csv('output_yolo.csv')

# 获取csv1和csv2中第一个值的集合
csv1_values = set(csv1.iloc[:, 0])
csv2_values = set(csv2.iloc[:, 0])

# 找到两个文件中都有的值
common_values = csv1_values.intersection(csv2_values)

# 根据共同的值过滤csv1和csv2
csv1_filtered = csv1[csv1.iloc[:, 0].isin(common_values)]
csv2_filtered = csv2[csv2.iloc[:, 0].isin(common_values)]

# 确保从第2行开始保留原文件的5位数字（假设原文件中的第一个值已经是五位数字）
csv1_filtered.iloc[0:, 0] = csv1_filtered.iloc[0:, 0].astype(str).str.zfill(5)
csv2_filtered.iloc[0:, 0] = csv2_filtered.iloc[0:, 0].astype(str).str.zfill(5)

# 将过滤后的结果保存为新的文件
csv1_filtered.to_csv('output_ssd2.csv', index=False)
csv2_filtered.to_csv('output_yolo2.csv', index=False)

print("修改后的文件已保存为output_ssd2.csv和output_yolo2.csv")
