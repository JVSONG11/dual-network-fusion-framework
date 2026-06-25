import os

# 文件路径
txt1_path = './ssd_1.txt'  # txt1的路径
txt2_path = './ssd_2.txt'  # txt2的路径
txt3_path = './ssd.txt'  # 最终生成的txt3路径
output_folder = './1txt'  # 保存生成的txt文件的文件夹

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取txt1和txt2并逐行合并，生成txt3，每个值使用逗号隔开
with open(txt1_path, 'r') as file1, open(txt2_path, 'r') as file2, open(txt3_path, 'w') as file3:
    txt1_lines = file1.readlines()
    txt2_lines = file2.readlines()

    # 确保两个txt文件的行数相同
    assert len(txt1_lines) == len(txt2_lines), "txt1 和 txt2 的行数不同！"

    # 合并每一行并写入txt3
    for line1, line2 in zip(txt1_lines, txt2_lines):
        merged_line = line1.strip() + ',' + line2.strip() + '\n'
        file3.write(merged_line)

# 读取txt3并生成单独的txt文件
with open(txt3_path, 'r') as file3:
    txt3_lines = file3.readlines()

    # 用于存储按第一个值命名的txt内容
    file_contents = {}

    for line in txt3_lines:
        split_line = line.strip().split(',')

        # 获取第一个值作为文件名，提取需要的值（第2个值和第4-7个值）
        file_name = split_line[0]
        values_to_keep = [split_line[1]] + split_line[3:7] + [split_line[2]]

        # 将这些值用空格连接
        content_line = ' '.join(values_to_keep) + '\n'

        # 如果文件名已存在，追加内容；否则新建一个文件内容列表
        if file_name not in file_contents:
            file_contents[file_name] = []
        file_contents[file_name].append(content_line)

    # 将内容写入对应的txt文件
    for file_name, contents in file_contents.items():
        output_file_path = os.path.join(output_folder, f"{file_name}.txt")
        with open(output_file_path, 'w') as output_file:
            output_file.writelines(contents)

print("任务完成，文件已生成。")



