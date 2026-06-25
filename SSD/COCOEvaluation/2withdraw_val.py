import os
import shutil

# 定义文件夹路径
folder1 = './val'  # 输出的标签路径
folder2 = './Annotations'  # 被复制的验证集路径
folder3 = './3xml_val'  # 图片的保存路径

# 获取文件夹1中所有文件的文件名（不包括后缀）
filenames1 = [os.path.splitext(f)[0] for f in os.listdir(folder1)]

# 遍历文件夹2中的文件
for file in os.listdir(folder2):
    # 获取文件名（不包括后缀）
    file_name_without_extension = os.path.splitext(file)[0]

    # 如果文件名在文件夹1的文件名列表中
    if file_name_without_extension in filenames1:
        # 构造完整的源文件路径和目标文件路径
        src = os.path.join(folder2, file)
        new_filename = f"{file_name_without_extension}{os.path.splitext(file)[1]}"
        dst = os.path.join(folder3, new_filename)

        # 将文件复制到文件夹3中
        shutil.copy(src, dst)

print("文件复制完成。")