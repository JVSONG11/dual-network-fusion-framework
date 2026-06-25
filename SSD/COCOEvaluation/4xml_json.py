# import os
# import xml.etree.ElementTree as ET
# import json
#
#
# def voc_to_coco(voc_dir, output_json_file):
#     coco_format = {
#         "images": [],
#         "annotations": [],
#         "categories": []
#     }
#
#     # 定义类别信息，确保类别与XML文件中的名称一致
#     categories = [
#         {"id": 0, "name": "0"},
#
#         # 添加更多类别
#     ]
#     coco_format["categories"] = categories
#
#     annotation_id = 1
#
#     for xml_file in os.listdir(voc_dir):
#         if xml_file.endswith('.xml'):
#             # 去掉文件后缀并将文件名转换为整数形式
#             image_id = int(os.path.splitext(xml_file)[0])
#
#             tree = ET.parse(os.path.join(voc_dir, xml_file))
#             root = tree.getroot()
#
#             # 图像文件信息
#             file_name = root.find('filename').text
#             image_info = {
#                 "id": image_id,
#                 "file_name": file_name,
#                 "width": int(root.find('size/width').text),
#                 "height": int(root.find('size/height').text)
#             }
#             coco_format["images"].append(image_info)
#
#             # 每个物体的标注信息
#             for obj in root.findall('object'):
#                 category_name = obj.find('name').text
#
#                 # 检查类别是否存在于categories中，如果不存在，给出提示
#                 category_id = next((c["id"] for c in categories if c["name"] == category_name), None)
#                 if category_id is None:
#                     print(f"类别 '{category_name}' 不存在于定义的类别列表中。请检查并添加此类别。")
#                     continue  # 跳过该类别
#
#                 # VOC的bbox是xmin, ymin, xmax, ymax
#                 bndbox = obj.find('bndbox')
#                 xmin = int(bndbox.find('xmin').text)
#                 ymin = int(bndbox.find('ymin').text)
#                 xmax = int(bndbox.find('xmax').text)
#                 ymax = int(bndbox.find('ymax').text)
#
#                 width = xmax - xmin
#                 height = ymax - ymin
#
#                 annotation_info = {
#                     "id": annotation_id,
#                     "image_id": image_id,
#                     "category_id": category_id,
#                     "bbox": [xmin, ymin, width, height],
#                     "area": width * height,
#                     "iscrowd": 0
#                 }
#                 coco_format["annotations"].append(annotation_info)
#                 annotation_id += 1
#
#     # 将结果保存为JSON文件
#     with open(output_json_file, 'w') as f:
#         json.dump(coco_format, f, indent=4)
#
#
# # 例子：将VOC格式的标注文件转换为COCO JSON格式
# voc_to_coco("./3xml_val", "dataset11.json")

import os
import xml.etree.ElementTree as ET
import json


def voc_to_coco(voc_dir, output_json_file):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 定义类别信息，确保类别与XML文件中的名称一致
    categories = [
        {"id": 1, "name": "1"}  # 将类别id改为1

        # 添加更多类别，如果需要
    ]
    coco_format["categories"] = categories

    annotation_id = 1

    for xml_file in os.listdir(voc_dir):
        if xml_file.endswith('.xml'):
            # 去掉文件后缀并将文件名转换为整数形式
            image_id = int(os.path.splitext(xml_file)[0])

            tree = ET.parse(os.path.join(voc_dir, xml_file))
            root = tree.getroot()

            # 图像文件信息
            file_name = root.find('filename').text
            image_info = {
                "id": image_id,
                "file_name": file_name,
                "width": int(root.find('size/width').text),
                "height": int(root.find('size/height').text)
            }
            coco_format["images"].append(image_info)

            # 每个物体的标注信息
            for obj in root.findall('object'):
                category_name = obj.find('name').text

                # 无论类别名称是什么，都将category_id设置为1
                category_id = 1

                # VOC的bbox是xmin, ymin, xmax, ymax
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                width = xmax - xmin
                height = ymax - ymin

                annotation_info = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,  # 固定为1
                    "bbox": [xmin, ymin, width, height],
                    "area": width * height,
                    "iscrowd": 0
                }
                coco_format["annotations"].append(annotation_info)
                annotation_id += 1

    # 将结果保存为JSON文件
    with open(output_json_file, 'w') as f:
        json.dump(coco_format, f, indent=4)


# 例子：将VOC格式的标注文件转换为COCO JSON格式
voc_to_coco("./3xml_val", "dataset11.json")

