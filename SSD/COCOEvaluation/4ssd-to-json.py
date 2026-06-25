# import xml.etree.ElementTree as ET
# import os
# import json
#
# json_output_path = 'ssd_1.json'
# xml_folder = './3ssd_xml/XML'
#
# annotations = []
#
# for xml_file in os.listdir(xml_folder):
#     if xml_file.endswith(".xml"):
#         tree = ET.parse(os.path.join(xml_folder, xml_file))
#         root = tree.getroot()
#
#         # Generate image_id from the file name (assuming it's numerical and related to image_id)
#         image_id = int(os.path.splitext(xml_file)[0])
#
#         for obj in root.findall("object"):
#             category_id = int(obj.find("name").text)  # Assuming name is a number
#             bndbox = obj.find("bndbox")
#
#             xmin = int(float(bndbox.find("xmin").text))
#             ymin = int(float(bndbox.find("ymin").text))
#             xmax = int(float(bndbox.find("xmax").text))
#             ymax = int(float(bndbox.find("ymax").text))
#
#             score = float(obj.find("score").text) if obj.find("score") is not None else None
#
#             annotation = {
#                 "image_id": image_id,
#                 "category_id": category_id,
#                 "bbox": [xmin, ymin, xmax, ymax],
#                 "score": score
#             }
#             annotations.append(annotation)
#
# # Write annotations to a JSON file
# with open(json_output_path, 'w') as json_file:
#     json.dump(annotations, json_file, indent=4)
#
# print(f"JSON file saved as {json_output_path}")


import xml.etree.ElementTree as ET
import os
import json

json_output_path = 'ssd11.json'
xml_folder = './3ssd_xml/XML'

annotations = []

for xml_file in os.listdir(xml_folder):
    if xml_file.endswith(".xml"):
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()

        # Generate image_id from the file name (assuming it's numerical and related to image_id)
        image_id = int(os.path.splitext(xml_file)[0])

        for obj in root.findall("object"):
            category_id = 1  # 将category_id固定为1
            bndbox = obj.find("bndbox")

            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            width = xmax - xmin
            height = ymax - ymin

            score = float(obj.find("score").text) if obj.find("score") is not None else None

            annotation = {
                "image_id": image_id,
                "category_id": category_id,  # 固定为1
                "bbox": [xmin, ymin, width, height],
                "score": score
            }
            annotations.append(annotation)

# Write annotations to a JSON file
with open(json_output_path, 'w') as json_file:
    json.dump(annotations, json_file, indent=4)

print(f"JSON file saved as {json_output_path}")
