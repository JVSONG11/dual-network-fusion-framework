import json
from json2html import *

# 读取JSON文件
with open('./mini.json', 'r') as f:
    data = json.load(f)

# 将JSON转换为HTML
html_content = json2html.convert(json=data)

# 将HTML保存到文件
with open('output.html', 'w') as html_file:
    html_file.write(html_content)

print("JSON结构已转换为HTML，请查看output.html")
