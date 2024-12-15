# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/6 下午8:37
@Author  : Kend
@FileName:
@Software: PyCharm
@modifier:
"""

import subprocess
import re

# 读取 requirements.txt 文件
with open('requirements.txt', 'r') as f:
    lines = f.readlines()

# 定义一个函数来获取包的版本号
def get_package_version(package_name):
    try:
        result = subprocess.run(['pip', 'show', package_name], stdout=subprocess.PIPE, text=True)
        for line in result.stdout.splitlines():
            if line.startswith('Version:'):
                return line.split(': ')[1]
        return None
    except Exception as e:
        print(f"Error getting version for {package_name}: {e}")
        return None

# 遍历每一行，检查是否是本地包，并添加版本号
updated_lines = []
for line in lines:
    match = re.match(r'^(.+?)\s*@', line)
    if match:
        package_name = match.group(1)
        version = get_package_version(package_name)
        if version:
            updated_line = f"{line.strip()} ; version==\"{version}\"\n"
            updated_lines.append(updated_line)
        else:
            updated_lines.append(line)
    else:
        updated_lines.append(line)

# 将更新后的内容写回 requirements.txt 文件
with open('requirements.txt', 'w') as f:
    f.writelines(updated_lines)

print("Updated requirements.txt with local package versions.")