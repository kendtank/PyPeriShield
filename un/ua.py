import os
import shutil


def convert_md_files_back(directory, target_extension='.py'):
    # 递归遍历目录中的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                # 构建旧文件路径
                old_file_path = os.path.join(root, file)

                # 获取不带扩展名的文件名
                base_name = os.path.splitext(file)[0]

                # 构建新文件路径（默认转换为.py）
                new_file_path = os.path.join(root, base_name + target_extension)

                # 检查是否已经存在同名的.py文件
                if os.path.exists(new_file_path):
                    print(f"Found existing {new_file_path}. Deleting it.")
                    try:
                        os.remove(new_file_path)
                        print(f"Deleted {new_file_path}")
                    except Exception as e:
                        print(f"Failed to delete {new_file_path}: {e}")
                        continue

                print(f"Converting {old_file_path} to {new_file_path}")

                try:
                    # 复制文件内容到新的目标文件
                    with open(old_file_path, 'r', encoding='utf-8') as src, \
                            open(new_file_path, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())

                    print(f"Successfully converted {old_file_path} to {new_file_path}")

                    # 删除转换后的.md文件
                    # os.remove(old_file_path)

                    print(f"Deleted {old_file_path}")
                except Exception as e:
                    print(f"Failed to convert {old_file_path}: {e}")

if __name__ == "__main__":
    # 指定要转换的文件夹路径
    target_directory = r"D:\kend\ours\PyPeriShield"
    
    if not os.path.isdir(target_directory):
        print("提供的路径不是一个有效的文件夹")
    else:
        convert_md_files_back(target_directory)