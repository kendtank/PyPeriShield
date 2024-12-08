import os
import shutil

def convert_files_to_md(directory):
    # 定义需要转换的文件扩展名
    extensions = ('.py', '.pt', '.engine')
    print("Converting files...")
    # 递归遍历目录中的所有文件
    for root, dirs, files in os.walk(directory):
        print(f"Processing directory: {root}")
        for file in files:
            if file.endswith(extensions):
                # 构建旧文件路径
                old_file_path = os.path.join(root, file)
                
                # 构建新文件路径（替换扩展名为.md）
                new_file_name = os.path.splitext(file)[0] + '.md'
                new_file_path = os.path.join(root, new_file_name)
                
                print(f"Converting {old_file_path} to {new_file_path}")
                
                try:
                    # 复制文件内容到新的.md文件
                    with open(old_file_path, 'r', encoding='utf-8') as src, \
                         open(new_file_path, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
                        
                    print(f"Successfully converted {old_file_path} to {new_file_path}")
                except Exception as e:
                    print(f"Failed to convert {old_file_path}: {e}")



if __name__ == "__main__":
    # 指定要转换的文件夹路径
    target_directory = r"D:\kend\ours\PyPeriShield"
    
    if not os.path.isdir(target_directory):
        print("提供的路径不是一个有效的文件夹")
    else:
        convert_files_to_md(target_directory)