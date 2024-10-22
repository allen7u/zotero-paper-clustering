import pyperclip
import os
import sys
import datetime
import time
from copy2clip import clip_files

def create_txt_from_clipboard(custom_filename=None, save_dir=None):
    try:
        # 从剪贴板获取文本
        clip_text = pyperclip.paste()
        
        def safe_file_name(file_name):
            # 首先将所有的冒号替换为连字符
            file_name = file_name.replace(':', ' -')
            # 移除非法字符，只保留字母、数字、中文字符、空格、下划线和连字符
            file_name = ''.join(c for c in file_name if c.isalnum() or '\u4e00' <= c <= '\u9fff' or c in (' ', '_', '-','.'))
            # 去除首尾空白字符
            file_name = file_name.strip()
            # 如果文件名为空，使用默认名称
            return file_name if file_name else '未命名'

        if custom_filename:
            # Remove "(123...)" from the beginning of the file name
            if custom_filename.startswith("("):
                custom_filename = custom_filename[1:]
                # Find the first closing parenthesis and remove everything before it
                end_index = custom_filename.find(")")
                if end_index != -1:
                    custom_filename = custom_filename[end_index + 1:]
            # 使用自定义文件名，但确保其有效
            file_name = safe_file_name(custom_filename)
        else:
            # 使用剪贴板内容的前10个字符作为文件名
            file_name = safe_file_name(clip_text[:25])
        
        # 如果文件名仍然为空，使用默认名称
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = file_name or f'未命名_{timestamp}'
        
        # 确保文件名以.txt结尾
        if not file_name.endswith('.txt'):
            file_name += '.txt'
        
        # 如果提供了保存目录，使用它；否则使用当前目录
        if save_dir:
            file_path = os.path.join(save_dir, file_name)
        else:
            file_path = file_name
        
        # 创建文件并写入内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(clip_text)
        
        # 获取文件的绝对路径
        file_path = os.path.abspath(file_path)
        
        # 将文件路径复制到剪贴板
        clip_files([file_path])
        
        print(f"文件已创建：{file_path}")
        print("文件路径已复制到剪贴板")
    except Exception as e:
        print(f"发生错误: {e}")
        # 倒计时100秒
        for i in range(100, 0, -1):
            print(f"\r窗口关闭倒计时: {i} 秒", end="", flush=True)
            time.sleep(1)


if __name__ == "__main__":
    # 检查是否有命令行参数
    print("开始执行")
    print(sys.argv[1])
    print(sys.argv[2])
    time.sleep(1)
    
    custom_filename = sys.argv[1] if len(sys.argv) > 1 else None
    save_dir = sys.argv[2] if len(sys.argv) > 2 else None
    create_txt_from_clipboard(custom_filename, save_dir)