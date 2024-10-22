# 导入所需模块
import ctypes
from ctypes import wintypes
import pythoncom
import win32clipboard
import os

# 定义 DROPFILES 结构体
class DROPFILES(ctypes.Structure):
    _fields_ = (('pFiles', wintypes.DWORD),
                ('pt', wintypes.POINT),
                ('fNC', wintypes.BOOL),
                ('fWide', wintypes.BOOL))

# 定义 clip_files 函数，用于将文件列表复制到剪贴板
def clip_files(file_list):
    # 计算 DROPFILES 结构体的大小
    offset = ctypes.sizeof(DROPFILES)
    # 计算所有文件路径的总长度
    length = sum(len(p) + 1 for p in file_list) + 1
    # 计算总缓冲区大小
    size = offset + length * ctypes.sizeof(ctypes.c_wchar)
    # 创建缓冲区
    buf = (ctypes.c_char * size)()
    # 从缓冲区创建 DROPFILES 结构体
    df = DROPFILES.from_buffer(buf)
    # 设置 DROPFILES 结构体的字段
    df.pFiles, df.fWide = offset, True
    # 遍历文件列表，将每个文件路径添加到缓冲区
    for path in file_list:
        print("正在复制到剪贴板，文件名 = " + path)
        array_t = ctypes.c_wchar * (len(path) + 1)
        path_buf = array_t.from_buffer(buf, offset)
        path_buf.value = path
        offset += ctypes.sizeof(path_buf)
    # 创建 STGMEDIUM 结构体
    stg = pythoncom.STGMEDIUM()
    # 设置 STGMEDIUM 结构体的数据
    stg.set(pythoncom.TYMED_HGLOBAL, buf)
    # 打开剪贴板
    win32clipboard.OpenClipboard()
    # 清空剪贴板
    win32clipboard.EmptyClipboard()
    try:
        # 输出 STGMEDIUM 结构体和数据
        print(stg)
        print(stg.data)
        # 将数据设置到剪贴板
        win32clipboard.SetClipboardData(win32clipboard.CF_HDROP, stg.data)
        print("clip_files() 成功")
        
        # 对于 TXT 文件，额外复制文本内容
        for path in file_list:
            if path.lower().endswith('.txt'):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print('path:' + path)
                        print('content:' + content[0:10])
                    win32clipboard.SetClipboardText(content, win32clipboard.CF_UNICODETEXT)
                    print(f"已复制 {path} 的内容到剪贴板")
                except UnicodeEncodeError:
                    print(f"无法编码 {path} 的内容，跳过此文件")
    finally:
        # 关闭剪贴板
        win32clipboard.CloseClipboard()
