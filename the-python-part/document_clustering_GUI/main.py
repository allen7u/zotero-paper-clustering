import random
import time
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.widgets import RectangleSelector
import numpy as np
import pandas as pd
import os
import pyperclip
import ctypes
import pyperclip
import os
import hashlib
import json
from openai import AsyncOpenAI, OpenAI
import tiktoken
from copy2clip import clip_files

# Print all environment variables
for key in os.environ:
    print(f"{key}: {os.environ[key]}")

# 查询地址：https://query.onechats.top
# 次数：https://api.onechats.top
# 额度全模型：https://chatapi.onechats.top
# 3.5：https://sapi.onechats.top

use_openai_embedding = False  # 设置是否使用OpenAI的嵌入API
# use_openai_embedding = True  # 设置是否使用OpenAI的嵌入API

if use_openai_embedding:
    # 使用OpenAI的嵌入API
    print("使用OpenAI的嵌入API")
else:
    # 使用本地模型
    print("使用本地模型")
    start_time = time.time()
    from transformers import AutoTokenizer, AutoModel
    import torch
    print(f"transformers and torch import 过程耗时: {time.time() - start_time:.4f}秒")
    
    initiate_start_time = time.time()
    model_name = "maidalun1020/bce-embedding-base_v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    initiate_time = time.time() - initiate_start_time
    print(f"模型初始化耗时: {initiate_time:.4f}秒")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    to_device_start_time = time.time()
    model.to(device)
    to_device_time = time.time() - to_device_start_time
    print(f"模型移至设备耗时: {to_device_time:.4f}秒")



# 保存选中的标签
selected_labels = []

print(os.getcwd())
# time.sleep(10)

# 从剪贴板读取JSON字符串
clipboard_content = pyperclip.paste()
try:
    input_data = json.loads(clipboard_content)
    print("成功从剪贴板读取JSON数据")
except json.JSONDecodeError:
    print("剪贴板内容不是有效的JSON格式")
    exit()

# 从JSON数据中提取路径
input_paths = [item.get('exportedPath') for item in input_data.get('data', []) if item.get('exportedPath')]
print(f"从JSON中提取的输入路径: {input_paths}")

# 验证路径
valid_paths = [path for path in input_paths if os.path.isfile(path) and path.lower().endswith('.txt')]
if not valid_paths:
    print("没有有效的TXT文件路径")
    exit()

print(f"将使用以下TXT文件路径：{valid_paths}")

# 从JSON数据中提取摘要
input_abstracts = [item.get('abstract', '') for item in input_data.get('data', [])]
print(f"从JSON中提取的摘要数量: {len(input_abstracts)}")

def get_dpi_scale():
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetDpiForSystem() / 96.0
    except:
        return 1.0


def get_all_direct_child_text_paths_include_and_above_current_dir():
    all_direct_child_text_paths_include_and_above_current_dir = []
    
    # 首先检查当前目录
    current_dir = os.path.abspath(".")
    print('current_dir:', current_dir)
    
    while True:
        # print('Check:', current_dir)
        for file_name in os.listdir(current_dir):
            file_path = os.path.join(current_dir, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.txt', '.html')):
                # print('-Check:', file_path)
                all_direct_child_text_paths_include_and_above_current_dir.append(file_path)
        
        # 如果已经到达根目录，退出循环
        if current_dir == os.path.dirname(current_dir):
            break
        
        # 移动到父目录
        current_dir = os.path.dirname(current_dir)

    return all_direct_child_text_paths_include_and_above_current_dir

all_direct_child_text_paths_include_and_above_current_dir = get_all_direct_child_text_paths_include_and_above_current_dir()
all_direct_child_text_paths_include_and_above_current_dir.extend(valid_paths)
print('all_direct_child_text_paths_include_and_above_current_dir[plus the valid_paths]:', all_direct_child_text_paths_include_and_above_current_dir)

  
def count_chars(text):
    chinese_count = 0
    english_count = 0
    other_count = 0
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            chinese_count += 1
        elif char.isascii() and char.isalpha():
            english_count += 1
        else:
            other_count += 1
    return chinese_count, english_count, other_count

def truncate_text(text, max_length):
    current_length = 0
    truncated_text = ""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            char_length = 2  # 假设一个中文字符的宽度等于两个英文字符
        else:
            char_length = 1
        if current_length + char_length > max_length:
            break
        truncated_text += char
        current_length += char_length
    return truncated_text + "..." if len(text) > len(truncated_text) else truncated_text

def format_doc_names(file_name, document='', max_file_name_length=1250, max_document_length=0):
    fn_chinese, fn_english, fn_other = count_chars(file_name)
    if fn_chinese > fn_english:
        fn_max = max_file_name_length // 2  # 如果主要是中文，长度减半
    else:
        fn_max = max_file_name_length
    truncated_file_name = truncate_text(file_name, fn_max)
    doc_chinese, doc_english, doc_other = count_chars(document)
    if doc_chinese > doc_english:
        doc_max = max_document_length // 2  # 如果主要是中文，长度减半
    else:
        doc_max = max_document_length
    truncated_document = truncate_text(document, doc_max)
    if max_document_length == 0:
        return f"{truncated_file_name}"
    else:
        return f"{truncated_file_name} - {truncated_document}"



# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class EmbeddingCache:
    def __init__(self, cache_file='D:\\My_Codes\\document_clustering_GUI\\embedding_cache.json'):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def get_embedding(self, text, model_name):
        text_hash = hashlib.md5((text + model_name).encode()).hexdigest()
        return self.cache.get(text_hash)

    def set_embedding(self, text, embedding, model_name):
        text_hash = hashlib.md5((text + model_name).encode()).hexdigest()
        self.cache[text_hash] = embedding

# 初始化OpenAI客户端和缓存
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url='https://chatapi.onechats.top/v1/'
)
cache = EmbeddingCache()

# 生成嵌入向量
def generate_embeddings(texts, use_openai=False, batch_size=32):
    model_name = "text-embedding-3-small" if use_openai else "maidalun1020/bce-embedding-base_v1"
    embeddings = []
    total_embedding_time = 0
    total_uncached_docs = 0
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # cached_embeddings = [cache.get_embedding(text, model_name) for text in batch]
        cached_embeddings = [None for text in batch]
        
        uncached_indices = [j for j, emb in enumerate(cached_embeddings) if emb is None]
        uncached_texts = [batch[j] for j in uncached_indices]
        
        if uncached_texts:
            batch_start_time = time.time()
            if use_openai:
                enc = tiktoken.get_encoding("cl100k_base")
                truncated_texts = [enc.decode(enc.encode(text)[:8000]) for text in uncached_texts]
                
                response = client.embeddings.create(input=truncated_texts, model=model_name)
                new_embeddings = [data.embedding for data in response.data]
            else:                
                embedding_start_time = time.time()
                inputs = tokenizer(uncached_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs_on_device, return_dict=True)
                
                new_embeddings = outputs.last_hidden_state[:, 0]  # cls pooler
                new_embeddings = new_embeddings / new_embeddings.norm(dim=1, keepdim=True)  # normalize
                new_embeddings = new_embeddings.cpu().tolist()
                
                embedding_time = time.time() - embedding_start_time
                print(f"嵌入向量生成过程耗时: {embedding_time:.4f}秒")
            
            batch_end_time = time.time()
            batch_embedding_time = batch_end_time - batch_start_time
            total_embedding_time += batch_embedding_time
            total_uncached_docs += len(uncached_texts)
            
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache.set_embedding(text, embedding, model_name)
                print("生成新的embedding并缓存")
            
            for j, new_emb in zip(uncached_indices, new_embeddings):
                cached_embeddings[j] = new_emb
        
        embeddings.extend(cached_embeddings)
    
    avg_uncached_embedding_time = total_embedding_time / total_uncached_docs if total_uncached_docs > 0 else 0
    print(f"每个未缓存文档的平均处理时间: {avg_uncached_embedding_time:.4f}秒")
    
    # 保存缓存到文件
    cache.save_cache()
    
    return embeddings

# 读取文件并生成嵌入向量
embeddings = []
file_names = []

total_time = 0
processed_files = 0
failed_files = []
files_over_threshold_processing_time = 0

# 读取所有文件内容
all_contents = []
for path in valid_paths:
    try:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        all_contents.append(content)
        file_names.append(os.path.basename(path)[:150])
    except Exception as e:
        print(f"无法读取文件 {path}: {str(e)}")
        failed_files.append(path)

# 批量生成嵌入向量
start_time = time.time()
embeddings = generate_embeddings(all_contents, use_openai=use_openai_embedding)
end_time = time.time()

total_time = end_time - start_time
processed_files = len(all_contents)

print(f"嵌入向量数量: {len(embeddings)}")
print(f"文件名数量: {len(file_names)}")
print(f"处理失败的文件数量: {len(failed_files)}")
print(f"总处理时间: {total_time:.2f} 秒")
print(f"平均每个文件处理时间: {total_time / processed_files:.2f} 秒")

if failed_files:
    log_file_path = f'D:\\My_Codes\\document_clustering_GUI\\failed_embedding_files_{time.strftime("%Y%m%d%H%M%S")}.log'
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        for failed_path in failed_files:
            log_file.write(f"{failed_path}\n")
    print(f"处理失败的文件名已记录在 '{log_file_path}' 中")


# 创建DataFrame
df = pd.DataFrame({
    'file_name': file_names,
    'embedding': embeddings
})

print("文件名列表:")
for filename in file_names:
    print(filename)
    
# 处理嵌入向量
use_mock_data = 0  # 设置是否使用模拟数据
if use_mock_data:
    # 模拟数据代码保持不变
    pass
else:
    embedding_dim = len(df['embedding'][0])
    for i in range(len(df['embedding'])):
        if df['embedding'][i] is None:
            df.at[i, 'embedding'] = [0.0] * embedding_dim

    embeddings = df['embedding'].tolist()
    file_names = df['file_name'].tolist()

# 创建主窗口
root = tk.Tk()
# root.title("聚类结果演示")
root.title("")

def search_labels():
    search_term = search_entry.get().lower()
    for text in label_texts:
        if search_term in text.get_text().lower() and text.get_text() not in selected_labels:
            # text.set_bbox(dict(facecolor='lightblue', edgecolor='blue', alpha=0.5))
            text.set_bbox(dict(facecolor='lightblue', alpha=0.5))
    fig.canvas.draw_idle()

def clear_highlights():
    for text in label_texts:
        if text.get_text() in selected_labels:
            text.set_bbox(dict(facecolor='yellow', edgecolor='white', alpha=1))
        else:
            text.set_bbox(dict(facecolor='white', edgecolor='white', alpha=1))
    fig.canvas.draw_idle()

selected_paths = []
lasted_selected_paths = []

def doc_names_to_abs_paths(doc_names, abs_paths, all_direct_child_text_paths_include_and_above_current_dir):   
    # 尝试匹配doc_names中的文件名，优先选择最直接的父文件夹
    print('abs_paths from doc_names_to_abs_paths:', abs_paths)
    matched_paths = []
    unmatched_names = []
    for doc_name in doc_names:
        matched = False
        for path in all_direct_child_text_paths_include_and_above_current_dir:
            if os.path.basename(path) == doc_name:
                matched_paths.append(path)
                matched = True
                # print(f"匹配成功: {doc_name} -> {path}")
                break
        if not matched:
            unmatched_names.append(doc_name)
            # print(f"未匹配: {doc_name}")
    
    # 将匹配的 doc_names 复制到剪贴板
    pyperclip.copy('\n'.join(doc_names))
    # print clip
    print(f"已复制 doc_names 到剪贴板: {doc_names}")
    
    # print(f"未匹配的文件名: {unmatched_names}")
    abs_paths.clear()
    abs_paths.extend(matched_paths)
    print(f"set selected_paths: {abs_paths}")
    return abs_paths

def set_clip_data_from_selected_paths(selected_paths = selected_paths):
    print(f"get selected_paths: {selected_paths}")
    clip_files(selected_paths)

def set_clip_data_from_latest_selected_paths(selected_paths = lasted_selected_paths):
    print(f"get selected_paths: {lasted_selected_paths}")
    clip_files(lasted_selected_paths)




def show_summary(summary):
    summary_window = tk.Toplevel(root)
    summary_window.title("总结结果")
    
    # 考虑 DPI 缩放调整窗口大小
    width = int(800 * dpi_scale)
    height = int(800 * dpi_scale)
    
    # 获取主窗口的位置
    root_x = root.winfo_x()
    root_y = root.winfo_y()
    root_width = root.winfo_width()
    root_height = root.winfo_height()
    
    # 计算居中位置
    x = root_x + (root_width - width) // 2
    y = root_y + (root_height - height) // 2
    
    # 设置窗口大小和位置
    summary_window.geometry(f"{width}x{height}+{x}+{y}")

    # 创建一个主框架来容纳文本小部件和按钮
    main_frame = tk.Frame(summary_window)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 创建一个框架来容纳按钮
    button_frame = tk.Frame(main_frame)
    button_frame.pack(side=tk.BOTTOM, pady=int(10*dpi_scale))
    
    # 调整字体大小
    font_size = int(14 * dpi_scale)
    text_widget = tk.Text(main_frame, wrap=tk.WORD, font=("SimHei", font_size))
    text_widget.pack(expand=True, fill=tk.BOTH, padx=int(10*dpi_scale), pady=int(10*dpi_scale))

    text_widget.insert(tk.END, summary)
    text_widget.config(state=tk.DISABLED)

    # 调整按钮大小和字体
    button_font_size = int(14 * dpi_scale)
    copy_button = tk.Button(button_frame, text="复制结果", 
                            command=lambda: pyperclip.copy(text_widget.get("1.0", tk.END)),
                            font=("SimHei", button_font_size))
    copy_button.pack(side=tk.LEFT, padx=(0, 15))

    # 添加一个选项让用户控制是否保持在最顶层
    def toggle_topmost():
        is_topmost = summary_window.attributes('-topmost')
        summary_window.attributes('-topmost', not is_topmost)
        topmost_button.config(text="取消置顶" if not is_topmost else "窗口置顶")

    topmost_button = tk.Button(button_frame, text="取消置顶", 
                               command=toggle_topmost,
                               font=("SimHei", button_font_size))
    topmost_button.pack(side=tk.LEFT, padx=(0, 15))
    
    # 使窗口大小可调整
    summary_window.resizable(True, True)

    # 确保窗口在最上层
    # summary_window.lift()
    # summary_window.attributes('-topmost', True)

    return summary_window, text_widget

from enum import Enum

class SummaryMode(Enum):
    TITLE = "title"
    ABSTRACT = "abstract"
    CONTENT = "content"
    TITLE_FIELD = "title_field"
    ABSTRACT_FIELD = "abstract_field"
    CONTENT_FIELD = "content_field"

def get_filtered_content(mode):
    global selected_indices, input_data
    
    filtered_data = [input_data['data'][idx] for idx in reversed(selected_indices)]
    
    if mode == SummaryMode.ABSTRACT or mode == SummaryMode.ABSTRACT_FIELD:
        return [f"标题：{item['title']}\n\n摘要：{item['abstract']}" for item in filtered_data if item.get('title') and item.get('abstract')]
    elif mode == SummaryMode.CONTENT or mode == SummaryMode.CONTENT_FIELD:
        return [f"全文：{' '.join(item['fulltext'])}" for item in filtered_data if item.get('fulltext')]
    elif mode == SummaryMode.TITLE_FIELD or mode == SummaryMode.TITLE:
        return [f"标题：{item['title']}" for item in filtered_data if item.get('title')]

def summarize_with_GPT(mode: SummaryMode):
    filtered_content = get_filtered_content(mode)
    
    if not filtered_content:
        messagebox.showinfo("提示", f"没有选中的{mode.value}可用于总结")
        return
    
    content_to_summarize = "\n\n---\n\n".join(filtered_content)
    
    system_message = "你是一个专业的文本分析助手。"
    if mode in [SummaryMode.CONTENT, SummaryMode.ABSTRACT, SummaryMode.TITLE]:
        user_message = f"请归纳下列文本所讨论的共同主题：\n\n{content_to_summarize}"
    elif mode in [SummaryMode.CONTENT_FIELD, SummaryMode.ABSTRACT_FIELD, SummaryMode.TITLE_FIELD]:
        user_message = f"For each item: Print title //in English//; Assign some most likely domain, field, subfield, \
            topic //in Chinese//;\n\n{content_to_summarize}"
    else:
        messagebox.showerror("错误", "无效的总结模式")
        return
    
    summary_window, text_widget = show_summary("")
    
    client_chat = OpenAI(
        api_key=os.environ["OPENAI_API_KEY_Sonnet_3_5"],
        base_url='https://chatapi.onechats.top/v1/'
    )
    
    print(f"user_message: {user_message}")
    
    try:
        max_tokens = 4000
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        original_token_count = len(encoding.encode(user_message))
        
        # 检查token数量是否超过限制
        if original_token_count > 15000 - max_tokens:
            truncated_message = encoding.decode(encoding.encode(user_message)[:15000-max_tokens])
            warning = f"警告：输入内容过长，已被截断。原始token数：{original_token_count}\n\n"
            text_widget.config(state=tk.NORMAL)
            text_widget.insert(tk.END, warning)
            text_widget.config(state=tk.DISABLED)
            user_message = truncated_message
        
        response = client_chat.chat.completions.create(
            model="claude-3-5-sonnet-20240620",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            stream=True
        )
        
        summary = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                summary += content
                text_widget.config(state=tk.NORMAL)
                text_widget.insert(tk.END, content)
                text_widget.see(tk.END)
                text_widget.config(state=tk.DISABLED)
                text_widget.update()
        
        print("总结结果：")
        print(summary)
        
    except Exception as e:
        error_message = f"总结过程中出现错误：{str(e)}"
        print(error_message)
        messagebox.showerror("错误", error_message)
        
search_frame = tk.Frame(root, padx=15, pady=15)
search_frame.pack(side=tk.TOP, fill=tk.X)

summary_button = tk.Button(search_frame, text="标题主题", command=lambda: summarize_with_GPT(SummaryMode.TITLE), font=("Arial", 40))
summary_button.pack(side=tk.LEFT)

abstract_summary_button = tk.Button(search_frame, text="摘要主题", command=lambda: summarize_with_GPT(SummaryMode.ABSTRACT), font=("Arial", 40))
abstract_summary_button.pack(side=tk.LEFT)

full_text_summary_button = tk.Button(search_frame, text="全文主题", command=lambda: summarize_with_GPT(SummaryMode.CONTENT), font=("Arial", 40))
full_text_summary_button.pack(side=tk.LEFT)

title_field_summary_button = tk.Button(search_frame, text="标题领域", command=lambda: summarize_with_GPT(SummaryMode.TITLE_FIELD), font=("Arial", 40))
title_field_summary_button.pack(side=tk.LEFT)

abstract_field_summary_button = tk.Button(search_frame, text="摘要领域", command=lambda: summarize_with_GPT(SummaryMode.ABSTRACT_FIELD), font=("Arial", 40))
abstract_field_summary_button.pack(side=tk.LEFT)

content_field_summary_button = tk.Button(search_frame, text="全文领域", command=lambda: summarize_with_GPT(SummaryMode.CONTENT_FIELD), font=("Arial", 40))
content_field_summary_button.pack(side=tk.LEFT)



# ... 现有的导入语句 ...
import asyncio
import threading

# ... 其他现有代码 ...

async def summarize_single_document(client, content, text_widget):
    try:
        max_tokens = 4000
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        original_token_count = len(encoding.encode(content))
        
        # 检查token数量是否超过限制
        if original_token_count > 15000 - max_tokens:
            truncated_content = encoding.decode(encoding.encode(content)[:15000-max_tokens])
            warning = f"警告：输入内容过长，已被截断。原始token数：{original_token_count}\n\n"
            text_widget.config(state=tk.NORMAL)
            text_widget.insert(tk.END, warning)
            text_widget.config(state=tk.DISABLED)
            content = truncated_content

        stream = await client.chat.completions.create(
            model="claude-3-5-sonnet-20240620",
            messages=[
                {"role": "system", "content": "你是一个专业的文本分析助手。"},
                {"role": "user", "content": f"请在反复阅读后总结要点, please response in Chinese：\n\n{content}"}
            ],
            max_tokens=max_tokens,
            stream=True
        )
        
        summary = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                summary += content
                text_widget.config(state=tk.NORMAL)
                text_widget.insert(tk.END, content)
                text_widget.see(tk.END)
                text_widget.config(state=tk.DISABLED)
                text_widget.update()
        
        print(f"摘要生成完成：{summary}")
        
    except Exception as e:
        error_message = f"摘要生成过程中出现错误：{str(e)}"
        print(error_message)
        text_widget.config(state=tk.NORMAL)
        text_widget.insert(tk.END, f"\n\n{error_message}")
        text_widget.config(state=tk.DISABLED)
def create_multi_summary_window(use_full_text=False):
    if not selected_indices:
        messagebox.showinfo("提示", "请先选择要摘要的文档")
        return
    
    summary_window = tk.Toplevel(root)
    summary_window.title("多文档摘要")
    summary_window.configure()  
    
    width = int(800 * dpi_scale)
    height = int(800 * dpi_scale)
    
    summary_window.update_idletasks()
    screen_width = int(summary_window.winfo_screenwidth() * dpi_scale)
    screen_height = int(summary_window.winfo_screenheight() * dpi_scale)
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    print(f"调试信息：屏幕宽度 = {screen_width}, 屏幕高度 = {screen_height}")
    print(f"调试信息：窗口位置 x = {x}, y = {y}")
    summary_window.geometry(f"{width}x{height}+{x}+{y}")

    main_frame = ttk.Frame(summary_window)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    close_button = tk.Button(main_frame, text="关闭", font=("Arial", 30), command=summary_window.destroy)
    close_button.pack(side=tk.BOTTOM, pady=10)
    
    canvas = tk.Canvas(main_frame) 
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    
    scrollable_frame = ttk.Frame(canvas)    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=canvas.winfo_width())
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    canvas.bind('<Configure>', lambda e: canvas.itemconfig(canvas.find_withtag('all')[0], width=e.width))
    
    text_widgets = []
    for idx in reversed(selected_indices):
        frame = ttk.Frame(scrollable_frame)
        frame.pack(fill=tk.X, padx=5, pady=10)
        
        title = input_data['data'][idx].get('title', f"文档 {idx + 1}")
        label = ttk.Label(frame, text=title, font=("SimHei", int(14 * dpi_scale)))  
        label.pack(anchor="w")
        
        text_widget = tk.Text(frame, wrap=tk.WORD, height=25, font=("SimHei", int(14 * dpi_scale)))  
        text_widget.pack(fill='both', expand=True, pady=(5, 0))
        text_widget.insert(tk.END, "")
        text_widget.config(state=tk.DISABLED)
        
        text_widgets.append(text_widget)
    
    client_chat = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY_Sonnet_3_5"],
        base_url='https://chatapi.onechats.top/v1/'
    )
    
    async def run_summaries():
        tasks = []
        for idx, text_widget in zip(reversed(selected_indices), text_widgets):
            if use_full_text:
                content = ' '.join(input_data['data'][idx]['fulltext'])
            else:
                title = input_data['data'][idx].get('title', '')
                abstract = input_data['data'][idx].get('abstract', '')
                content = f"标题：{title}\n\n摘要：{abstract}"
            print(content)
            task = asyncio.create_task(summarize_single_document(client_chat, content, text_widget))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    def start_summaries():
        asyncio.run(run_summaries())
    
    thread = threading.Thread(target=start_summaries)
    thread.start()

    summary_window.resizable(True, True)

    summary_window.lift()
    summary_window.attributes('-topmost', True)
    summary_window.after_idle(summary_window.attributes, '-topmost', False)

# 在搜索框下方添加新按钮
multi_summary_button = tk.Button(search_frame, text="各摘要总结", command=lambda: create_multi_summary_window(False), font=("Arial", 40))  
multi_summary_button.pack(side=tk.LEFT)

# 添加全文多文档摘要按钮
multi_summary_full_text_button = tk.Button(search_frame, text="各全文总结", command=lambda: create_multi_summary_window(True), font=("Arial", 40))  
multi_summary_full_text_button.pack(side=tk.LEFT)

# ... 其他现有代码 ...


search_entry = tk.Entry(search_frame, font=("Arial", 40))
search_entry.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
copy_button = tk.Button(search_frame, text="复制", command=set_clip_data_from_selected_paths, font=("Arial", 40))
copy_button.pack(side=tk.RIGHT)
search_button = tk.Button(search_frame, text="搜索", command=search_labels, font=("Arial", 40))
search_button.pack(side=tk.RIGHT)
clear_button = tk.Button(search_frame, text="清除", command=clear_highlights, font=("Arial", 40))
clear_button.pack(side=tk.RIGHT)

# 创建画布和滚动条
canvas_frame = tk.Frame(root, bg='red')
canvas_frame.pack(fill=tk.BOTH, expand=True)

# 配置垂直滚动条
canvas_scroll_y = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, width=50, bg='white')
canvas_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

# 配置水平滚动条
canvas_scroll_x = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, width=50, bg='white')
canvas_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

# 聚类图配置
linkage_matrix = linkage(embeddings, method='ward')
# 计算树状图的深度
def get_dendrogram_depth(linkage_matrix):
    n_samples = linkage_matrix.shape[0] + 1
    depths = np.zeros(2 * n_samples - 1)
    for i, link in enumerate(linkage_matrix):
        left, right = int(link[0]), int(link[1])
        depths[n_samples + i] = max(depths[left], depths[right]) + 1
    return max(depths)

depth = get_dendrogram_depth(linkage_matrix)
print('The depth of the dendrogram is:', depth)

dpi_scale = get_dpi_scale()  # 获取系统 DPI 缩放比例

# 定义期望的像素尺寸
screen_width = 3200
dendrogram_base_width = 580
effective_depth = max(depth - 10, 1)
depth_factor = 1
width_pixels = dendrogram_base_width * effective_depth * depth_factor + screen_width - dendrogram_base_width
# width_pixels = max(3200, width_pixels)
print(f"width_pixels: {width_pixels}")
# width_pixels = 3200
# height_pixels = 2000

# 计算英寸尺寸
width_inches = width_pixels / (100*dpi_scale)

print(width_inches)
# height_inches = height_pixels / (96*dpi_scale)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_inches, len(file_names)*0.3))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_inches, len(file_names)*0.3), dpi=100*dpi_scale)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, len(file_names)*0.4))
# 对于fig（matplotlib的Figure对象）：
# 1. 大小：fig.set_size_inches(width, height)
# 2. DPI：fig.set_dpi(dpi)
# 3. 背景色：fig.patch.set_facecolor('color')
# 4. 标题：fig.suptitle('标题')
# 5. 子图间距：fig.subplots_adjust(left, right, bottom, top, wspace, hspace)
# 6. 紧凑布局：fig.tight_layout()
leaf_info = dendrogram(linkage_matrix, labels=file_names, orientation='left', ax=ax1)
ax1.set_axis_off()

ordered_labels = [file_names[idx] for idx in leaf_info['leaves']]
print(ordered_labels)

label_positions = ax1.get_yticks()
print(label_positions)


truncated_ordered_labels = [format_doc_names(file_name) for file_name in ordered_labels]

ax1.set_yticklabels([])
ax1.set_xticks([])



############## ax2

        
        
# 获取标签和对应的y坐标

# 在第二个子图中绘制标签
ax2.set_ylim(ax1.get_ylim())
# ax2.set_yticks(label_positions)
ax2.set_yticklabels([])
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
# 设置ax2的背景颜色为蓝色
# ax2.set_facecolor('blue')
ax2.patch.set_facecolor('blue')

# 创建一个列表来存储文本对象
label_texts = []

# 在第二个子图中绘制标签框并存储文本对象
for label, y in zip(truncated_ordered_labels, label_positions):
    text = ax2.text(0.01, y, label[0:1000], ha='left', va='center', fontsize=15)
    # text = ax2.text(0.01, y, label[0:1000], ha='left', va='center')
    label_texts.append(text)
    text.set_bbox(dict(facecolor='white', edgecolor='white', alpha=1))


ax2.set_axis_off()

plt.tight_layout()
current_time = time.strftime("%Y%m%d_%H%M%S")
try:
    plt.savefig(f'dendrogram_figure_output\\_{ordered_labels[0]}_etc_{current_time}.png')
except:
    plt.savefig(f'D:\\My_Codes\\document_clustering_GUI\\dendrogram_figure_output\\_{ordered_labels[0]}_etc_{current_time}.png')



# 创建matplotlib画布
canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
canvas.draw()
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
canvas_widget.config(yscrollcommand=canvas_scroll_y.set, xscrollcommand=canvas_scroll_x.set)
canvas_scroll_y.config(command=canvas_widget.yview)
canvas_scroll_x.config(command=canvas_widget.xview)
# 设置画布背景颜色为绿色
canvas_widget.configure()

# 使画布可滚动
canvas_widget.bind("<Configure>", lambda e: canvas_widget.configure(scrollregion=canvas_widget.bbox("all")))


# 对于FigureCanvasTkAgg：
# 1. 主窗口：在创建时指定，如FigureCanvasTkAgg(fig, master=root)
# 2. 绘制：canvas.draw()
# 3. 获取Tkinter小部件：canvas.get_tk_widget()
# 4. 设置大小：canvas.get_tk_widget().config(width=width, height=height)
# 5. 背景色：canvas.get_tk_widget().config(bg='color')
# 6. 绑定事件：canvas.mpl_connect('event_name', handler_function)

# 滚动函数
def scroll(event):
    if event.delta > 0:
        canvas_widget.yview_scroll(-1, "units")
    else:
        canvas_widget.yview_scroll(1, "units")
# 绑定鼠标滚轮事件
canvas_widget.bind("<MouseWheel>", scroll)




# # 保存选中的标签
# selected_labels = []


class ClickManager:
    def __init__(self, tolerance=5):
        self.press_event = None
        self.distance_between_press_and_release = 0

    def on_press(self, event):
        self.press_event = event
        self.distance_between_press_and_release = 0
        if event.button == 1:
            pass

    def on_release(self, event):
        self.release_event = event
        if event.button == 1:
            pass

    def on_move(self, event):
        if event.button == 1:
            self.distance_between_press_and_release = np.sqrt(np.power(event.x - self.press_event.x, 2) + np.power(event.y - self.press_event.y, 2))
        

click_manager = ClickManager()

# 全局变量来存储选中项目的索引
selected_indices = []

def on_release(event):
    global selected_paths, selected_indices
    
    if event.button != 1 or click_manager.distance_between_press_and_release > 10:
        return
    for i, text in enumerate(label_texts):
        bbox = text.get_window_extent()
        if not bbox.contains(event.x, event.y):
            continue
        label = text.get_text()
        print(f'Clicked on: {label}')
        idx = leaf_info['leaves'][i]  # 获取原始数据中的索引
        if idx in selected_indices:
            selected_indices.remove(idx)
            text.set_bbox(dict(facecolor='white', edgecolor='white', alpha=1))
            text.set_fontweight('normal')
        else:
            selected_indices.append(idx)
            text.set_bbox(dict(facecolor='yellow', edgecolor='white', alpha=1))
            text.set_fontweight('bold')
        
        ax2.draw_artist(text)
        canvas.blit(ax2.bbox)
        canvas.flush_events()
        fig.canvas.draw_idle()
        
        print(f"选择的索引: {selected_indices}")
        selected_labels = [file_names[idx] for idx in selected_indices]
        doc_names_to_abs_paths(selected_labels, selected_paths, all_direct_child_text_paths_include_and_above_current_dir)
        print(f"get selected_paths from on release: {selected_paths}")
        
        break

def on_release_right_click(event):
    global lasted_selected_paths

    if event.button != 3:  # 确保是右键
        return
    for text in label_texts:
        bbox = text.get_window_extent()
        if not bbox.contains(event.x, event.y):
            continue
        label = text.get_text()
        print(f'右键释放: {text.get_text()}')
        text.set_bbox(dict(facecolor='yellow', edgecolor='white', alpha=1))
        text.set_fontweight('bold')
        ax2.draw_artist(text)
        canvas.blit(ax2.bbox)
        canvas.flush_events()
        time.sleep(0.1)
        text.set_bbox(dict(facecolor='white', edgecolor='white', alpha=1))
        text.set_fontweight('normal')
        ax2.draw_artist(text)
        canvas.blit(ax2.bbox)
        canvas.flush_events()
        print(f"选择的标签: {lasted_selected_paths}")
        doc_names_to_abs_paths([label], lasted_selected_paths, all_direct_child_text_paths_include_and_above_current_dir)
        set_clip_data_from_latest_selected_paths(lasted_selected_paths)
        
        break
       
fig.canvas.mpl_connect('button_press_event', click_manager.on_press)
fig.canvas.mpl_connect('motion_notify_event', click_manager.on_move)
fig.canvas.mpl_connect('button_release_event', click_manager.on_release)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('button_release_event', on_release_right_click)


        
# 回调函数，用于选择标签框
def line_select_callback(eclick, erelease):
    global selected_indices
    y1, y2 = eclick.ydata, erelease.ydata
    print(f"选择区域: ({y1:.2f}) 到 ({y2:.2f})")
    
    changed_labels = []
    for i, (label, y, text) in enumerate(zip(ordered_labels, label_positions, label_texts)):
        if min(y1, y2) <= y <= max(y1, y2):
            idx = leaf_info['leaves'][i]  # 获取原始数据中的索引
            if idx in selected_indices:
                selected_indices.remove(idx)
                text.set_bbox(dict(facecolor='white', edgecolor='white', alpha=1))
                text.set_fontweight('normal')
            else:
                selected_indices.append(idx)
                text.set_bbox(dict(facecolor='yellow', edgecolor='white', alpha=1))
                text.set_fontweight('bold')
            changed_labels.append(text)
    
    print(f"选择的索引: {selected_indices}")
    
    # 只重绘更改的标签
    if changed_labels:
        for text in changed_labels:
            ax2.draw_artist(text)
        canvas.blit(ax2.bbox)
        canvas.flush_events()
        fig.canvas.draw_idle()
    
    global selected_paths
    selected_labels = [file_names[idx] for idx in selected_indices]
    doc_names_to_abs_paths(selected_labels, selected_paths, all_direct_child_text_paths_include_and_above_current_dir)

# 创建 RectangleSelector
rs = RectangleSelector(ax2, line_select_callback,
                       useblit=True,
                       button=[1],  # 仅响应鼠标左键
                       minspanx=10, minspany=10,
                       spancoords='pixels',
                       interactive=False,
                       drag_from_anywhere=False)

# fig.canvas.mpl_connect('button_press_event', on_click)
# fig.canvas.mpl_connect('button_release_event', on_click2)
# fig.canvas.mpl_connect('pick_event', on_click2)



# doc_names_to_abs_paths(['tmp.py', 'pdf-translator-main-en-cn.json.txt', 'cumulative_tokens_num_table.txt'])


# 调整窗口大小
# 设置最大窗口高度
# 获取桌面高度
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
print(f"screen_width: {screen_width}, screen_height: {screen_height}, dpi_scale: {dpi_scale}")
root.update()
window_width = int(canvas_frame.winfo_reqwidth())
window_height = int(screen_height * dpi_scale * 0.8) 
# window_width = canvas_frame.winfo_reqwidth()
# window_height = screen_height - 400
root.geometry(f"{window_width}x{window_height}")
root.state('zoomed')  # 在Windows上最大化窗口
# 或者使用
# root.attributes('-zoomed', True)  # 在Linux上最大化窗口
root.mainloop()



#### truncation 备忘：
# for index, path in enumerate(valid_paths, 1):
#     try:
#         # ... 其他处理代码 ...
#         file_names.append(os.path.basename(path)[:150])       
#         # ... 其他处理代码 ...
#     except Exception as e:
#         print(f"无法处理文件 {path}: {str(e)}")
#         failed_files.append(path)

# for label, y in zip(truncated_ordered_labels, label_positions):
#     text = ax2.text(0.01, y, label[0:1000], ha='left', va='center', fontsize=15)
#     label_texts.append(text)
#     text.set_bbox(dict(facecolor='white', edgecolor='white', alpha=1))