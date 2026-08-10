import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pangu  # 用於在中英文之間加空格
from opencc import OpenCC  # 用於簡繁轉換
import multiprocessing
from functools import partial


def clean_markdown_content(content):
    """
    清理 Markdown 文本內容
    - 移除 HTML 註釋、標籤
    - 移除 Markdown 的文件路徑註釋
    - 移除維基特殊標記
    - 徹底清理格式標記
    """
    # 移除 HTML 註釋和文件路徑
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    content = re.sub(r'filepath:.*?\n', '', content)
    
    # 移除維基信息框（Infobox）- 這是主要問題來源
    content = re.sub(r'\{\{[Ii]nfobox.*?\n\}\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\{\{基礎資訊.*?\n\}\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\{\{資訊框.*?\n\}\}', '', content, flags=re.DOTALL)
    #簡體
    content = re.sub(r'\{\{信息框.*?\n\}\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\{\{基礎資訊.*?\n\}\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\{\{資訊框.*?\n\}\}', '', content, flags=re.DOTALL)

    # 移除所有維基模板和特殊標記
    content = re.sub(r'\{\{.*?\}\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\{\|.*?\|\}', '', content, flags=re.DOTALL)  # 表格
    
    # 移除分類、文件、圖片鏈接
    content = re.sub(r'\[\[Category:.*?\]\]', '', content)
    content = re.sub(r'\[\[File:.*?\]\]', '', content)
    content = re.sub(r'\[\[Image:.*?\]\]', '', content)
    content = re.sub(r'\[\[檔案:.*?\]\]', '', content)
    content = re.sub(r'\[\[文件:.*?\]\]', '', content)
    content = re.sub(r'\[\[圖片:.*?\]\]', '', content)
    #簡體
    content = re.sub(r'\[\[图像:.*?\]\]', '', content)
    content = re.sub(r'\[\[图片:.*?\]\]', '', content)
    # 移除分組
    content = re.sub(r'\|group="[^"]*"\}\}', '', content)


    # 移除腳註引用和所有維基引用格式
    content = re.sub(r'\[\^.+?\]', '', content)
    content = re.sub(r'<ref.*?</ref>', '', content, flags=re.DOTALL)
    content = re.sub(r'<ref.*?/>', '', content)
    
    # 移除所有 HTML 標籤（包括屬性）
    content = re.sub(r'<[^>]+>', '', content)
    
    # 移除圖片標記和各種媒體標記
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    
    # 處理維基鏈接，保留顯示文字
    content = re.sub(r'\[\[([^|\]]+)\|([^|\]]+)\]\]', r'\2', content)  # [[鏈接|顯示文字]]
    content = re.sub(r'\[\[([^|\]]+)\]\]', r'\1', content)  # [[鏈接]]
    
    # 移除各種維基特殊語法
    content = re.sub(r"'''(.*?)'''", r'\1', content)  # 粗體
    content = re.sub(r"''(.*?)''", r'\1', content)    # 斜體
    content = re.sub(r'=====(.*?)=====', r'\1', content)  # 五級標題
    content = re.sub(r'====(.*?)====', r'\1', content)    # 四級標題
    content = re.sub(r'===(.*?)===', r'\1', content)      # 三級標題
    content = re.sub(r'==(.*?)==', r'\1', content)        # 二級標題
    content = re.sub(r'=(.*?)=', r'\1', content)          # 一級標題
    
    # 移除維基變數和參數
    content = re.sub(r'\{\{\{[^}]*\}\}\}', '', content)  # {{{變數}}}
      # 移除各種維基語法殘留
    content = re.sub(r'\|[^|=\n]*=', '', content)  # 移除 |參數=
    content = re.sub(r'[a-zA-Z_]+\s*=\s*[^\n]*', '', content)  # 移除 key=value
    
    # 只移除相連的'}} 是'、'}} 為'等（即'}}'後緊跟'是'）
    content = re.sub(r'\}\}\s*是', '是', content)  # 只處理'}} 是'
    content = re.sub(r'\}\}\s*為', '為', content)  # 只處理'}} 為'
    content = re.sub(r'\}\}\s*在', '在', content)  # 只處理'}} 在'
    content = re.sub(r'\}\}\s*有', '有', content)  # 只處理'}} 有'
    content = re.sub(r'\}\}\s*的', '的', content)  # 只處理'}} 的'
    content = re.sub(r'\}\}\s*', '', content)  # 移除孤立的 }}
    
    # 移除多餘的符號和格式
    content = re.sub(r'\|\s*\|', '', content)  # ||
    content = re.sub(r'\|\s*$', '', content, flags=re.MULTILINE)  # 行尾的 |
    content = re.sub(r'^\s*\|', '', content, flags=re.MULTILINE)  # 行首的 |
    content = re.sub(r'\}\}\s*$', '', content, flags=re.MULTILINE)  # 單獨的 }}
    
    return content


def extract_sections(content):
    """
    從 Markdown 提取有用的章節，將每個章節作為獨立的段落
    返回 [(title, content), ...] 列表，每個元組代表一個段落
    """
    sections_data = []
    skip_keywords = [
         # 參考相關
            "參考資料", "参考资料", "參考書目", "参考书目", "參考文獻", "参考文献", "參考來源", "参考来源", "參考", "参考", "參考資源", "参考资源", "參考網站", "参考网站", 
            "參考網頁", "参考网页", "參考工具", "参考工具", "參考連結", "参考链接", "參考著作", "参考著作", "參考書籍", "参考书籍", "延伸閱讀", "延伸阅读",
            # 註釋相關
            "註釋", "注释", "注釋", "註解", "注解", "腳註", "脚注", "註腳", "注脚", "注腳", "註記", "注记", "注記", "附註", "附注", "備註", "备注", "說明", "说明",
            # 外部連結
            "外部連結", "外部链接", "外部連結和參考資料", "外部链接和参考资料", "相關連結", "相关链接", "連結", "链接", "網站連結", "网站链接", "外部資源", "外部资源", 
            "外部網站", "外部网站", "官方網站", "官方网站", "官網", "官网", "相關網站", "相关网站", "官方連結", "官方链接",
            # 相關條目
            "參見", "参见", "參看", "参看", "相關條目", "相关条目", "相關主題", "相关主题", "相關詞條", "相关词条", "另見", "另见", "另看", "另看", "其他", "其他", 
            "相關文章", "相关文章", "相關頁面", "相关页面", "關聯條目", "关联条目", "類似條目", "类似条目",
            # 資料來源
            "引用", "引用", "資料來源", "资料来源", "來源", "来源", "文獻", "文献", "資源", "资源", "出處", "出处", "資料出處", "资料出处", "資料引用", "资料引用", 
            "研究書目", "研究书目", "書目", "书目", "文獻資料", "文献资料", "資料參考", "资料参考",
            # 其他無用章節
            "附錄", "附录", "附件", "附件", "圖片來源", "图片来源", "圖片引用", "图片引用", "圖片出處", "图片出处", "圖表來源", "图表来源", "圖表引用", "图表引用", 
            "圖表出處", "图表出处", "影片來源", "影片来源", "外部鏈接", "外部链接", "外部鏈結", "外部链结", "額外資源", "额外资源", "補充資料", "补充资料",
            "版權", "版权", "版權信息", "版权信息", "授權", "授权", "許可", "许可", "免責聲明", "免责声明", "聲明", "声明", "版本歷史", "版本历史", "修訂歷史", "修订历史",
            "編輯歷史", "编辑历史", "討論頁", "讨论页", "討論", "讨论", "Talk", "talk","外部连结"
    ]

    lines = content.split('\n')
    title_stack = []
    current_content = []
    last_level = None
    def is_title(line):
        m = re.match(r'^(#{1,6})\s*(.+)', line)
        if m:
            # 移除行首數字編號（如 4.1.1、4.1、4）
            title_text = re.sub(r'^\d+(?:\.\d+)*\s*', '', m.group(2)).strip()
            return len(m.group(1)), title_text
        return None, None

    for idx, line in enumerate(lines):
        level, title = is_title(line)
        if level:
            # 如果有累積內容，則保存上一章節
            if current_content:
                section_text = '\n'.join(current_content).strip()
                # 檢查標題是否跳過
                full_title = '-'.join(title_stack)
                should_skip = False
                for keyword in skip_keywords:
                    if full_title and keyword.lower() in full_title.lower():
                        should_skip = True
                        break
                if not should_skip and section_text and len(section_text) > 20:
                    sections_data.append((full_title, section_text))
                current_content = []
            # 更新標題棧
            if last_level is None:
                title_stack = [title]
            else:
                # 如果新標題層級 <= 棧長度，則彈出到該層級
                while len(title_stack) >= level:
                    title_stack.pop()
                title_stack.append(title)
            last_level = level
        else:
            current_content.append(line)
    # 最後一個章節
    if current_content and title_stack:
        section_text = '\n'.join(current_content).strip()
        full_title = '-'.join(title_stack)
        should_skip = False
        for keyword in skip_keywords:
            if full_title and keyword.lower() in full_title.lower():
                should_skip = True
                break
        if not should_skip and section_text and len(section_text) > 20:
            sections_data.append((full_title, section_text))
    return sections_data


def convert_md_to_text(md_content):
    """
    將 Markdown 轉換為純文本，生成高質量的描述性內容
    - 完全移除格式標記
    - 規範化標點符號
    - 確保文本流暢性
    """
    # 移除所有 Markdown 標題符號，但保留標題內容
    text = re.sub(r'^#{1,6}\s+(.*?)$', r'\1', md_content, flags=re.MULTILINE)
    
    # 更溫和地處理列表：使用「、」或「，」連接，避免在每項前面直接加句號
    # 1) 把開頭的列表項標記替換成頓號 + 內容（保留項目文字）
    text = re.sub(r'(?m)^\s*[\*\-\+]\s+(.*?)\s*$', r'、\1', text)
    text = re.sub(r'(?m)^\s*\d+\.\s+(.*?)\s*$', r'、\1', text)
    # 2) 如果列表項是行內延續（前一行有文字），將換行+頓號替換為逗號連接
    text = re.sub(r'([^\n])\n、', r'\1 •', text)
    # 3) 如果段落/文檔以列表開始，去除開頭的頓號（保留項目內容）
    text = re.sub(r'(?m)^\s*、', '', text)
    
    # 處理多層級列表
    text = re.sub(r'^\s+[\*\-\+]\s+(.*?)$', r'，\1', text, flags=re.MULTILINE)
    text = re.sub(r'^\s+\d+\.\s+(.*?)$', r'，\1', text, flags=re.MULTILINE)
    
    # 移除連結格式但保留文字
    text = re.sub(r'\[(.*?)\]\([^)]*\)', r'\1', text)
    
    # 移除粗體和斜體標記
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # 移除程式碼區塊
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    
    # 清理各種空格和特殊字符
    text = re.sub(r'[\u00A0\u2000-\u200D\u2060\uFEFF]', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'　+', '', text)
    
    # 清理空括號和各種符號組合
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\{\s*\}', '', text)
    text = re.sub(r'（\s*）', '', text)
    text = re.sub(r'「\s*」', '', text)
    text = re.sub(r'『\s*』', '', text)
    
    # 移除奇怪的符號組合和格式殘留
    text = re.sub(r'[（\(][；，\s]*[）\)]', '', text)  # (;) (, ) 等
    text = re.sub(r'[（\(][，\s]*[）\)]', '', text)   # (,) 等
    text = re.sub(r'[（\(]\s*[；;]\s*[）\)]', '', text)  # (;) 等
    
    # 清理維基語法殘留
    text = re.sub(r'[a-zA-Z_]+\s*=\s*', '', text)  # key= 格式
    text = re.sub(r'\|\s*[a-zA-Z_]+\s*=', '', text)  # |key= 格式
    text = re.sub(r'=\s*[^\n=]*\n', '\n', text)  # =value 格式
    
    # 移除孤立的等號、豎線等符號
    text = re.sub(r'^\s*[=|]\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+[=|]\s+', ' ', text)
      # 清理重複標點
    text = re.sub(r'[,，]{2,}', '，', text)
    text = re.sub(r'[.。]{2,}', '。', text)
    text = re.sub(r'[;；]{2,}', '；', text)
    text = re.sub(r'[!！]{2,}', '！', text)
    text = re.sub(r'[?？]{2,}', '？', text)
    
    # 處理章節編號和標題
    text = re.sub(r'###\s*\d+\.\d+\s+', '', text)  # 移除 ### 1.1 格式
    text = re.sub(r'##\s*\d+\.\d+\s+', '', text)   # 移除 ## 1.1 格式
    text = re.sub(r'#\s*\d+\.\d+\s+', '', text)    # 移除 # 1.1 格式
    # 不再全局移除行首小數開頭，避免誤刪如 "120.2型機車" 這類內容
    
    # 處理連續的章節標記
    text = re.sub(r'\\n\d+\.\d+\s+', '，', text)   # 將 \n1.2 轉為逗號
    text = re.sub(r'\n\d+\.\d+\s+', '，', text)    # 將換行+編號轉為逗號
    
    # 修正標點符號空格
    text = re.sub(r'([，。；！？])\s+', r'\1', text)
    text = re.sub(r'\s+([，。；！？])', r'\1', text)
    
    # 清理奇怪的字符組合
    text = re.sub(r'[（\(][a-zA-Z=\s]*[）\)]', '', text)  # (a=xxx) 等
    
    # 轉換 HTML 實體
    html_entities = {
        '&lt;': '<', '&gt;': '>', '&amp;': '&', '&quot;': '"', 
        '&apos;': "'", '&nbsp;': ' ', '&#39;': "'", '&mdash;': '—',
        '&ndash;': '–', '&ldquo;': '"', '&rdquo;': '"'
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)
    
    return text


def normalize_text(text, title=""):
    """
    標準化文本，使其更適合 LLM 訓練
    - 移除過長的空白和重複內容
    - 確保自然段落結構
    - 生成高質量的描述性文本
    返回 (final_title, normalized_text) 元組
    """
    extracted_title = title
    lines = text.split('\n')
    if len(lines) > 1 and not extracted_title:
        first_line = lines[0].strip()
        clean_first_line = re.sub(r'^#{1,6}\s*', '', first_line).strip()
        if len(clean_first_line) < 20 and not any(punct in clean_first_line for punct in '。！？；，：'):
            extracted_title = clean_first_line
            text = '\n'.join(lines[1:])
    
    # 移除多餘的空白字符
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    
    # 規範化段落結構
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 移除單字符行（通常是格式問題）
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) >= 2:  # 至少2個字符才保留
            clean_lines.append(stripped)
    
    text = '\n'.join(clean_lines)
    
    # 合併過短的段落
    paragraphs = text.split('\n\n')
    merged_paragraphs = []
    temp_paragraph = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # 如果當前段落很短，嘗試與下一段合併
        if len(paragraph) < 100 and temp_paragraph:
            temp_paragraph += "，" + paragraph
        else:
            if temp_paragraph:
                merged_paragraphs.append(temp_paragraph)
            temp_paragraph = paragraph
    
    if temp_paragraph:
        merged_paragraphs.append(temp_paragraph)
      # 確保句子結構合理
    final_paragraphs = []
    for paragraph in merged_paragraphs:
        # 修正句子結構
        paragraph = re.sub(r'([。！？])\s*([，、])', r'\1', paragraph)
        paragraph = re.sub(r'([，、])\s*([。！？])', r'\2', paragraph)
        
        # 處理重複的標點符號
        paragraph = re.sub(r'[。]{2,}', '。', paragraph)
        paragraph = re.sub(r'[，]{2,}', '，', paragraph)
        
        # 移除段落中的維基格式殘留
        paragraph = re.sub(r'方面，###\s*\d+\.\d+\s+', '方面，', paragraph)
        paragraph = re.sub(r'方面，##\s*\d+\.\d+\s+', '方面，', paragraph) 
        paragraph = re.sub(r'方面，#\s*\d+\.\d+\s+', '方面，', paragraph)
        
        # 處理可以指：這種格式
        paragraph = re.sub(r'可以指：。', '可以指：', paragraph)
        paragraph = re.sub(r'包括：。', '包括：', paragraph)
        
        # 確保段落以句號結尾（除非是列舉格式）
        if paragraph and not paragraph.endswith(('。', '！', '？', '：')):
            paragraph += '。'
        
        final_paragraphs.append(paragraph)
    
    result = '\n\n'.join(final_paragraphs).strip()

    # 最終清理格式問題
    result = re.sub(r'。。+', '。', result)  # 移除重複句號
    result = re.sub(r'，，+', '，', result)  # 移除重複逗號
    result = re.sub(r'方面，\s*$', '', result, flags=re.MULTILINE)  # 移除單獨的"方面，"


    # 直接移除所有\n，全部替換為空格
    result = re.sub(r'\n+', ' ', result)
    # 再將多餘的空白合併
    result = re.sub(r'[ \t]+', ' ', result)
    result = result.strip()

    # 移除表格/infobox字段串連（如 date_event1 p1 flag_p1 ...）
    result = re.sub(r'(?:[a-zA-Z_]+\d*\s+){4,}', '', result)
    # 移除 url 字段
    result = re.sub(r'[a-zA-Z_]+_url\s*https?://\S+', '', result)
    # 移除括號內只有標點、逗號、空白或為空的括號
    result = re.sub(r'（[，；、\s]*）', '', result)
    result = re.sub(r'\([，；、\s]*\)', '', result)
    # 移除括號內只有非字母數字漢字的內容
    result = re.sub(r'（[^\u4e00-\u9fa5a-zA-Z0-9]{1,10}）', '', result)
    result = re.sub(r'\([^\u4e00-\u9fa5a-zA-Z0-9]{1,10}\)', '', result)
    # 移除逗號、分號、頓號等開頭的句子
    result = re.sub(r'([。！？])\s*[，；、]+', r'\1', result)
    result = re.sub(r'^[，；、]+', '', result)
    # 修正 ：。 為 ：
    result = re.sub(r'：。', '：', result)
    # 移除多餘句號
    result = re.sub(r'([。！？])\1+', r'\1', result)
    # 移除孤立的分號、逗號、頓號
    result = re.sub(r'[，；、]{2,}', '，', result)
    result = re.sub(r'([。！？])[,，；、]+', r'\1', result)
    # 移除以逗號開頭的內容
    result = re.sub(r'^，+', '', result)
    # 移除括號內結尾為逗號、僅標點、僅英文、僅引號、空白等內容
    result = re.sub(r'（[^）\u4e00-\u9fa5]*[，,。；、\s"“”]*）', '', result)
    result = re.sub(r'\([^\)\u4e00-\u9fa5]*[，,。；、\s"“”]*\)', '', result)
    # 移除括號內僅有英文、標點、空白、引號
    result = re.sub(r'（[a-zA-Z\s,，。；、"“”]*）', '', result)
    result = re.sub(r'\([a-zA-Z\s,，。；、"“”]*\)', '', result)
    # 進一步清理括號內僅標點、逗號、空白或無內容
    result = re.sub(r'（[，、；：\s]*）', '', result)
    result = re.sub(r'\([，、；：\s]*\)', '', result)
    # 移除括號內僅有英文逗號的情況
    result = re.sub(r'（[a-zA-Z,\s]*，?）', '', result)
    result = re.sub(r'\([a-zA-Z,\s]*，?\)', '', result)
    # 移除冒號、頓號、逗號後直接句號
    result = re.sub(r'([：、，])。', r'\1', result)
    # 移除冒号后无内容或仅标点的情况，但保留正常的冒号用法
    result = re.sub(r'[：:]\s*$', '', result)  # 只移除行尾的孤立冒号
    result = re.sub(r'[：:]\s*[，、；\s]+', '：', result)  # 移除冒号后的无意义标点，但保留冒号
    # 移除“可能是：”或“可能意指：”等無意義開頭
    result = re.sub(r'可能[是為意指][：:]', '', result)
    #簡體
    result = re.sub(r'可能[是为意指][：:]', '', result)
    # 移除stat2- stat3-等無意義字段
    result = re.sub(r'(?:stat\d+-\s*)+', '', result)
    # 移除奇怪的符號串（如!utSTRf!~MFADEf...）
    result = re.sub(r'!+u?t?STR\w+!~\w+\\?u?t?STR\w+!~\w+~+', '', result)
    # 移除孤立的圖片文件名（如 .jpg。 .png。等）
    result = re.sub(r'\b\w+\.(jpg|png|jpeg|gif|svg|bmp|webp)[。.]', '', result, flags=re.IGNORECASE)
    # 移除多餘空白
    result = re.sub(r'\s+', ' ', result).strip()

    # 最後檢查：如果文本開頭仍然是單獨的標題行，再次移除
    lines = result.split('\n')
    if len(lines) > 2:
        first_line = lines[0].strip()
        # 移除 Markdown 標題符號
        clean_first_line = re.sub(r'^#{1,6}\s*', '', first_line).strip()
        if (len(clean_first_line) < 15 and 
            not any(punct in clean_first_line for punct in '。！？；，：') and
            len(lines[1].strip()) == 0):  # 第二行是空行
            if not extracted_title:  # 只有當沒有提取到標題時才使用第一行
                extracted_title = clean_first_line
            result = '\n'.join(lines[2:]).strip()
    # ====== 删除除中英文以外的语言文字 ======
    # 仅保留中英文、阿拉伯数字、常用标点
    # 中文：\u4e00-\u9fff，英文：a-zA-Z，数字：0-9，常用标点
    # 僅刪除孤立符號，保留語句結構標點（如《》、“”、（）等）
    # 只移除非語義性符號（如 @ # $ % ^ & * 等），保留所有常用中文標點
    # 保留數字中的小數點，只刪除孤立符號
    result = re.sub(r'(?<!\d)[\.](?!\d)', '', result)  # 只刪除非數字間的小數點
    result = re.sub(r'[\@#\$%\^&\*=\|~`<>]', '', result)


    # ====== 中英/数字混合处理 ======：
    # 多余空格合并
    result = re.sub(r'\s+', ' ', result).strip()

    # ====== 修正中英文标点与中英文之间的空格问题 ======
    # 句末中文标点后紧跟空格+中英文，去除空格
    result = re.sub(r'([。！？；：,.!?;:])\s+([\u4e00-\u9fffA-Za-z])', r'\1\2', result)
    # 句首中英文与标点之间的多余空格
    result = re.sub(r'([\u4e00-\u9fffA-Za-z])\s+([。！？；：,.!?;:])', r'\1\2', result)
    # 再次合并多余空白
    result = re.sub(r'\s+', ' ', result).strip()
    # ====== 彻底移除侧边栏字段串连、表格线、无意义字段、残留表格内容 ======
    # 1. 移除典型侧边栏字段串连（如 image 2 imagecityelevationmetricr 1 r 1-length... stat ...）
    result = re.sub(r'(?:[a-zA-Z_\-]+\s+){3,}(?:是|為|在|有|的)', '', result)
    # 2. 移除典型 infobox/表格字段串连（如 blankname 8 1 successor 8 ...）
    result = re.sub(r'(?:[a-zA-Z_\-]+\s+\d+\s+){2,}', '', result)
    # 3. 移除典型表格线（如 ----、---、——、—、-、_、= 等连续符号）
    result = re.sub(r'[\-—_=]{3,}', '', result)
    # 4. 移除典型比赛表格字段（如小组赛 3.A 组 ... 淘汰赛 ... 决赛等）
    result = re.sub(r'(小組賽|分組賽|預賽|淘汰賽|復活賽|半決賽|銅牌賽|決賽)[^。]*', '', result)
    #簡體
    result = re.sub(r'(小组赛|分组赛|预赛|淘汰赛|复活赛|半决赛|铜牌赛|决赛)[^。]*', '', result)
    # 5. 移除“date of 居住地 ...”等字段串连
    result = re.sub(r'(date of|居住地|出生地|國籍|職業|政黨|學歷|信仰|配偶|父母|子女|母校|學校|畢業院校|任期|前任|繼任|出生)[^。]{0,30}[。]', '', result)
    #簡體
    result = re.sub(r'(date of|居住地|出生地|国籍|职业|政党|学历|信仰|配偶|父母|子女|母校|学校|毕业院校|任期|前任|继任|出生)[^。]{0,30}[。]', '', result)
    # 6. 移除“stat ...”等无意义字段
    result = re.sub(r'stat\s+[a-zA-Z0-9_\-]+', '', result)

    # ====== 进一步清理边缘异常 ======
    # 1. 移除括号内仅有非中文内容（如仅希腊字母、引号、空白、标点、英文、数字等）
    result = re.sub(r'[（\(][^\u4e00-\u9fa5]{1,20}[）\)]', '', result)
    # 2. 移除括号内内容为“或”、“和”、“與”、“及”等无意义连接词
    result = re.sub(r'[（\(][或和與及,，、\s]{1,5}[）\)]', '', result)
    # 3. 移除括号内内容为“See ...”等英文注释
    result = re.sub(r'[（\(][sS]ee[^）\)]*[）\)]', '', result)
    # 4. 移除括号内开头即为标点的情况（如（，）（；）（、）等）
    result = re.sub(r'[（\(][，；、。:：\s]+[）\)]', '', result)
    # 5. 移除孤立的引号、括号、分号、冒号（僅當前後為空白或標點時）
    result = re.sub(r'(?<=\s)["“”‘’\'\':：;；](?=\s)', '', result)
    result = re.sub(r'^["“”‘’\'\':：;；]+', '', result)
    result = re.sub(r'["“”‘’\'\':：;；]+$', '', result)
    # 6. 移除“image2 image- city- ... 是/为/在/有/的”这类字段串连
    result = re.sub(r'(?:[a-zA-Z_\-]+\s+){4,}(是|為|在|有|的)', '', result)
    # 1. 移除3个及以上英文/数字/下划线/连字符的字段串连（无论是否有“是/為/在/有/的”结尾）
    result = re.sub(r'(?:[a-zA-Z0-9_\-]+\s+){3,}', '', result)
    # 7. 移除“* ”、“- ”、“+ ”等列表符号残留
    result = re.sub(r'[\*\-\+]\s+', '', result)
    # 8. \"文字\" 改成 “文字”
    result = re.sub(r'["“”‘’](.*?)["“”‘’]', r'“\1”', result)
    #移除孤立的“”
    result = re.sub(r'(?<=\s)“(?=\s)', '', result)
    result = re.sub(r'(?<=\s)”(?=\s)', '', result)

    # 多余空格合并
    result = re.sub(r'\s+', ' ', result).strip()

    # 再次合并多余空白
    result = re.sub(r'\s+', ' ', result).strip()

    # 强力删除开头的各种不合适标点符号（重复多次以确保完全清理）
    unwanted_start_punctuation = ['。', '，', '、', '；', '：', '！', '？', '(', ')', '（', '）', '[', ']', '{', '}', '"', "'", '"', '"', ''', ''']
    
    # 重复清理直到开头没有不合适的标点
    max_iterations = 10  # 防止无限循环
    iteration = 0
    while result and iteration < max_iterations:
        original_result = result
        # 删除开头的标点符号
        for punct in unwanted_start_punctuation:
            if result.startswith(punct):
                result = result[len(punct):].strip()
                break
        
        # 如果没有变化，说明已经清理完毕
        if result == original_result:
            break
        iteration += 1

    # 只刪除極短且無語義的開頭（如單一標點或空白），不刪除常見連接詞
    result = re.sub(r'^[，、；：。!?\s]+', '', result)

    # 最后再次删除可能残留的开头标点符号
    while result and any(result.startswith(punct) for punct in unwanted_start_punctuation):
        for punct in unwanted_start_punctuation:
            if result.startswith(punct):
                result = result[len(punct):].strip()
                break

    return extracted_title, result


def quality_check_text(text):
    """
    檢查文本質量，過濾低質量內容（針對段落級別調整）
    """
    if not text or len(text.strip()) < 30:  # 段落可以更短
        return False
    
    # 檢查是否包含過多的特殊字符、數字或英文
    special_chars = len(re.findall(r'[^\w\s\u4e00-\u9fff，。！？；：「」『』（）]', text))
    if special_chars / len(text) > 0.3:  # 特殊字符超過30%
        return False
    
    # 檢查是否包含維基語法殘留
    wiki_patterns = [
        r'[a-zA-Z_]+\s*=',  # key=
        r'\{\{.*?\}\}',     # {{template}}
        r'\|\s*[a-zA-Z]',   # |variable
        r'settlement_type',  # infobox 殘留
        r'subdivision_',     # infobox 殘留
        r'image_',          # infobox 殘留
        r'pushpin_',        # infobox 殘留
        r'population_',     # infobox 殘留
        r'blank\d*_',       # infobox 殘留
        r'###\s*\d+\.\d+',  # 章節標記殘留
        r'##\s*\d+\.\d+',   # 章節標記殘留
        r'\}\}.*?是',       # }} 文字 是
    ]
    
    for pattern in wiki_patterns:
        if re.search(pattern, text):
            return False
    
    # 檢查是否有過多的重複標點
    if text.count('。。') > 2 or text.count('，，') > 2:
        return False
    
    # 檢查是否包含過多的章節編號（段落級別放寬限制）
    section_numbers = len(re.findall(r'\d+\.\d+', text))
    if section_numbers > 30:  # 段落級別減少到5個
        return False
    
    # 檢查是否有足夠的中文內容（段落級別降低要求）
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    if chinese_chars < 15:  # 減少到15個中文字符
        return False
    
    # 檢查中文字符比例
    if chinese_chars / len(text) < 0.25:  # 降低到25%
        return False
    
    # 檢查句子結構是否合理（段落級別可以只有1個句子）
    sentences = re.split(r'[。！？]', text)
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    if len(valid_sentences) < 1:  # 至少1個有效句子
        return False
    
    # 檢查是否包含過多的括號內容（通常是格式殘留）
    parentheses_content = re.findall(r'[（\(][^）\)]*[）\)]', text)
    if len(''.join(parentheses_content)) > len(text) * 0.3:  # 放寬到30%
        return False
    
    # 檢查是否有過多的單字符行（格式問題）
    lines = text.split('\n')
    single_char_lines = [line for line in lines if len(line.strip()) == 1]
    if len(single_char_lines) > 2:  # 段落級別減少到2個
        return False
    
    return True


def process_markdown_file(file_path, target_keywords=None, use_opencc=True, min_length=10, lang='tw'):
    """處理單個 Markdown 文件，生成高質量的預訓練數據，每個段落作為獨立條目
    
    Args:
        file_path: Markdown 文件路径
        target_keywords: 目标关键字列表，如果标题包含其中任一关键字则采集
        lang: 語言版本 ('tw' for 繁體, 'cn' for 簡體)
    """
    try:
        file_path_str = str(file_path)
        
        # 讀取文件內容
        try:
            with open(file_path_str, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path_str, 'r', encoding='utf-8-sig') as f:
                content = f.read()
        
        if not content.strip():
            return []
        
        # 使用 OpenCC 進行簡繁轉換（根據 lang 參數）
        if use_opencc:
            # 根據語言選擇轉換模式
            if lang == 'tw':
                # 簡體轉臺灣繁體
                cc = OpenCC('s2twp')
            else:  # lang == 'cn'
                # 臺灣繁體轉簡體
                cc = OpenCC('tw2sp')
            converted_text = cc.convert(content)
        else:
            converted_text = content
        
        # 使用 pangu 進行中英文間空格處理
        spaced_text = pangu.spacing(converted_text)
        
        
        # 清理 Markdown 內容
        cleaned_content = clean_markdown_content(spaced_text)
        
        # 提取有用的章節，返回 [(title, content), ...] 列表
        sections_data = extract_sections(cleaned_content)
        
        # 處理每個段落
        results = []
        for section_title, section_content in sections_data:
            # 轉換為純文本
            text = convert_md_to_text(section_content)
            
            # 標準化文本
            final_title, normalized_text = normalize_text(text, section_title)
            
            # 最終清理
            final_text = normalized_text.strip()
            
            # 質量檢查
            if not quality_check_text(final_text):
                continue
            
            # 長度檢查
            if len(final_text) < min_length:  # 使用參數控制最小段落長度
                continue
            
            # 如果沒有提取到標題，使用章節標題
            if not final_title:
                final_title = section_title
                
            # 检查标题是否包含目标关键字（如果设置了筛选条件）
            if target_keywords:
                contains_keyword = any(keyword in final_title for keyword in target_keywords)
                if not contains_keyword:
                    continue
              # 創建高質量的預訓練樣本
            results.append({
                "title": final_title,
                "text": final_text,
                "source_file": Path(file_path).name,  # 添加源文件名
            })
        
        return results
    
    except Exception as e:
        print(f"處理文件 {file_path_str} 時出錯: {type(e).__name__}: {e}")
        return []


def process_directory(input_dir, output_file, output_format="jsonl", max_size_mb=500, num_workers=None, max_files=None, target_keywords=None, use_opencc=True, min_length=10, lang='tw'):
    """處理目錄中的所有 Markdown 文件，並分割大小超過 max_size_mb 的輸出文件
    
    Args:
        input_dir: 输入目录
        output_file: 输出文件路径
        output_format: 输出格式 (json/jsonl)
        max_size_mb: 最大文件大小(MB)
        num_workers: 工作进程数
        max_files: 最大处理文件数
        target_keywords: 目标关键字列表，如果标题包含其中任一关键字则采集
        lang: 語言版本 ('tw' for 繁體, 'cn' for 簡體)
    """
    input_path = Path(input_dir)
    md_files = list(input_path.glob('**/*.md'))
    
    # 如果指定了最大文件數，限制處理的文件數量
    if max_files and max_files > 0:
        md_files = md_files[:max_files]
        print(f"將處理前 {max_files} 個 Markdown 文件 (共找到 {len(md_files)} 個)")
    
    # 如果未指定工作進程數，則使用可用 CPU 核心數量
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    print(f"使用 {num_workers} 個 CPU 核心並行處理")
    
    if target_keywords:
        print(f"標題篩選關鍵字: {target_keywords}")
    
    # 創建輸出目錄(如果不存在)
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    output_base_name, output_ext = os.path.splitext(output_file)
      # 使用多進程處理文件，传递关键字参数
    # 將 use_opencc 和 lang 傳遞到子進程
    process_func = partial(process_markdown_file, target_keywords=target_keywords, use_opencc=use_opencc, min_length=min_length, lang=lang)
    with multiprocessing.Pool(processes=num_workers) as pool:
        # 用 tqdm 顯示進度
        file_results = list(tqdm(
            pool.imap(process_func, md_files),
            total=len(md_files),
            desc="處理 Markdown 文件"
        ))
    
    # 展平結果列表（每個文件可能產生多個段落）
    results = []
    for file_result in file_results:
        if file_result:  # file_result 現在是一個列表
            results.extend(file_result)
    
    print(f"成功處理 {len(file_results)} 個文件，生成 {len(results)} 個段落條目")
    
    # 寫入結果
    file_count = 1
    current_size = 0
    max_size_bytes = max_size_mb * 1024 * 1024
    current_output_file = f"{output_base_name}_part{file_count}{output_ext}"
    
    try:
        if output_format.lower() == "jsonl":
            with open(current_output_file, 'w', encoding='utf-8') as out_file:
                for result in tqdm(results, desc="寫入 JSONL 文件"):
                    json_line = json.dumps(result, ensure_ascii=False) + '\n'
                    line_size_bytes = len(json_line.encode('utf-8'))
                    
                    # 檢查是否需要開新檔案
                    if current_size + line_size_bytes > max_size_bytes:
                        out_file.close()
                        file_count += 1
                        # 修正：確保文件名格式正確，避免 .json_part1.json 的問題
                        current_output_file = f"{output_base_name}_part{file_count}{output_ext}"
                        out_file = open(current_output_file, 'w', encoding='utf-8')
                        current_size = 0
                        current_size_mb = current_size / (1024 * 1024)
                        print(f"創建新文件: {current_output_file} (上一個文件大小: {current_size_mb:.2f} MB)")
            with open(current_output_file, 'w', encoding='utf-8') as out_file:
                for result in tqdm(results, desc="寫入 JSONL 文件"):
                    json_line = json.dumps(result, ensure_ascii=False) + '\n'
                    line_size_bytes = len(json_line.encode('utf-8'))
                    
                    # 檢查是否需要開新檔案
                    if current_size + line_size_bytes > max_size_bytes:
                        out_file.close()
                        file_count += 1
                        current_output_file = f"{output_base_name}_part{file_count}{output_ext}"
                        out_file = open(current_output_file, 'w', encoding='utf-8')
                        current_size = 0
                        current_size_mb = current_size / (1024 * 1024)
                        print(f"創建新文件: {current_output_file} (上一個文件大小: {current_size_mb:.2f} MB)")
                    
                    # 寫入當前行並更新大小
                    out_file.write(json_line)
                    current_size += line_size_bytes
            
            current_size_mb = current_size / (1024 * 1024)
            print(f"完成處理，共分割為 {file_count} 個文件，最後文件大小: {current_size_mb:.2f} MB")
        else:
            # JSON 格式處理
            current_batch = []
            current_batch_size = 0
            batch_num = 1
            
            for item in tqdm(results, desc="分割 JSON 文件"):
                item_json = json.dumps(item, ensure_ascii=False)
                item_size = len(item_json.encode('utf-8'))
                
                # 如果加入此項目會超出限制，則先保存當前批次
                if current_batch and (current_batch_size + item_size + 10) > max_size_bytes:
                    # 修正：確保文件名格式正確，避免 .json_part1.json 的問題
                    batch_file = f"{output_base_name}_part{batch_num}{output_ext}"
                    with open(batch_file, 'w', encoding='utf-8') as f:
                        json.dump(current_batch, f, ensure_ascii=False, indent=2)
                    batch_size_mb = current_batch_size / (1024 * 1024)
                    print(f"保存批次 {batch_num} 至 {batch_file} (大小: {batch_size_mb:.2f} MB)")
                    
                    current_batch = [item]
                    current_batch_size = item_size
                    batch_num += 1
                else:
                    current_batch.append(item)
                    current_batch_size += item_size
            
            # 保存最後一個批次
            if current_batch:
                batch_file = f"{output_base_name}_part{batch_num}{output_ext}"
                with open(batch_file, 'w', encoding='utf-8') as f:
                    json.dump(current_batch, f, ensure_ascii=False, indent=2)
                batch_size_mb = current_batch_size / (1024 * 1024)
                print(f"保存最後批次 {batch_num} 至 {batch_file} (大小: {batch_size_mb:.2f} MB)")
            
            print(f"完成處理，共分割為 {batch_num} 個文件")
    
    except Exception as e:
        print(f"處理過程中發生錯誤: {type(e).__name__}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='將 Wiki Markdown 文件轉換為高質量預訓練用的 JSON 格式（每個段落作為獨立條目）')
    parser.add_argument('--input_dir', type=str, default='./markdown/202601',
                        help='包含 Markdown 文件的目錄 (預設: ./markdown/202601)')
    parser.add_argument('--output_file', type=str, default='./output/tw/wiki_pretrain.json',
                        help='輸出的文件路徑 (預設: ./output/tw/wiki_pretrain.json，會自動分割為 part1, part2...)')
    parser.add_argument('--format', type=str, choices=['json', 'jsonl'], default='json',
                        help='輸出格式: json (單一陣列) 或 jsonl (每行一個 JSON 物件)')
    parser.add_argument('--max_size_mb', type=int, default=500,
                        help='每個輸出文件的最大大小(MB)，超過此大小將分割為多個文件')
    parser.add_argument('--workers', type=int, default=19,
                        help='同時處理文件的工作進程數量，預設為系統CPU核心數')
    parser.add_argument('--max_files', type=int, default=None,
                        help='要處理的最大文件數量（用於測試）')
    parser.add_argument('--min_length', type=int, default=30,
                        help='段落最小長度要求（字符數）')
    parser.add_argument('--keywords', type=str, nargs='*', default=None,
                        help='標題篩選關鍵字列表，只有標題包含其中任一關鍵字的條目才會被採集')
    # OpenCC 開關: 預設啟用，提供 --no-opencc 可關閉
    parser.add_argument('--opencc', dest='opencc', action='store_true',
                        help='啟用 OpenCC 簡繁轉換 (預設)')
    parser.add_argument('--no-opencc', dest='opencc', action='store_false',
                        help='停用 OpenCC 簡繁轉換')
    parser.set_defaults(opencc=True)
    # 語言選項: tw=繁體中文, cn=簡體中文
    parser.add_argument('--lang', type=str, choices=['tw', 'cn'], default='tw',
                        help='輸出語言版本 (tw=繁體中文, cn=簡體中文, 預設: tw)')
    args = parser.parse_args()
    
    print("=== Wiki Markdown 到高質量預訓練數據轉換工具（段落級別） ===")
    print(f"輸入目錄: {args.input_dir}")
    print(f"輸出文件: {args.output_file}")
    print(f"輸出格式: {args.format}")
    print(f"最小段落長度: {args.min_length} 字符")
    print(f"工作進程數: {args.workers or '自動'}")
    if args.keywords:
        print(f"標題篩選關鍵字: {args.keywords}")
    else:
        print("標題篩選: 無（處理所有條目）")
    print(f"OpenCC 簡繁轉換: {'啟用' if args.opencc else '停用'}")
    print(f"語言版本: {'繁體中文' if args.lang == 'tw' else '簡體中文'}")
    print("注意：每個 Markdown 文件的每個段落將作為獨立的數據條目")
    
    process_directory(args.input_dir, args.output_file, args.format,
                     args.max_size_mb, args.workers, args.max_files, args.keywords, args.opencc, args.min_length, args.lang)
    
    print("=== 處理完成 ===")
    print("建議檢查輸出文件的質量，確認是否適合用於預訓練")
    print("每個段落已被分割為獨立條目，可以增加數據多樣性")


if __name__ == "__main__":
    main()