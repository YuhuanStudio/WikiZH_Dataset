import re
import json
import sys
import bz2
from opencc import OpenCC
from tqdm import tqdm


def extract_wiki_images(xml_path, output_json, max_images=None, lang='tw'):
    """
    從 XML 提取圖片資訊
    
    Args:
        xml_path: XML 文件路徑
        output_json: 輸出 JSONL 文件路徑
        max_images: 最大圖片數量
        lang: 語言版本 ('tw' for 繁體, 'cn' for 簡體)
    """
    # 根據語言選擇 OpenCC 轉換模式
    if lang == 'tw':
        # 簡體轉臺灣繁體
        cc = OpenCC('s2twp')
    else:  # lang == 'cn'
        # 臺灣繁體轉簡體
        cc = OpenCC('tw2sp')
    def parse_tag_types(raw_file_tag):
        # 提取 tag 里的参数类型（如 thumb、right、left、px 等）
        tag_content = raw_file_tag[8:-2]  # 去掉[[File:和]]
        parts = tag_content.split('|')
        types = []
        for part in parts[1:]:  # 第一个是文件名，跳过
            p = part.strip()
            # 只保留常见类型参数
            format_tags = {
                "thumb", "right", "left", "upright", "缩略图", "center", "none", "frameless", "border",
                "top", "bottom", "middle", "sub", "super", "baseline", "text-top", "text-bottom",
                "右上", "右下", "左上", "左下", "居中", "置中", "无边框", "有边框","右", "左", "上", "下", "中","thumbnail"
            }
            if p in format_tags:
                types.append(p)
            # 支持各种尺寸格式参数
            if re.search(r'\b\d{2,4}(x|×)\d{2,4}(px|像素|pixels)?\b', p):
                types.append(p)
            elif re.search(r'[0-9]+px', p):
                types.append(p)
        return types
    def clean_wiki_links(text):
        # 递归清理所有 wiki 链接格式
        # 只保留 wiki 链接显示文本
        def repl_link(m):
            if m.group(2):
                return m.group(2)
            else:
                return m.group(1)
        # 处理 Circa 模板 {{Circa|1000}} => 约1000
        text = re.sub(r'\{\{[Cc]irca\|([^|}]+)\}\}', r'约\1', text)
        # 处理 le/Le/link-en 模板 {{le|中文|英文}}、{{Le|中文|英文}}、{{link-en|中文|英文}}
        text = re.sub(r'\{\{([lL]e|[lL]ink-en)\|([^|}]+)\|[^}]+\}\}', lambda m: m.group(2), text)
        # 处理 convert 模板 {{convert|5774|ft|m|...}}
        def repl_convert(m):
            params = m.group(0)[2:-2].split('|')[1:]
            if len(params) >= 3:
                val = params[0]
                unit1 = params[1]
                unit2 = params[2]
                # 英尺/米等常见单位转换
                try:
                    v = float(val)
                    if unit1 == 'ft' and unit2 == 'm':
                        v2 = round(v * 0.3048)
                        return f"{val}英尺（或{v2}米）"
                    elif unit1 == 'm' and unit2 == 'ft':
                        v2 = round(v / 0.3048)
                        return f"{val}米（或{v2}英尺）"
                    # 可扩展更多单位
                except:
                    pass
                return f"{val}{unit1}"
            return m.group(0)
        text = re.sub(r'\{\{convert\|[^}]+\}\}', repl_convert, text)
        # 处理 lang-xx 模板 {{lang-xx|原文|translit=拼音}}
        def repl_lang(m):
            # 如果有 translit=，取其值，否则取第一个参数
            params = m.group(0)[2:-2].split('|')[1:]
            for p in params:
                if p.startswith('translit='):
                    return p.split('=',1)[1]
            return params[0] if params else ''
        text = re.sub(r'\{\{lang-[a-zA-Z0-9]+\|[^}]+\}\}', repl_lang, text)
        # 递归处理所有 wiki 链接
        pattern = re.compile(r'\[\[([^|\]]+)(?:\|([^\]]+))?\]\]')
        prev = None
        while text != prev:
            prev = text
            text = pattern.sub(repl_link, text)
        # 去除 <ref> 及其后内容
        text = re.sub(r'<ref[\s\S]*$', '', text)
        #&lt;ref 
        text = re.sub(r'&lt;ref[\s\S]*$', '', text)
        # 保留 <span> 标签内说明内容，只去除标签本身
        text = re.sub(r'&lt;span[^&]*?&gt;', '', text)
        text = re.sub(r'&lt;/span&gt;', '', text)
        # 还原 HTML 实体
        import html
        text = html.unescape(text)
        # 清理 html 标签
        text = re.sub(r'<[^>]+>', '', text)
        # 清理多余的 ' 及空格
        text = text.replace("'", "").strip()
        return text

    def parse_title(raw_file_tag, file_name):
        tag_content = raw_file_tag[8:-2]  # 去掉[[File:和]]
        parts = tag_content.split('|')
        format_tags = {
            "thumb", "right", "left", "upright", "缩略图", "center", "none", "frameless", "border",
            "top", "bottom", "middle", "sub", "super", "baseline", "text-top", "text-bottom",
            "右上", "右下", "左上", "左下", "居中", "置中", "无边框", "有边框", "右", "左", "上", "下", "中","thumbnail"
        }
        def is_format(p):
            p = p.strip()
            # 支持 upright=xxx 作为格式参数
            if p in format_tags:
                return True
            if re.match(r'^upright(=.+)?$', p):
                return True
            # 支持各种尺寸格式参数
            if re.search(r'\b\d{2,4}(x|×)\d{2,4}(px|像素|pixels)?\b', p):
                return True
            if re.search(r'[0-9]+px', p):
                return True
            return False
        # 优先说明内容，其次 alt=，都没有则用图片名
        alt_text = None
        desc_idx = None
        for i, p in enumerate(parts[1:], 1):
            p_strip = p.strip()
            # 记录 alt= 内容
            if p_strip.startswith('alt='):
                alt_text = p_strip[4:]
                continue
            # 跳过格式参数、link=、class=
            if is_format(p_strip) or p_strip.startswith('link=') or p_strip.startswith('class='):
                continue
            # 找到第一个说明内容
            if p_strip:
                desc_idx = i
                break
        # 如果有说明内容，且后面不是 alt= 或格式参数，则只取说明部分
        if desc_idx is not None:
            # 检查后续参数是否为 alt= 或格式参数
            for j in range(desc_idx + 1, len(parts)):
                next_p = parts[j].strip()
                if next_p.startswith('alt=') or is_format(next_p) or next_p.startswith('link=') or next_p.startswith('class='):
                    continue
                # 如果后面还有说明内容，合并
                if next_p:
                    return clean_wiki_links(parts[desc_idx] + '|' + '|'.join(parts[j:]))
            # 只取第一个说明内容
            return clean_wiki_links(parts[desc_idx])
        # 没有說明內容，優先 alt=，否則視為無描述（回傳 None）
        if alt_text:
            return clean_wiki_links(alt_text)
        # 不要使用檔名當作描述（因為沒有實際資訊），回傳 None 以便呼叫端決定跳過
        return None
    page_title_pattern = re.compile(r'<title>(.*?)<\/title>')
    images = []
    seen_files = set()
    file_idx = 1
    bytes_written = 0
    max_bytes = 500 * 1024 * 1024  # 500MB
    current_page = None

    def find_file_tags(text):
        results = []
        idx = 0
        while True:
            start = text.find('[[File:', idx)
            if start == -1:
                break
            depth = 1
            i = start + 7
            while i < len(text):
                if text[i:i+2] == '[[':
                    depth += 1
                    i += 2
                elif text[i:i+2] == ']]':
                    depth -= 1
                    i += 2
                    if depth == 0:
                        results.append(text[start:i])
                        break
                else:
                    i += 1
            idx = i
        return results

    is_bz2 = xml_path.endswith('.bz2')
    print("開始擷取圖片資訊...")
    processed_lines = 0
    if is_bz2:
        file_obj = bz2.open(xml_path, 'rt', encoding='utf-8')
    else:
        file_obj = open(xml_path, 'r', encoding='utf-8')

    # tqdm进度条按图片数显示
    with file_obj as f:
        pbar = tqdm(total=max_images if max_images else None, desc='擷取圖片數')
        for line in f:
            page_match = page_title_pattern.search(line)
            if page_match:
                current_page = page_match.group(1)
            file_tags = find_file_tags(line)
            for raw_file_tag in file_tags:
                if max_images is not None and len(seen_files) >= max_images:
                    break
                file_name_match = re.match(r'\[\[File:([^|\]]+)', raw_file_tag)
                if not file_name_match:
                    continue
                file_name = file_name_match.group(1).strip()
                fn_lower = file_name.lower()
                # 只保留圖片副檔名（擴充更多格式）
                image_exts = (
                    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tif', '.tiff', '.ico', '.jfif',
                    '.heic', '.heif', '.apng', '.avif', '.emf', '.wmf', '.pbm', '.pgm', '.ppm', '.xbm', '.xpm', '.gif'
                )
                if not fn_lower.endswith(image_exts):
                    continue
                if "no image" in fn_lower or "no free image" in fn_lower:
                    continue
                if file_name in seen_files:
                    continue  # 已經處理過，跳過重複
                # 修改 file_name 欄位為 image/file_name
                file_name_out = f"image/{file_name}"
                url = f"https://zh.wikipedia.org/wiki/Special:FilePath/{file_name.replace(' ', '_')}"
                title_raw = parse_title(raw_file_tag, file_name)
                # 如果沒有說明（title_raw 為 None），則跳過，不使用檔名當描述
                if not title_raw:
                    continue
                # 標記為已處理
                seen_files.add(file_name)
                title_conv = cc.convert(title_raw) if title_raw else title_raw
                images.append({
                    "url": url,
                    "title": title_conv,
                    "file_name": file_name,
                    "page": current_page,
                    "tag": raw_file_tag,
                    # "tag_types": parse_tag_types(raw_file_tag)
                })
                pbar.update(1)
                # 分割逻辑：只根據檔案大小分割
                jsonl_str = json.dumps(images[-1], ensure_ascii=False) + '\n'
                bytes_written += len(jsonl_str.encode('utf-8'))
                if bytes_written >= max_bytes:
                    out_name = f"{output_json.rsplit('.',1)[0]}_{file_idx}.jsonl"
                    with open(out_name, 'w', encoding='utf-8') as out:
                        for img in images:
                            out.write(json.dumps(img, ensure_ascii=False) + '\n')
                    print(f"分割儲存: {out_name}")
                    images = []
                    bytes_written = 0
                    file_idx += 1
            if max_images is not None and len(seen_files) >= max_images:
                break
        pbar.close()

    # 最后剩余未分割部分，输出为jsonl格式
    if images:
        out_name = f"{output_json.rsplit('.',1)[0]}_{file_idx}.jsonl"
        with open(out_name, 'w', encoding='utf-8') as out:
            for img in images:
                out.write(json.dumps(img, ensure_ascii=False) + '\n')
        print(f"儲存完成: {out_name}")