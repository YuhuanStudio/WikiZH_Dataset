import re
import json
import bz2
from tqdm import tqdm

from wiki_parser import expand_inline_templates, resolve_variant_markers
from wiki_text import convert_script, normalize_whitespace, strip_leftover_markup


def extract_wiki_images(xml_path, output_json, max_images=None, lang='tw'):
    """
    從 XML 提取圖片資訊
    
    Args:
        xml_path: XML 文件路徑
        output_json: 輸出 JSONL 文件路徑
        max_images: 最大圖片數量
        lang: 語言版本 ('tw' for 繁體, 'cn' for 簡體)
    """
    # 繁簡轉換走與文字資料集相同的路徑（維基官方轉換表 + 台灣慣用詞白名單）。
    # 這裡原本用 OpenCC 的 s2twp／tw2sp，跟正文用的是兩套不同的轉換規則，
    # 同一個詞在圖片資料集與文字資料集會長得不一樣。
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
    def clean_wiki_links(text, page_title=''):
        """
        清理圖說，直接沿用正文那一套規則。

        原本這裡自己寫了一份較弱的清理邏輯，漏掉 {{PAGENAMEBASE}}、{{Tsl}}、
        {{le}} 等模板（實測 10.8% 的圖說帶著未展開的模板），還會把所有單引號
        一併刪掉（`'85`、英文所有格都會壞掉）。改成呼叫正文的模板展開與殘留
        標記清理，兩邊規則一致，日後修一次就好。
        """
        # {{PAGENAMEBASE}}／{{PAGENAME}} 渲染出來是條目名
        for magic in ('{{PAGENAMEBASE}}', '{{PAGENAME}}', '{{BASEPAGENAME}}'):
            text = text.replace(magic, page_title)
        # 圖說裡沒收尾的 <ref 之後全是引用，一路截掉（正文不能這樣做，
        # 但圖說是單一句子，後面不會再有內容）
        text = re.sub(r'(?is)<ref(?![a-z]).*$', '', text)
        text = re.sub(r'(?is)&lt;ref(?![a-z]).*$', '', text)
        text = resolve_variant_markers(text)
        text = expand_inline_templates(text)
        text = strip_leftover_markup(text)
        # 還留著 `[[`／`{{` 表示原始碼裡的標記本來就沒收尾（圖說被截斷），
        # 後面已經不是可讀的圖說，比照 <ref 的處理直接截掉
        cut = min((i for i in (text.find('[['), text.find('{{')) if i >= 0), default=-1)
        if cut >= 0:
            text = text[:cut].rstrip(' \t（(「『【《,，、')
        return normalize_whitespace(text).strip()

    def split_tag_params(tag_content):
        """
        依 `|` 切開圖片語法的參數，但不切在 `[[…]]`／`{{…}}` 裡面。

        直接 str.split('|') 會把圖說裡的內部連結 `[[聖保羅 (巴西)|聖保羅]]`
        從中間剖開，留下 `聖保羅]]的交易所` 這種殘骸（實測 115 筆）。
        """
        parts, buf, depth = [], [], 0
        i, n = 0, len(tag_content)
        while i < n:
            two = tag_content[i:i + 2]
            if two in ('[[', '{{'):
                depth += 1
                buf.append(two)
                i += 2
            elif two in (']]', '}}'):
                depth = max(0, depth - 1)
                buf.append(two)
                i += 2
            elif tag_content[i] == '|' and depth == 0:
                parts.append(''.join(buf))
                buf = []
                i += 1
            else:
                buf.append(tag_content[i])
                i += 1
        parts.append(''.join(buf))
        return parts

    def parse_title(raw_file_tag, file_name, page_title=''):
        # 去掉外層的 `[[` 與 `]]`；前綴（File:／檔案: 等）長度不固定，
        # 用第一個 `:` 定位，不能寫死切幾個字元
        inner = raw_file_tag[2:-2]
        tag_content = inner.split(':', 1)[1] if ':' in inner.split('|', 1)[0] else inner
        parts = split_tag_params(tag_content)
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
                    return clean_wiki_links(parts[desc_idx] + '|' + '|'.join(parts[j:]), page_title)
            # 只取第一个说明内容
            return clean_wiki_links(parts[desc_idx], page_title)
        # 没有說明內容，優先 alt=，否則視為無描述（回傳 None）
        if alt_text:
            return clean_wiki_links(alt_text, page_title)
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
                url = f"https://zh.wikipedia.org/wiki/Special:FilePath/{file_name.replace(' ', '_')}"
                title_raw = parse_title(raw_file_tag, file_name, current_page)
                # 如果沒有說明（title_raw 為 None），則跳過，不使用檔名當描述
                if not title_raw:
                    continue
                # 標記為已處理
                seen_files.add(file_name)
                title_conv = convert_script(title_raw, lang) if title_raw else title_raw
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
