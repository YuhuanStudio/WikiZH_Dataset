"""
從 dump 擷取圖片與圖說

跟正文共用同一套 wikitext 處理（模板展開、語言變體、殘留標記清理、繁簡轉換），
不再另外維護一份較弱的邏輯——舊版自己寫的清理漏掉 {{PAGENAMEBASE}}、{{Tsl}}
等模板，10.8% 的圖說帶著未展開的模板，還會把所有單引號一併刪掉（`'85`、
英文所有格都會壞掉）。

抓取範圍也跟正文對齊。舊版逐行掃描、只認 `[[File:` 一種前綴、完全不看
<gallery>，實測 12 萬頁的漏失：

    [[檔案:／[[Image:／[[圖片: 等前綴    5.6%
    <gallery> 區塊裡的圖片            ~18%
    跨行寫的 [[File:…]]                0.3%

合計漏掉約五分之一。現在整頁讀入，用括號配對找出所有圖片語法。

**一次使用一列**：同一張圖出現在不同條目、配著不同圖說，正是圖文配對最有
價值的部分，因此不做全域去重（舊版只留最早出現的那一個圖說）。

沒有圖說也沒有 alt 的純裝飾圖片會略過——這個資料集的用途是圖文配對，
沒有文字的列沒有意義，也不拿檔名充當描述。
"""

import json
import os
import re

import bz2
from gensim.corpora.wikicorpus import extract_pages
from tqdm import tqdm

from wiki_parser import (_GALLERY_FILE_RE, expand_inline_templates,
                         resolve_variant_markers)
from wiki_text import (convert_script, normalize_whitespace, resolve_variants,
                       strip_leftover_markup)

# 每個輸出分片的上限（沿用舊版，Hugging Face 端的檔名規則不變）
MAX_SHARD_BYTES = 500 * 1024 * 1024

FILE_PREFIX = r'(?:File|Image|Media|檔案|档案|文件|圖片|图片|圖像|图像|媒體|媒体)'
_FILE_OPEN_RE = re.compile(r'\[\[\s*' + FILE_PREFIX + r'\s*:', re.I)
_FILE_NAME_PREFIX_RE = re.compile(r'^\s*' + FILE_PREFIX + r'\s*:\s*', re.I)
# 收尾標籤是必要的，而且跨度要設上限。原本寫成 `(?:</gallery>|\Z)`，
# 未閉合的 gallery 會一路吃到頁尾——後面的章節連同正文整段消失。
# 單一圖片語法的跨度上限（遠大於任何真實圖片語法，只當煞車）
_MAX_IMAGE_SPAN = 20000
_GALLERY_RE = re.compile(r'(?is)<gallery\b[^>]*>(.{0,20000}?)</gallery>')

# 非條目命名空間（模板、分類、專題頁…）不進資料集
_NAMESPACE_RE = re.compile(
    r'^(?:Template|Category|Portal|Help|Draft|MediaWiki|Wikipedia|WikiProject'
    r'|File|Image|Topic|Special|Talk|Module|模块|模組|分類|分类|模板|幫助|帮助'
    r'|維基百科|维基百科)\s*:', re.I)

_REDIRECT_RE = re.compile(r'\s*#\s*(?:REDIRECT|重定向|重新導向)', re.I)

# 圖片語法的顯示參數，不是圖說
_FORMAT_PARAM_RE = re.compile(
    r'(?i)^(?:thumb|thumbnail|frame|frameless|border|right|left|center|centre|none'
    r'|top|bottom|middle|baseline|sub|super|text-top|text-bottom'
    r'|upright(?:\s*=.*)?|\d+\s*x?\s*\d*\s*px|x\d+px'
    r'|(?:link|class|lang|page|thumbtime|start|end|連結|链接|連接|链结|替代)\s*=.*'
    r'|縮圖|缩略图|縮略圖|無框|无框|有框|邊框|边框|右|左|居中|置中|上|下|中)$')

# 章節標題（用來標出圖片出現在條目的哪一節）
_SECTION_RE = re.compile(r'(?m)^\s*(={2,6})\s*(.+?)\s*\1\s*$')


def section_positions(text):
    """回傳 [(位置, 章節標題)]，用來回推每張圖屬於哪一節"""
    return [(m.start(), m.group(2).strip()) for m in _SECTION_RE.finditer(text)]


def section_at(positions, pos):
    """某個位置之前最近的章節標題；在第一個標題之前就是前言（空字串）"""
    name = ''
    for start, title in positions:
        if start > pos:
            break
        name = title
    return name


_ALT_RE = re.compile(r'(?i)^alt\s*=\s*(.*)$')
_IMAGE_EXT_RE = re.compile(
    r'\.(?:jpg|jpeg|png|gif|svg|webp|tif|tiff|bmp|xcf|ogv|ogg|webm|djvu|pdf)$', re.I)
# 佔位用的「暫無圖片」檔名
_PLACEHOLDER_RE = re.compile(r'(?i)no[ _-]?(?:free[ _-]?)?image')
_URL_UNSAFE_RE = re.compile(r'[%#?&+"\'<>\[\]{}|\\^`\s]')


def split_params(body):
    """
    依 `|` 切開圖片語法的參數，但不切在 `[[…]]`／`{{…}}` 裡面。

    直接 str.split('|') 會把圖說裡的內部連結 `[[聖保羅 (巴西)|聖保羅]]`
    從中間剖開，留下 `聖保羅]]的交易所` 這種殘骸。
    """
    parts, buf, depth = [], [], 0
    i, n = 0, len(body)
    while i < n:
        two = body[i:i + 2]
        if two in ('[[', '{{'):
            depth += 1
            buf.append(two)
            i += 2
        elif two in (']]', '}}'):
            depth = max(0, depth - 1)
            buf.append(two)
            i += 2
        elif body[i] == '|' and depth == 0:
            parts.append(''.join(buf))
            buf = []
            i += 1
        else:
            buf.append(body[i])
            i += 1
    parts.append(''.join(buf))
    return parts


def find_image_tags(text):
    """用括號配對找出整頁的 [[File:…]]，回傳 (在原文的位置, 內容)

    位置是用來回推「這張圖出現在哪一節」的——圖文配對要能對應到條目的哪個
    段落，只知道屬於哪一篇是不夠的。
    """
    bodies = []
    length = len(text)
    for m in _FILE_OPEN_RE.finditer(text):
        start = m.start()
        # 跨度設上限，並用 str.find 在標記之間跳躍而不是逐字元前進。
        # 原本每個沒收尾的 `[[File:` 都會一路掃到頁尾，整體是 O(n²)——
        # 實測 500／1000／2000 個未閉合圖片各要 0.4／1.5／6.1 秒。
        limit = min(length, start + _MAX_IMAGE_SPAN)
        depth, i, end = 0, start, -1
        while i < limit:
            nxt_open = text.find('[[', i, limit)
            nxt_close = text.find(']]', i, limit)
            if nxt_close == -1:
                break
            if nxt_open != -1 and nxt_open < nxt_close:
                depth += 1
                i = nxt_open + 2
            else:
                depth -= 1
                i = nxt_close + 2
                if depth == 0:
                    end = i
                    break
        if end != -1:           # 配不到 `]]` 的是寫壞的原始碼，跳過
            bodies.append((start, text[start + 2:end - 2]))
    return bodies


def iter_gallery_bodies(text):
    """<gallery> 區塊裡每一行就是一張圖：`File:名稱.jpg|圖說`（附位置）"""
    if not _GALLERY_RE.search(text):
        return
    for block in _GALLERY_RE.finditer(text):
        offset = block.start(1)
        for line in block.group(1).split('\n'):
            stripped = line.strip()
            # 跟 convert_galleries 用同一個「這一列是一張圖」的判準。兩邊不一致
            # 的話，圖片資料集與 omni 的佔位符會對不上號。
            if stripped and _GALLERY_FILE_RE.match(stripped):
                yield offset, stripped
            offset += len(line) + 1


def clean_caption(raw, page_title, lang):
    """圖說清理：跟正文走同一套規則"""
    if not raw:
        return ''
    text = raw
    for magic in ('{{PAGENAMEBASE}}', '{{PAGENAME}}', '{{BASEPAGENAME}}'):
        text = text.replace(magic, page_title)
    # 圖說裡沒收尾的 <ref 之後全是引用，一路截掉（圖說是單一句子，後面
    # 不會再有內容；正文不能這樣處理）
    text = re.sub(r'(?is)<ref(?![a-z]).*$', '', text)
    text = re.sub(r'(?is)&lt;ref(?![a-z]).*$', '', text)
    text = resolve_variant_markers(text)
    text = expand_inline_templates(text)
    # 模板本體裡也寫著 -{zh-cn:…;zh-tw:…}-，展開後要再挑一次（同 wiki_parser）
    text = resolve_variant_markers(text)
    text = strip_leftover_markup(text)
    # 還留著 `[[`／`{{` 表示原始碼本來就沒收尾，後面已不是可讀的圖說
    cut = min((i for i in (text.find('[['), text.find('{{')) if i >= 0), default=-1)
    if cut >= 0:
        text = text[:cut].rstrip(' \t（(「『【《,，、')
    # 落單的收尾括號（跨語言連結模板展開後留下的 `]]`），自然語言不會有
    text = text.replace(']]', '').replace('}}', '')
    # 變體標記是「兩版都留著」的中間形式，一定要依語言挑一支再轉換。
    # 漏掉這一步的話兩個版本都會拿到 `\ue000賽局理論\ue001博弈论\ue002`，
    # 私有區字元被清掉後變成「賽局理論博弈论」——兩個譯名黏在一起，
    # 而且繁簡兩份圖片資料完全相同（實測各 653 筆）。
    text = resolve_variants(text, lang)
    text = convert_script(text, lang)
    return normalize_whitespace(text).strip()


# 圖說如果本身就是個檔名，那不是描述
_FILENAME_ONLY_RE = re.compile(
    r'(?i)^[\w.()\-–—,&+! ]{1,80}\.(?:jpg|jpeg|png|gif|svg|webp|tif|tiff|pdf)$')


def parse_usage(body, page_title, lang):
    """
    把一段圖片語法內容拆成 (檔名, 圖說, alt)，不合格時回傳 None。

    圖說取「最後一個不是顯示參數的參數」——維基的慣例是圖說寫在最後，
    前面才是 thumb／right／220px 這些排版參數。
    """
    parts = split_params(body)
    if not parts:
        return None
    file_name = _FILE_NAME_PREFIX_RE.sub('', parts[0]).replace('_', ' ').strip()
    if not file_name or not _IMAGE_EXT_RE.search(file_name):
        return None
    if _PLACEHOLDER_RE.search(file_name):
        return None

    alt_raw, caption_raw = '', ''
    for part in parts[1:]:
        stripped = part.strip()
        if not stripped:
            continue
        alt_match = _ALT_RE.match(stripped)
        if alt_match:
            alt_raw = alt_match.group(1).strip()
            continue
        if _FORMAT_PARAM_RE.match(stripped):
            continue
        caption_raw = stripped          # 後面的覆蓋前面的：圖說寫在最後

    caption = clean_caption(caption_raw, page_title, lang)
    alt = clean_caption(alt_raw, page_title, lang)
    # 沒有任何文字（只剩符號、殘留引號）或整段就是個檔名，都不是圖說
    for _ in range(1):
        if caption and (not re.search(r'[\w一-鿿]', caption)
                        or _FILENAME_ONLY_RE.match(caption)):
            caption = ''
        if alt and (not re.search(r'[\w一-鿿]', alt)
                    or _FILENAME_ONLY_RE.match(alt)):
            alt = ''
    return file_name, caption, alt


def _file_url(file_name):
    return ('https://zh.wikipedia.org/wiki/Special:FilePath/'
            + file_name.replace(' ', '_'))


def _page_url(raw_title):
    """條目網址用未經繁簡轉換的原始標題，確保直接命中而非 301 轉址"""
    slug = (raw_title or '').replace(' ', '_')
    return 'https://zh.wikipedia.org/wiki/' + _URL_UNSAFE_RE.sub(
        lambda m: f'%{ord(m.group(0)):02X}', slug)


def extract_wiki_images(xml_path, output_json, max_images=None, lang='tw'):
    """
    走訪整份 dump，輸出圖片與圖說的 JSONL。

    Args:
        xml_path: dump 路徑（.xml 或 .xml.bz2）
        output_json: 輸出路徑；分片時加上 `_N` 序號
        max_images: 上限（測試用）
        lang: 'tw' 或 'cn'

    Returns:
        int: 寫出的記錄數
    """
    opener = bz2.open if xml_path.endswith('.bz2') else open
    base, ext = os.path.splitext(output_json)
    if os.path.dirname(output_json):
        os.makedirs(os.path.dirname(output_json), exist_ok=True)

    total = file_idx = 0
    file_idx = 1
    bytes_written = 0
    failed = 0
    out = open(f'{base}_{file_idx}{ext}', 'w', encoding='utf-8')
    pbar = tqdm(total=max_images, desc='擷取圖片數')

    try:
        with opener(xml_path, 'rb') as f:
            for raw_title, text, page_id in extract_pages(f):
                if not text or _NAMESPACE_RE.match(raw_title or ''):
                    continue
                if _REDIRECT_RE.match(text):
                    continue

                page_title = convert_script(
                    resolve_variant_markers(raw_title or ''), lang)
                bodies = find_image_tags(text)
                bodies.extend(iter_gallery_bodies(text))
                sections = section_positions(text)

                for pos, body in bodies:
                    # 單一圖片解析失敗不該中斷整批
                    try:
                        parsed = parse_usage(body, page_title, lang)
                    except Exception:
                        failed += 1
                        continue
                    if not parsed:
                        continue
                    file_name, caption, alt = parsed
                    # 沒有任何文字的純裝飾圖片，對圖文配對沒有用處
                    if not caption and not alt:
                        continue

                    line = json.dumps({
                        'url': _file_url(file_name),
                        'file_name': file_name,
                        'caption': caption,
                        'alt': alt,
                        'page': page_title,
                        'page_id': str(page_id),
                        'page_url': _page_url(raw_title),
                        # 圖片出現在條目的哪一節。圖文互相學習需要知道圖對應
                        # 的是哪一段，只知道屬於哪一篇不夠——章節名跟正文
                        # 資料集的 `##` 標題用同一套清理，可以直接對起來。
                        # 前言位置的圖片為空字串。
                        'section': clean_caption(
                            section_at(sections, pos), page_title, lang),
                    }, ensure_ascii=False) + '\n'
                    out.write(line)
                    bytes_written += len(line.encode('utf-8'))
                    total += 1
                    pbar.update(1)

                    if bytes_written >= MAX_SHARD_BYTES:
                        out.close()
                        file_idx += 1
                        out = open(f'{base}_{file_idx}{ext}', 'w', encoding='utf-8')
                        bytes_written = 0

                    if max_images is not None and total >= max_images:
                        raise StopIteration
    except StopIteration:
        pass
    finally:
        out.close()
        pbar.close()

    if failed:
        print(f'⚠ {failed} 個圖片語法解析失敗（已跳過）')
    print(f'✓ 完成：{total:,} 筆圖片記錄，{file_idx} 個分片')
    return total


def extract_wiki_images_variants(xml_path, output_jsons, max_images=None):
    """單次走訪 dump，同時輸出 tw / cn 圖片 JSONL。

    圖片語法定位、章節定位與 XML/bz2 解壓都只做一次；只有真正與語言有關的
    標題、圖說、alt 與章節清理各做一遍。每個語言各自維持分片大小、筆數與
    ``max_images``，所以輸出順序和呼叫 :func:`extract_wiki_images` 兩次相同。

    Args:
        output_jsons: ``{'tw': '/path/tw.jsonl', 'cn': '/path/cn.jsonl'}``

    Returns:
        dict: ``{'tw': 筆數, 'cn': 筆數}``
    """
    if set(output_jsons) != {'tw', 'cn'}:
        raise ValueError("output_jsons 必須同時提供 'tw' 與 'cn'")

    opener = bz2.open if xml_path.endswith('.bz2') else open
    states = {}
    for lang in ('tw', 'cn'):
        output_json = output_jsons[lang]
        base, ext = os.path.splitext(output_json)
        if os.path.dirname(output_json):
            os.makedirs(os.path.dirname(output_json), exist_ok=True)
        states[lang] = {
            'base': base, 'ext': ext, 'total': 0, 'file_idx': 1,
            'bytes': 0, 'failed': 0,
            'out': open(f'{base}_1{ext}', 'w', encoding='utf-8'),
        }

    pbar = tqdm(total=(max_images * 2 if max_images is not None else None),
                desc='擷取 tw/cn 圖片數')
    try:
        with opener(xml_path, 'rb') as f:
            for raw_title, text, page_id in extract_pages(f):
                if not text or _NAMESPACE_RE.match(raw_title or ''):
                    continue
                if _REDIRECT_RE.match(text):
                    continue

                bodies = find_image_tags(text)
                bodies.extend(iter_gallery_bodies(text))
                sections = section_positions(text)
                page_titles = {
                    lang: convert_script(
                        resolve_variant_markers(raw_title or ''), lang)
                    for lang in ('tw', 'cn')
                }
                page_url = _page_url(raw_title)

                for pos, body in bodies:
                    raw_section = section_at(sections, pos)
                    for lang in ('tw', 'cn'):
                        state = states[lang]
                        if (max_images is not None
                                and state['total'] >= max_images):
                            continue
                        page_title = page_titles[lang]
                        try:
                            parsed = parse_usage(body, page_title, lang)
                        except Exception:
                            state['failed'] += 1
                            continue
                        if not parsed:
                            continue
                        file_name, caption, alt = parsed
                        if not caption and not alt:
                            continue

                        line = json.dumps({
                            'url': _file_url(file_name),
                            'file_name': file_name,
                            'caption': caption,
                            'alt': alt,
                            'page': page_title,
                            'page_id': str(page_id),
                            'page_url': page_url,
                            'section': clean_caption(
                                raw_section, page_title, lang),
                        }, ensure_ascii=False) + '\n'
                        state['out'].write(line)
                        state['bytes'] += len(line.encode('utf-8'))
                        state['total'] += 1
                        pbar.update(1)

                        if state['bytes'] >= MAX_SHARD_BYTES:
                            state['out'].close()
                            state['file_idx'] += 1
                            state['out'] = open(
                                f"{state['base']}_{state['file_idx']}"
                                f"{state['ext']}", 'w', encoding='utf-8')
                            state['bytes'] = 0

                if (max_images is not None
                        and all(s['total'] >= max_images
                                for s in states.values())):
                    break
    finally:
        for state in states.values():
            state['out'].close()
        pbar.close()

    totals = {}
    for lang, state in states.items():
        if state['failed']:
            print(f"⚠ {lang}: {state['failed']} 個圖片語法解析失敗（已跳過）")
        print(f"✓ {lang}: {state['total']:,} 筆圖片記錄，"
              f"{state['file_idx']} 個分片")
        totals[lang] = state['total']
    return totals


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='從 dump 擷取圖片與圖說')
    ap.add_argument('xml_path')
    ap.add_argument('output_json')
    ap.add_argument('--lang', choices=['tw', 'cn'], default='tw')
    ap.add_argument('--max-images', type=int)
    args = ap.parse_args()
    extract_wiki_images(args.xml_path, args.output_json,
                        max_images=args.max_images, lang=args.lang)
