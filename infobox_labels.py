"""
從 dump 的 Template 命名空間建立「資訊框參數 → 中文標籤」對照表

為什麼需要：資訊框抽出來的事實，標籤原本靠一份手寫的 `_INFOBOX_LABELS`。
量過 25,000 篇條目、200,517 條事實行之後，**75.4% 的標籤仍是英文原鍵**：

    * carlicense：冀X
    * subdivision_type1：省
    * subdivision_name1：广东省
    * leader_name1：彼得·穆塔里卡

相異的英文鍵有 8,604 個，手寫補不完，而且補到哪裡算完全沒有依據。

正確答案本來就在 dump 裡：MediaWiki 渲染資訊框時，標籤是模板頁自己寫的。
`Template:Infobox person/core` 就寫著

    | label9 = 出生
    | data9  = {{{birth_date|}}}

於是 `birth_date` 的標籤是「出生」——這不是猜測，是維基自己的渲染規則。

三種對照關係都要收：

**靜態標籤**   `| label28 = 职业` ＋ `| data28 = {{{occupation|}}}` → occupation：职业
**動態標籤**   `| label33 = {{{subdivision_type1}}}` ＋ `| data33 = {{{subdivision_name1}}}`
               → subdivision_name1 的標籤寫在條目自己的 subdivision_type1 裡，
               所以「广东省」該渲染成「省：广东省」，不是「subdivision_name1：广东省」
**參數別名**   `{{{name|{{{名字|{{{姓名|}}}}}}}}}` → name／名字／姓名 是同一個欄位，
               任一個有標籤就整組都有；中文別名本身也能當標籤用

模板之間會轉包（`Infobox person` 把參數原樣傳給 `Infobox person/core`），
所以收完之後要沿著轉包關係把子模板的對照表併回母模板。
"""

import json
import os
import re
from collections import Counter, defaultdict

import bz2
from gensim.corpora.wikicorpus import extract_pages
from tqdm import tqdm

STORE_NAME = 'infobox_labels.json'

_TEMPLATE_NS_RE = re.compile(r'^(?:Template|模板)\s*:\s*(.+)$', re.I)
# 只看資訊框類模板。導覽框、警告框沒有「欄位標籤」這回事。
_INFOBOX_NAME_RE = re.compile(
    r'(?i)(?:^|[\s_/-])(?:infobox|info box|taxobox|speciesbox|chembox|drugbox'
    r'|信息框|資訊框|资讯框|信息盒)')
_DOC_SUBPAGE_RE = re.compile(r'(?i)/(?:doc|sandbox|testcases|說明文件|说明文件)$')

# 模板本體裡「行首的 `|鍵 =`」。切片邊界要認**所有**參數，不能只認 label/data
# ——`| rowclass31 = …`、`| data31a = …` 夾在中間時，只認 label/data 會把它們
# 連同下一列的參數一起併進上一個 data，標籤就對錯欄位
# （`admin_center`、`largest_settlement` 都被標成「語源」）。
_ANY_SLOT_RE = re.compile(r'(?m)^[ \t]*\|[ \t]*([A-Za-z_][\w -]{0,40}?)[ \t]*=')
_SLOT_KIND_RE = re.compile(r'^(label|data|header)(\d+[a-z]?)$')
# `Infobox settlement` 家族的另一種寫法：`blank_name_sec2` 配 `blank_info_sec2`、
# `blank6_name_sec2` 配 `blank6_info_sec2`。中國行政區模板整批用這個慣例，
# 只認 label/data 的話 `GDP`、`web_title`、`phone code` 這些欄位永遠掛著英文原鍵。
_NAME_INFO_RE = re.compile(r'^(.*?)_(name|info)((?:_sec\d+)?)$')
# 標籤本身可能寫著地區用詞：`[[国内生产总值|-{zh-cn:国内生产总值;zh-tw:國內生產毛額;}-]]`
_VARIANT_LABEL_RE = re.compile(r'-\{([^{}]*)\}-')
# 值裡引用的參數名
_PARAM_REF_RE = re.compile(r'\{\{\{[ \t]*([^|{}\n=]{1,40}?)[ \t]*[|}]')
_LINK_RE = re.compile(r'\[\[(?:[^\[\]|]*\|)?([^\[\]|]+)\]\]')
_TAG_RE = re.compile(r'(?s)<[^>]{0,200}>')
_TEMPLATE_CALL_RE = re.compile(r'(?s)\{\{[^{}]*\}\}')
_CJK_RE = re.compile(r'[一-鿿]')
# 轉包對象：`{{Infobox person/core` / `{{infobox3cols`
_DELEGATE_RE = re.compile(r'(?i)\{\{\s*(infobox[^|}\n]{0,60}|模板:[^|}\n]{0,60})\s*[|}]')

# 標籤合理長度。超過就不是欄位名（是整段說明文字或沒清乾淨的標記）。
MAX_LABEL_LEN = 12


def _norm(name):
    return name.replace('_', ' ').strip().lower()


def _slots(body):
    """把模板本體切成 {('label'|'data'|'header', 編號): 值}

    認兩種配對慣例：`labelN`／`dataN`，以及 `X_name`／`X_info`（後者是
    `Infobox settlement` 家族的寫法，中國行政區模板整批在用）。
    """
    out = {}
    marks = list(_ANY_SLOT_RE.finditer(body))
    for i, m in enumerate(marks):
        key = m.group(1).strip().lower()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        value = body[m.end():end].strip()
        kind = _SLOT_KIND_RE.match(key)
        if kind:
            out[(kind.group(1), kind.group(2))] = value
            continue
        ni = _NAME_INFO_RE.match(key)
        if ni:
            slot = 'label' if ni.group(2) == 'name' else 'data'
            out[(slot, ni.group(1) + ni.group(3))] = value
    return out


def _clean_label(text):
    """把 label 欄的內容整理成一個可讀的中文欄位名（不合格回傳空字串）"""
    # 標籤寫成地區用詞時取簡體那一支——對照表是語言中立的中間層，
    # 事實行之後會跟正文一起做繁簡轉換。
    text = _VARIANT_LABEL_RE.sub(_pick_variant, text)
    # 欄位名後面常掛一段條件式註記（統計年份、單位、換行的補充說明）：
    # `[[国内生产总值|國內生產毛額]]{{#if:{{{GDP_as_of|}}}|<span…>（2015）</span>}}`。
    # 那是註記不是欄位名，整段拿去清理只會超長而被丟掉——欄位名是前面那一截。
    cut = min((i for i in (text.find('{{#'), text.find('<span'), text.find('<br'))
               if i > 0), default=-1)
    if cut > 0:
        text = text[:cut]
    text = _LINK_RE.sub(r'\1', text)
    # `{{ifempty|{{{grid_name|}}}|網格位置}}` 這種：最內層先收，收到剩純文字
    for _ in range(3):
        new = _TEMPLATE_CALL_RE.sub(
            lambda m: max(re.split(r'[|}{]', m.group(0)), key=len), text)
        if new == text:
            break
        text = new
    text = _TAG_RE.sub('', text)
    text = text.replace('&nbsp;', ' ').replace('•', ' ').replace("'", '')
    text = re.sub(r'\{+|\}+|\|', ' ', text)
    text = ' '.join(text.split()).strip(' :：*#-–—')
    if not text or len(text) > MAX_LABEL_LEN:
        return ''
    # 標籤要是中文。純英文標籤留著沒有意義——那正是我們要取代的東西。
    if not _CJK_RE.search(text):
        return ''
    return text


def _pick_variant(m):
    """`zh-cn:国内生产总值;zh-tw:國內生產毛額;` → 取 zh-cn 那一支"""
    body = m.group(1)
    for tag in ('zh-cn:', 'zh-hans:', 'zh-sg:', 'zh-my:'):
        i = body.find(tag)
        if i >= 0:
            return body[i + len(tag):].split(';')[0].strip()
    return body.split(';')[0].split(':')[-1].strip()


def _alias_groups(body):
    """`{{{name|{{{名字|{{{姓名|}}}}}}}}}` → 同一欄位的別名組

    只認**真正巢狀**的參數鏈。原本用「從 `{{{` 起算 400 字」的視窗抓，
    同一行裡並排的參數會被併成一組：`{{{birth_date|}}}<br>{{{birth_place|}}}`
    是兩個欄位，`{{#if:{{{a|}}}|{{{b|}}}}}` 是條件式的兩個分支。誤併之後標籤
    會沿著假的別名關係擴散，實測污染出 `carlicense → 山峰`、`taxon → 母公司`
    這種完全不相干的對應。
    """
    groups = []
    i, n = 0, len(body)
    while True:
        i = body.find('{{{', i)
        if i < 0:
            return groups
        depth, j, names = 0, i, []
        while j < n:
            if body.startswith('{{{', j):
                depth += 1
                m = _PARAM_REF_RE.match(body, j)
                if m:
                    names.append(_norm(m.group(1)))
                j += 3
            elif body.startswith('}}}', j):
                depth -= 1
                j += 3
                if depth == 0:
                    break
            else:
                j += 1
        names = [x for x in names if x and len(x) <= 40]
        if len(names) > 1:
            groups.append(names)
        i = max(j, i + 3)


def _scan_template(body):
    """單一模板 → (靜態標籤, 動態標籤, 別名組)"""
    static, dynamic = {}, {}
    slots = _slots(body)
    # 只引用一個參數的列先配。一個 data 欄可以引用好幾個參數——人口密度是
    # `{{#expr:{{{population_total}}}/{{{area_km2}}}}}`，照單全收的話
    # `population_total` 會被標成「密度」。獨佔一列的那個參數才是這一列的主角。
    rows = [(idx, value) for (kind, idx), value in slots.items() if kind == 'data']
    rows.sort(key=lambda r: len(set(_PARAM_REF_RE.findall(r[1]))))
    for idx, value in rows:
        label_raw = slots.get(('label', idx))
        if label_raw is None:
            continue
        params = [_norm(p) for p in _PARAM_REF_RE.findall(value)]
        params = [p for p in params if p]
        if not params:
            continue
        label = _clean_label(label_raw)
        if label:
            for p in params:
                static.setdefault(p, label)
            continue
        # 標籤本身就是另一個參數：欄位名寫在條目裡（`subdivision_type1=省`）
        label_params = [_norm(p) for p in _PARAM_REF_RE.findall(label_raw)]
        if len(label_params) == 1 and label_params[0]:
            for p in params:
                if p != label_params[0]:   # 自己當自己的標籤沒有意義
                    dynamic.setdefault(p, label_params[0])
    return static, dynamic, _alias_groups(body)


def _new_collector():
    """建立可供外部 dispatcher 共用的掃描狀態。"""
    return {
        'per_box_static': {}, 'per_box_dynamic': {}, 'delegates': {},
        'global_static': defaultdict(Counter),
        'global_dynamic': defaultdict(Counter), 'alias_pairs': [],
    }


def _collect_page(state, title, text):
    """收集單一 dump 頁；不做 I/O 與掃描後傳播。"""
    m = _TEMPLATE_NS_RE.match(title or '')
    if not m or not text:
        return
    name = _norm(m.group(1))
    if _DOC_SUBPAGE_RE.search(name) or not _INFOBOX_NAME_RE.search(name):
        return
    static, dynamic, groups = _scan_template(text)
    if static:
        state['per_box_static'][name] = static
        for p, label in static.items():
            state['global_static'][p][label] += 1
    if dynamic:
        state['per_box_dynamic'][name] = dynamic
        for p, tp in dynamic.items():
            state['global_dynamic'][p][tp] += 1
    state['alias_pairs'].extend(groups)
    targets = {_norm(re.sub(r'(?i)^模板:', '', t))
               for t in _DELEGATE_RE.findall(text)}
    targets.discard(name)
    if targets:
        state['delegates'][name] = sorted(targets)


def _finish_collector(state, out_dir):
    """完成轉包與別名傳播，寫出已掃完的狀態。"""
    per_box_static = state['per_box_static']
    per_box_dynamic = state['per_box_dynamic']
    delegates = state['delegates']
    global_static = state['global_static']
    global_dynamic = state['global_dynamic']
    alias_pairs = state['alias_pairs']

    # 轉包：`Infobox person` 的標籤實際寫在 `Infobox person/core`。
    # 兩輪就夠——實際的轉包鏈最多兩層。
    for _ in range(2):
        for name, targets in delegates.items():
            merged = per_box_static.setdefault(name, {})
            merged_dyn = per_box_dynamic.setdefault(name, {})
            for target in targets:
                for p, label in per_box_static.get(target, {}).items():
                    merged.setdefault(p, label)
                for p, tp in per_box_dynamic.get(target, {}).items():
                    merged_dyn.setdefault(p, tp)

    static_map = {p: c.most_common(1)[0][0] for p, c in global_static.items()}
    dynamic_map = {p: c.most_common(1)[0][0] for p, c in global_dynamic.items()}

    # 別名傳播：同一組別名共用一個標籤。中文別名本身就是最好的標籤來源
    # （`{{{caption|{{{圖片簡介|…}}}}}}` → caption 的標籤是「圖片簡介」）。
    #
    # 推導出來的標籤另存一張表，優先權排在「模板頁自己寫的 label」之後。
    # 別名鏈是間接證據，直接寫在 label 欄的才是維基真正渲染出來的字。
    # 只傳播「模板頁真的寫在 label 欄」的標籤。曾經也拿中文參數名當標籤，
    # 那條規則會憑空造出錯的欄位名：taxobox 用界別決定配色，寫成
    # `{{{顏色|{{{regnum|}}}}}}`，於是「植物界」被標成「顏色：植物界」；
    # `{{{母公司|{{{taxon|}}}}}}` 讓學名變成「母公司：Ginkgo biloba」。
    # 錯的標籤比英文原鍵糟得多——它斷言了一件假事實。
    alias_map = {}
    for _ in range(2):
        for group in alias_pairs:
            label = next((static_map[p] for p in group if p in static_map), '')
            if not label:
                label = next((alias_map[p] for p in group if p in alias_map), '')
            if not label:
                continue
            for p in group:
                if p not in static_map:
                    alias_map.setdefault(p, label)
    added = len(alias_map)

    # 兩張表都留著。哪一張優先在**使用時**決定：條目真的填了那個型別參數
    # （`subdivision_type1=省`）就用動態標籤，沒填才退回靜態標籤。
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, STORE_NAME)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'static': static_map, 'dynamic': dynamic_map,
                   'alias': alias_map,
                   'by_box': per_box_static, 'by_box_dynamic': per_box_dynamic},
                  f, ensure_ascii=False)
    print(f'資訊框標籤：靜態 {len(static_map):,}、別名推導 {added:,}、'
          f'動態 {len(dynamic_map):,}、模板 {len(per_box_static):,} 個 → {path}')
    return static_map, dynamic_map, alias_map, per_box_static, per_box_dynamic


def build(xml_path, out_dir):
    """掃一趟 dump，把資訊框標籤對照表寫成 JSON"""
    opener = bz2.open if xml_path.endswith('.bz2') else open
    state = _new_collector()
    with opener(xml_path, 'rb') as f:
        for title, text, _pid in tqdm(extract_pages(f), desc='收集資訊框標籤'):
            _collect_page(state, title, text)
    return _finish_collector(state, out_dir)


RENDERED_NAME = 'infobox_labels_rendered.json'


def load_rendered(out_dir):
    """讀取「從渲染結果反推」的標籤表（見 qa/label_from_render.py）

    Lua 模組算版面的模板（中國行政區那一批）在模板頁裡沒有 `label = …`，
    這裡挖不到。那份表是拿渲染後的 HTML 反推出來的，優先權排在模板頁自己
    寫的標籤之後、全域最常見標籤之前——它有兩篇以上條目佐證，比跨模板
    猜一個可靠，但畢竟是反推，不如模板頁的白紙黑字。
    """
    path = os.path.join(out_dir, RENDERED_NAME)
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load(out_dir):
    """回傳 (靜態, 動態, 別名推導, 各模板靜態, 各模板動態)；沒有檔案就回傳空表"""
    path = os.path.join(out_dir, STORE_NAME)
    if not os.path.exists(path):
        return {}, {}, {}, {}, {}
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return (data.get('static', {}), data.get('dynamic', {}), data.get('alias', {}),
            data.get('by_box', {}), data.get('by_box_dynamic', {}))
