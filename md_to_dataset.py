"""
維基百科中間層 → 文檔級資料集

一個條目一筆記錄，保留 `##` 章節結構，輸出 Parquet。

設計原則見 wiki_text.py：只移除確定是標記的東西，絕不刪除自然語言內容。
模板被清空後留下的殘句（「的郵政編碼為，INSEE市鎮編碼為。」）採子句層級
修剪，只切掉壞掉的那一小塊，不整句丟棄——整句丟棄會連坐真實內容。
"""

import os
import re
import sys
from functools import partial

import multiprocessing

from page_store import iter_pages, shard_paths
from wiki_parser import IMAGE_MARK, strip_image_marks, unmask_verbatim_braces
from wiki_text import (
    _canon_section,
    normalize_whitespace,
    convert_script,
    drop_empty_brackets,
    finalize_block,
    resolve_variants,
    should_skip_section,
    strip_leftover_markup,
)


# 每個 Parquet 分片的目標原始文本大小（壓縮後約為 1/3）
SHARD_TARGET_BYTES = 600 * 1024 * 1024

# zstd 9 在實測上只比 level 3 小約 13%，壓縮 CPU 卻是 4.2 倍。
# 訓練資料的邏輯內容不變；若發佈場景優先檔案大小，可用
# WIKIZH_ZSTD_LEVEL=9 恢復舊設定。
PARQUET_COMPRESSION_LEVEL = int(os.environ.get('WIKIZH_ZSTD_LEVEL', '3'))

# HF 以 row group 為串流單位；現行每個 600 MiB 分片只有一組，
# 讀前幾筆也必須解整組。以現有每列約 7.2 KiB 計，16,384 列
# 約是 118 MiB uncompressed。這些參數只改變 Parquet 物理佈局。
PARQUET_ROW_GROUP_SIZE = int(os.environ.get('WIKIZH_ROW_GROUP_SIZE', '16384'))
PARQUET_USE_DICTIONARY = os.environ.get(
    'WIKIZH_PARQUET_DICTIONARY', '0').lower() in {'1', 'true', 'yes'}

# 正文（不含標題行）的最低字數。
#
# 預設為 0：**沒有內容才刪，太短不是理由**。中文維基有大量物種、小行星、
# 村里、地理特徵的一句話條目（「PySide是跨平台的圖形使用界面框架Qt的Python
# 版本。」），那是完整的知識內容。維基官方資料集（wikimedia/wikipedia）也不做
# 長度過濾——使用者要篩隨時可以自己做，被我們丟掉的救不回來。
#
# 一度設 30，理由是「擋掉空殼」；但空殼本來就會因為 body 為空而被擋下，
# 30 這個數字只多刪掉了真正的短條目（「李某某是中國演員。」9 字）。
# 需要篩選的人傳 min_length 進來即可。
MIN_DOC_LENGTH = 0

# 消歧義頁的判定不在這裡，在解析階段讀維基自己掛的 `{{disambig}}` 標記模板
# （見 wiki_parser.__is_disambiguation）。這裡只留「標題自己就寫著消歧義」
# 這個同樣明確的標記。
#
# 刪除的理由必須是「沒有內容」，不能是「措辭長得像消歧義頁」。這裡曾經有一條
# 「開頭寫著『可以指：』且全篇沒有成段散文就丟掉」的啟發式，那是拿措辭當依據：
# `大滿貫` 逐條說明各運動項目的大滿貫定義、`町` 講中日兩地的行政區沿革，
# 都有實質內容卻會被它刪掉。維基官方標記已經給了權威答案，猜測沒有存在理由；
# 真正空殼的頁面由 MIN_DOC_LENGTH 擋下即可。
DISAMBIGUATION_TITLE_RE = re.compile(r'[（(]\s*(消歧義|消歧义|disambiguation)\s*[）)]', re.I)

# 非條目命名空間。wiki_parser 的過濾清單漏掉 Module / WikiProject 等，
# 這些頁面是程式碼或專案討論，不該進語料
NAMESPACE_RE = re.compile(
    r'^(Module|WikiProject|Template|Category|Portal|Help|Draft|MediaWiki|Wikipedia'
    r'|File|Image|Topic|Special|Talk|模块|模塊|模組|模组|分類|分类|模板|幫助|帮助|維基百科|维基百科)\s*:',
    re.I,
)

HEADING_RE = re.compile(r'^(#{1,6})\s*(.+?)\s*$')

# 解析階段產生的章節標題一定是「`#`×層級 + 空白 + 編號 + 空白 + 標題」
# （`## 1 定義`、`### 2.1 起源`），編號是我們自己加的，正文絕不會有這個形態。
# 只憑「開頭是 `#`」認標題會把正文誤判成章節：`{{Colors|#fff|…}}` 展開成
# `#fff` 之後，「伊予鐵道」的一整列表格就變成了一個假章節。
_REAL_HEADING_RE = re.compile(r'^#{2,6}[ \t]+\d+(?:\.\d+)*[ \t]')

# 本專案在 wiki_parser 自行加上的章節編號（`## 1 定義`、`### 2.1 起源`）。
# 數字後面一定跟著空白，據此區分「編號」與「標題本身就以數字開頭」
# （例如條目「2010年至2011年英格蘭足球甲級聯賽」的年份不能被當成編號砍掉）。
HEADING_NUMBER_RE = re.compile(r'^\d+(?:\.\d+)*[ \t]+')

WIKI_BASE_URL = 'https://zh.wikipedia.org/wiki/'

# URL 中必須百分比編碼的字元（CJK 保持原樣，維持可讀性）
URL_UNSAFE_RE = re.compile(r'[%#?&+"\'<>\[\]{}|\\^`\s]')


# 解析階段移除 ==標題== 時可能留下落單的 =，只清這個符號
# （不碰引號、破折號，避免傷到「-273度」這類正常標題）
HEADING_JUNK_RE = re.compile(r'^[=\s]+|[=\s]+$')


def _strip_heading_number(title, level):
    """
    去掉章節編號：`1 定義` → `定義`、`2.1 起源` → `起源`。

    層級 1 是條目標題，本來就沒有編號，絕對不能動——否則
    「2018年澳門丙組足球聯賽」會被砍成「年澳門丙組足球聯賽」。
    """
    # 解析階段移除 ==標題== 時會留下落單的 =
    title = HEADING_JUNK_RE.sub('', title.strip())
    title = title.strip()
    if level <= 1:
        return title
    return HEADING_NUMBER_RE.sub('', title).strip()


def _article_url(title):
    """組出條目網址，只對 URL 不安全的字元做百分比編碼"""
    slug = title.replace(' ', '_')
    return WIKI_BASE_URL + URL_UNSAFE_RE.sub(lambda m: f'%{ord(m.group(0)):02X}', slug)


# 跳過清單裡有「其他」「來源」「資源」「說明」「文獻」這種兩字通用詞，
# 用子字串比對會把實質內容誤判成參考章節。實測誤刪案例：
#   [烏西與烏克蘭其他地區的文化差異] 672 字
#   [理論來源] 523 字（社團主義的理論淵源）
#   [天然資源 > 石油及天然氣] 272 字
#   [翻譯說明] 630 字（聖經譯本特色）
# 因此短關鍵字一律只認完全相同的標題，長關鍵字才允許前綴比對。
_EXACT_ONLY_LEN = 2


def iter_sections(content):
    """
    走訪 Markdown，產出 (層級, 標題, 內文, 標題路徑)。

    層級 1 是條目標題本身，內文為前言（lead），沒有標題的開頭段落也算在這裡。
    """
    title_stack = []
    buffer = []
    level = 0
    title = ''

    for index, line in enumerate(content.split('\n')):
        # 第一行是條目標題（`# 條目名`），層級 1 沒有編號，只認位置。
        is_heading = _REAL_HEADING_RE.match(line) or (
            index == 0 and line.startswith('# '))
        match = HEADING_RE.match(line) if is_heading else None
        if match:
            if title_stack or buffer:
                yield level, title, '\n'.join(buffer).strip(), list(title_stack)
            buffer = []
            level = len(match.group(1))
            title = _strip_heading_number(match.group(2), level)
            while len(title_stack) >= level:
                title_stack.pop()
            title_stack.append(title)
        else:
            buffer.append(line)

    if title_stack or buffer:
        yield level, title, '\n'.join(buffer).strip(), list(title_stack)


# 模板被移除後留下的殘缺句子。這類句子語法不完整、事實也不見了
# （例如「的郵政編碼為，INSEE市鎮編碼為。」「於時的人口數量為人。」），
# 大量出現在機器生成的行政區條目，佔約 12% 的條目。
# 會被模板填值的屬性名。殘句偵測一律要求先出現這些詞，
# 才不會誤傷正常句子。
# 關鍵字要涵蓋繁簡兩種寫法。這些規則跑在**轉換後**的文字上，只寫繁體的話
# 簡體資料集完全不會生效——實測簡體殘句比繁體還多。
_ATTR_WORDS = [
    '總面積', '面積', '人口', '人口數量', '人口密度', '居民數量', '居民數',
    '海拔', '高程', '長度', '寬度', '高度', '深度', '距離', '時區',
    '郵政編碼', '郵編', '郵區編號', '電話區號', '區號',
    'INSEE市鎮編碼', '市鎮編碼', '編碼', '坐標', '座標', '經度', '緯度',
    '選區', '人數',
]
_COPULA_WORDS = ['為', '是', '約為', '達']
_UNIT_WORDS = ['人', '個', '座', '隻', '頭', '份']


def _both_scripts(words):
    """把詞表展開成繁簡兩種寫法的正則選項（長的排前面，避免短詞先匹配）"""
    import zhconv
    hant = zhconv._Converter([zhconv._load()['zh2Hant']])
    hans = zhconv._Converter([zhconv._load()['zh2Hans']])
    out = set()
    for w in words:
        out.update({w, hant.convert(w), hans.convert(w)})
    return '(?:' + '|'.join(sorted(out, key=len, reverse=True)) + ')'


_ATTR = _both_scripts(_ATTR_WORDS)
_COPULA = _both_scripts(_COPULA_WORDS)
_UNIT = _both_scripts(_UNIT_WORDS)

# 這份規則刻意寫得很窄：寧可漏掉幾個殘句，也不能誤刪正常內容。
# 早期版本用「句子以助詞開頭」「是/為後接標點」判斷，結果把
# 「以下是在原作漫畫…」「但是，部分角色…」這類正常句子砍掉了 23%，
# 屬於嚴重的品質倒退，因此改為只認模板留下的特定形態。
# 殘句偵測改在「子句」層級動刀，而不是整句丟棄。
#
# 整句丟棄會連坐真實內容，實測誤殺案例：
#   「…其中心月面座標為，直徑22.85公里，深度約1.85公里。」→ 直徑深度一起不見
#   「安德爾河畔沙蒂永…面積為45.3平方公里，時人口數量為人，排名第4,242位。」→ 整個前言消失
# 現在只切掉「屬性＋繫詞後面沒有值」的那一個子句，其餘保留。
#
# 判準只看**子句結尾**，不限制前面有多長。原本寫成 `^.{0,8}?屬性繫詞$`，那個
# 字數預算會讓同一句話在繁簡兩版得到不同結果——地區譯名的長度本來就不一樣：
#     簡體「沙特奈所属的省级選區為」  前綴 8 字 → 命中，丟掉  ✓
#     繁體「沙烏地奈所屬的省級選區為」前綴 9 字 → 不中，留下  ✗
# 於是繁體版多出一個「## 政治」章節，內容是「沙烏地奈所屬的省級選區為。」。
# `_ATTR` 是封閉的資訊框屬性詞表，子句以「屬性＋繫詞」收尾本身就代表值沒填上，
# 前面有多長並不影響這個判斷。
BROKEN_CLAUSE_RES = [
    re.compile(_ATTR + _COPULA + r'$'),                    # 「其中心月面座標為」
    re.compile(_ATTR + _COPULA + r'?' + _UNIT + r'$'),     # 「時人口數量為人」
    re.compile(r'^(?:於|于)?(?:時|时)$'),                   # 時間被模板帶走
    re.compile(r'^[\s，。、；：]*$'),
]
# 前兩條只看結尾，比對時只需要餵最後這麼多字（最長的「INSEE市鎮編碼」＋「約為」
# ＋單位詞也不到 20 字）。整句餵進去的話，沒有標點的長行會讓每個位置都試一遍
# 那六十幾個詞的選擇組。
_CLAUSE_TAIL = 24

# 子句層級清理後，句首若還留著孤立的「的」「於時」，直接切掉這個引導字，
# 但保留後面的內容（「的時區為UTC+01:00。」→「時區為UTC+01:00。」）
LEADING_JUNK_RE = re.compile(r'^(?:[於于][時时]的?|的(?![確士话確话]))')

# 整句都無法挽救時才丟棄
BROKEN_SENTENCE_RES = [
    re.compile(r'^[\s，。、；：]*$'),
    # 冒號後面直接就是句號——引導語有了、內容沒了
    # （「與接壤的市鎮包括：。」的清單來自無法解析的實體連結）
    re.compile(r'[：:]\s*[。！？]\s*$'),
]

# 只剩標記與標點的列表項
# 表格儲存格內被壓平的清單標記
_CELL_LIST_RE = re.compile(r'(?<=[^\s*])[ \t]*\*[ \t]*(?=[^\s*])')

_EMPTY_ITEM_RE = re.compile(r'^[-*•]\s*[^\w一-鿿]*$')

# 沒有值的事實不是事實。資訊框抽出來的 `- 標籤：值` 走一般清理流程，值若是
# 模板（`| 成立 = {{start date|…}}`）會在展開後變成空的，留下「- 成立：」
# 這種有頭沒尾的行（`鄄城縣` 就是這樣）。抽取當下擋不掉，只能在清理後再篩。
#
# 判準是「冒號前面是不是一個標籤」，不是「有多短」。標籤不含句讀，引導語含：
#     - 成立：                              ← 標籤，值沒填上，丟掉
#     - L1A路目前配置23台公車，詳細情況如下：   ← 引導語，後面接著表格，要留
# 原本只用 `{1,20}` 的字數預算，那個預算會讓繁簡兩版判斷不同——`公交车` 在
# 繁體是 `公車`，同一行簡體 21 字（保留）、繁體 20 字（被丟掉）。字數上限留著
# 只當保險，真正把兩者分開的是句讀。
_EMPTY_FACT_RE = re.compile(r'^[-*•]\s*[^：:\n，。！？；、,.!?;]{1,30}[：:]\s*$')

CLAUSE_SPLIT_RE = re.compile(r'(?<=[，、；])')
SENTENCE_SPLIT_RE = re.compile(r'(?<=[。！？])')


def _drop_broken_sentences(text):
    """
    逐句掃描，丟掉模板被清空後留下的殘句。

    Returns:
        (清理後的文本, 被丟掉的句子數)
    """
    dropped = 0
    out_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            out_lines.append(line)
            continue

        # 內容被移除後只剩標點的列表項（`-。`、`- 、`）不是內容
        if _EMPTY_ITEM_RE.match(stripped) or _EMPTY_FACT_RE.match(stripped):
            dropped += 1
            continue

        # 列表項整行判斷，一般段落逐句判斷
        prefix = '- ' if stripped.startswith('- ') else ''
        body = stripped[2:] if prefix else stripped

        kept = []
        for sentence in SENTENCE_SPLIT_RE.split(body):
            s = sentence.strip()
            if not s:
                continue

            # 子句層級：只切掉「屬性＋繫詞後面沒有值」的子句，其餘保留
            clauses = CLAUSE_SPLIT_RE.split(s)
            good = []
            for clause in clauses:
                # 句末標點也要去掉。原本只去 `，、；`，於是「人口數量為人。」
                # 這種**自成一句**的殘句永遠匹配不到規則（規則要求以單位詞結尾），
                # 實測 2.1% 的條目因此留著「INSEE市鎮編碼為。」這類空句。
                core = clause.strip().rstrip('，、；。！？：')
                if core and (any(p.search(core[-_CLAUSE_TAIL:])
                                 for p in BROKEN_CLAUSE_RES[:2])
                             or any(p.match(core) for p in BROKEN_CLAUSE_RES[2:])):
                    dropped += 1
                    continue
                good.append(clause)

            s = ''.join(good).strip()
            if not s:
                continue
            # 切掉尾端子句後會留下逗號（「面積為34.22平方公里，」），補回句號
            s = re.sub(r'[，、；]+$', '。', s)

            # 主語被模板帶走時句首會留下孤立的「的」「於時」，切掉引導字保留內容
            trimmed = LEADING_JUNK_RE.sub('', s)
            if trimmed != s:
                dropped += 1
                s = trimmed.strip()

            # 切完後可能只剩標點
            s = re.sub(r'^[，、；：\s]+', '', s)
            if not s or any(p.search(s) for p in BROKEN_SENTENCE_RES):
                continue
            kept.append(s)

        if kept:
            out_lines.append(prefix + ''.join(kept))

    return '\n'.join(out_lines), dropped


# 表格列裡的圖片顯示參數（thumb、220px…），不是資料。
# 只在含有「｜」的表格列上套用——right、left、none 這些詞在一般英文句子裡
# 很常見，全文套用會誤刪正文。
IMAGE_PARAM_RE = re.compile(
    r'(?:(?<=^)|(?<=｜))\s*(?:thumb|thumbnail|frameless|border|upright'
    r'|left|right|center|centre|none|\d+x?\d*px|縮圖|缩图|縮略圖|缩略图|無框|无框|有框|邊框|边框)'
    r'\s*(?=｜|$)', re.I)


# 表格屬性殘骸的收尾網。解析階段已經會剝掉儲存格屬性，但維基上的寫法太雜
# （`rowspan=4 6|`、`align =center; height: 30px;|`），這裡再兜一次底，
# 讓資料集不受上游是否漏接影響。只在含全形「｜」的表格列上套用。
_CELL_ATTR_JUNK_RE = re.compile(
    r'(?i)[.\s!"\']{0,6}'
    r'(?:style|width|height|align|valign|bgcolor|color|colspan|rowspan|scope|class'
    r'|id|abbr|nowrap|border|cellpadding|cellspacing|dir|sortable|data-sort-value)'
    r'\s*=[^｜\n]{0,120}?\|')


def _tidy_tables(text):
    """表格列收尾：屬性殘骸、顯示參數與空儲存格（字元類不能含 \\s，否則會吃掉換行黏住段落）"""
    out = []
    for line in text.split('\n'):
        # 屬性殘骸不一定落在有全形「｜」的列上（表頭被拆行時就會單獨成行），
        # 規則本身夠明確（已知屬性名 + 值 + 半形管線），可以整篇套用
        # 規則必定同時包含屬性指派的 = 和收尾的半形 |。
        if '=' in line and '|' in line:
            line = _CELL_ATTR_JUNK_RE.sub('', line)
        if '｜' in line:
            # 表格儲存格裡的清單不能拆行（會破壞欄位對齊），把 `*` 換成頓號
            # （`澧水支流 *涔水 *道水` → `澧水支流：涔水、道水`）
            line = _CELL_LIST_RE.sub('、', line)
            line = IMAGE_PARAM_RE.sub('', line)
            line = re.sub(r'｜{2,}', '｜', line)
            line = re.sub(r'^[｜ \t]+|[｜ \t]+$', '', line)
        out.append(line)
    return '\n'.join(out).strip()


# 前言區（第一段正文之前）殘留的模板參數碎片。
#
# 實測案例：「美國」開頭有 `20F24`、「泰勒斯」有 `fr姓名 = 泰勒斯Θαλῆς`、
# 「詩人」有 `en`。條目的前言一定是完整句子，在它之前的短碎片可以安全丟掉。
# 純長度門檻會在邊界上砍掉真內容：「柯萊門斯開局是西洋棋開局非正規開局的一種，
# 走法為：」只有 24 字又沒有句號，繁體版被當成碎片刪掉，簡體版 26 字剛好留下來
# ——同一個條目兩個語言版本內容不同，顯然是規則錯而不是資料錯。
# 改成判斷「這一行像不像文字」，長度只用來輔助。
_LEAD_DEBRIS_MAX = 25
_CJK_RE = re.compile(r'[一-鿿]')
# 句讀。半形句點要求後面接空白或行尾，否則 `20F24.jpg`、`p.108` 會被當成句子
_SENTENCE_PUNCT_RE = re.compile(r'[。！？!?]|\.(?:\s|$)')
# 模板參數行 `ImageFile=Hydrogenglow.jpg`、`Head = 白族民族乡`。
# 參數名要求兩字以上且緊接 `=`，才不會把公式 `E = mc²`、`sin x = …` 當成碎片。
_PARAM_LINE_RE = re.compile(r'[A-Za-z_][\w\-]{1,23}\s*=')


def _is_lead_debris(s):
    """前言第一句之前的這一行是不是模板碎片"""
    if _SENTENCE_PUNCT_RE.search(s):
        return False                      # 有句讀就是句子
    if s.endswith(('：', ':')):
        return False                      # 冒號引導後面的列表，是正文
    if _PARAM_LINE_RE.match(s):
        return True                       # 模板參數殘留：`fr姓名 = 泰勒斯Θαλῆς`
    if not _CJK_RE.search(s):
        # 純外文：又短又只有一兩個詞才是碎片（`en`、`20F24`、`20px`）。
        # 「International Business Machines Corporation」是原文全名，
        # 「E = mc²」是公式，都要留。
        return len(s) <= _LEAD_DEBRIS_MAX and len(s.split()) <= 2
    # 有中文卻既無句讀也不長，才當碎片
    return len(s) <= 12


# 沒收尾的模板會讓參數黏在前言第一句前面：
# 「Name = Denmark丹麦国家室内足球队，是丹麦在室内足球项目上的国家代表队。」
# 這一行有句讀，是正文，不能整行丟掉；只切掉開頭那段 `鍵 = 值`。
# 刻意不寫成 `(?:鍵=值[^中文]*)+` 這種巢狀量詞——會指數級回溯，改用迴圈重複套用。
_LEAD_PARAM_PREFIX_RE = re.compile(r'^[A-Za-z_][\w\-]{1,23}[ \t]*=[ \t]*[^一-鿿\n]{0,200}')


def _strip_lead_param_prefix(s):
    for _ in range(6):
        trimmed = _LEAD_PARAM_PREFIX_RE.sub('', s, count=1).lstrip()
        if trimmed == s or not trimmed:
            break
        s = trimmed
    return s


def _drop_lead_debris(text):
    """
    丟掉前言第一個完整句子之前的模板碎片。

    範圍必須止於第一個章節標題。前言的定義就是「第一個標題之前的段落」，
    而 `_is_lead_debris` 的判準（沒句讀又不到 12 字就算碎片）只在前言那個
    位置成立——正文裡這種短句多得是。

    之前沒有停在標題上，於是「沒有前言、開頭就是章節」的條目（年份條目、
    列表型條目很常見）會拿前言的規則去砍第一節的第一句：
        [1953年香港] ## 政府 / 英国君主：伊丽莎白二世   ← 整句被當碎片刪掉
    留下一個有標題沒內容的空章節。前言整段都是碎片時也一樣會漏進正文。
    """
    lines = text.split('\n')
    out, seen_sentence = [], False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#'):
            # 進入正文，前言處理結束，其餘原封不動
            out.extend(lines[i:])
            break
        if stripped.startswith('```'):
            # 前言位置就是程式碼區塊：整塊逐字保留，前言處理到此為止。
            # 不擋的話「沒句讀又不到 12 字」的判準會把 `#include <stdio.h>`、
            # `int main(void) {` 連同開頭的圍欄一起當碎片刪掉——`電腦程式` 的
            # C 範例就是這樣少掉前兩行，而且開頭圍欄一消失，後面每一組圍欄
            # 都跟著配對錯位（整篇 9 個標記變成奇數）。
            out.extend(lines[i:])
            break
        if seen_sentence or not stripped or stripped.startswith('- '):
            out.append(line)
            continue
        if _is_lead_debris(stripped):
            continue
        seen_sentence = True
        out.append(_strip_lead_param_prefix(stripped))
    return '\n'.join(out)


# 章節標題撞到跳過清單時，再看內文決定要不要真的丟。
#
# 清單裡的詞有一半在條目裡是正文而不是參考章節：
#     [咖啡因 > 来源]   咖啡因的天然來源與含量表
#     [OpenBSD > 许可]  ISC 授權條款的沿革，1,500 字
#     [OpenBSD > 版本历史] 各版本發行日期表
# 全量掃 50 萬頁的結果：「版本历史」有 71 節是正文、只有 5 節是參考；
# 「资源」72 比 46、「说明」49 比 23、「其他」614 比 770。光看標題丟掉，
# 這一份 dump 就有約 160 萬字的正文被當成參考資料刪掉。
#
# 用同一條規則判所有關鍵字，不逐個開特例：有成段散文或成形的表格就是正文。
# 參考章節在這個階段 `<ref>` 與 cite 模板都已經被拿掉，剩下的是條列書目，
# 兩者都不會命中。
# 參考／連結章節長什麼樣：整行是書目、裸網址、ISBN、出版社資訊。
# `<ref>` 與 cite 模板在這個階段都已經被拿掉，剩下的就是這些條列。
_SENTENCE_MARK_RE = re.compile(r'[。！？，；、]')
_TITLE_MARK_RE = re.compile(r'[《》〈〉「」『』]')
_REFERENCE_LINE_RE = re.compile(
    r'(?i)^\s*[-*•]?\s*(?:'
    r'https?://|www\.'                       # 裸網址
    r'|ISBN[\s:]|ISSN[\s:]|doi[\s:]|OCLC[\s:]'
    r'|\S+\.(?:com|org|net|edu|gov|info|cn|tw|hk|jp)\b'
    r')')
# 一行裡同時有「出版年」與「出版社／期刊」味道的，也是書目
_CITATION_HINT_RE = re.compile(
    r'(?:\d{4}\s*年?\s*[.,、，；]'          # 出版年後面接標點
    r'|第\s*\d+\s*[卷期版]'
    r'|出版社|出版公司|書館|書局|文庫|叢書|編著|譯著|主編'
    r'|Press\b|Journal\b|Publisher|Vol\.|pp?\.\s*\d)')


def _looks_like_reference_section(body):
    """這一節**證明得了**是參考書目／外部連結嗎

    判準從「像不像正文」反轉成「證明得了是參考資料嗎」。原本是
    「少於 200 字或少於 3 列表格就當成參考章節刪掉」——那是拿長度猜內容價值，
    兩列的表格、簡短的定義、作品列表全都是反例，而核心原則說得很清楚：
    **刪除的理由必須是「沒有內容」，不能是「太短」**。

    現在只有「絕大多數行都證明得了是書目／連結」才丟。證不了就留著。
    """
    lines = [l.strip() for l in body.split('\n') if l.strip()]
    if not lines:
        return True                       # 空章節，丟掉沒有損失
    refs = 0
    for line in lines:
        # 以句末標點收尾的是散文，不是書目條目——「該公司於2003年，由三位
        # 工程師共同創立。」有出版年的形態，但它是正文
        if line.endswith(('。', '！', '？')):
            continue
        if _REFERENCE_LINE_RE.match(line) or _CITATION_HINT_RE.search(line):
            refs += 1
            continue
        # 外部連結的網址在清理階段就被剝掉了，只剩標籤（「官方網站」
        # 「官方Facebook粉絲頁」）。在**編者自己標成連結／參考的章節裡**，
        # 沒有句讀的短標籤就是連結標題。這個判斷只在標題已經命中跳過清單時
        # 才會用到，不是全域規則。
        body_text = line.lstrip('-*• ').strip()
        if not body_text or len(body_text) > 20:
            continue
        # 表格列（有全形｜）與作品名（《》「」包起來）不是連結標籤——
        # 兩列的表格、作品列表都是短的，但它們是內容
        if '｜' in body_text or _TITLE_MARK_RE.search(body_text):
            continue
        if not _SENTENCE_MARK_RE.search(body_text):
            refs += 1
    return refs * 4 >= len(lines) * 3     # 四分之三以上


def _drop_orphan_fences(doc):
    """清掉配不成對的程式碼圍欄

    圍欄一定成對出現，而且一定獨佔一行——這是我們自己產生的格式。落單的、或
    夾在句子中間的 ``` 都不是圍欄，是原始碼裡本來就有的雜訊：`湘西土家族苗族
    自治州` 的正文中間就躺著一個編者留下的孤立 ```，`總是有愛在隔離` 的演員表
    裡則是「飄```移」。留著會讓下游把後面整段正文誤判成程式碼。

    只動標記本身，不碰任何內容。
    """
    if '```' not in doc:
        return doc
    lines = doc.split('\n')
    opener, drop = None, set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if opener is None:
            if stripped.startswith('```'):
                opener = i
            elif '```' in line:
                lines[i] = line.replace('```', '')
        elif stripped == '```':
            opener = None
    if opener is not None:
        drop.add(opener)
    if drop:
        lines = [line for i, line in enumerate(lines) if i not in drop]
    return normalize_whitespace('\n'.join(lines))


def build_document(content, article_title, drop_broken=True, skip_flags=None):
    """
    把清理過的 Markdown 組裝成一篇完整文檔。

    skip_flags 是「每個章節要不要丟掉」的既定答案（由簡體版算好）。給了就照用，
    不再自己判斷——判斷結果必須與語言無關，見 process_page 的說明。

    Returns:
        (文本, 被丟掉的殘句數, 章節跳過決策)
    """
    # 先把每個章節整理成 (層級, 標題, 內文)，內文可能是空的。
    # 必須保留空內文的節點，否則像「歷史」這種只有子章節的父標題會消失，
    # 導致文章的層級結構被壓平。
    sections = []
    dropped_total = 0
    flags = []

    nodes = list(iter_sections(content))
    # 旗標帶著自己那一節的「身分」（層級＋正規化標題）一起傳，不能只靠索引。
    # 章節數相同但結構不同是會發生的——變體分支裡含 `==標題==` 時，繁簡兩版
    # 的章節可能一樣多卻不是同一批。只比長度的話旗標會整段錯位，把不該丟的
    # 章節丟掉，而且兩版錯得不一樣，parity 反而看不出來。
    # 對不上的那一節就地重算，其餘照用。
    if skip_flags is not None and len(skip_flags) != len(nodes):
        skip_flags = None

    for index, (level, title, body, title_path) in enumerate(nodes):
        skip = None
        if skip_flags is not None:
            flag = skip_flags[index]
            # 身分只比**層級**，不比標題文字。標題本來就會因繁簡而不同
            # ——同一節在繁體是「連結」、簡體是「链接」，兩者正規化後仍然
            # 不同字。拿標題當身分的話這一節永遠對不上，兩版各自判斷，
            # 結果就是繁體留著、簡體丟掉（`辛貝特`、`清華大學歷史系`）。
            # 層級序列相同就代表是同一批節點；真正的結構錯位會讓層級對不上。
            if isinstance(flag, tuple) and flag[0] == level:
                skip = flag[2]
        if skip is None:
            skip = (level > 1 and should_skip_section(title_path[1:] or title_path)
                    and _looks_like_reference_section(body))
        flags.append((level, _canon_section(title), skip))
        if skip:
            # 保留節點但清空內容，不能直接 continue。
            #
            # 子章節可能因為有實質內容而被留下（`彩虹貓` 的 `來源 > 彩虹貓的動畫`、
            # `關鍵字驅動測試` 的 `說明 > 規劃階段 > 關鍵字的範例`）。整個移出清單
            # 的話，下面的「祖先保留」邏輯看不到這個父節點，輸出就會冒出沒有父層的
            # 孤兒標題——`彩虹貓` 開頭直接是 `###`、`關鍵字驅動測試` 從 `##` 跳到 `####`。
            # 留成空節點，讓既有機制決定：底下有內容才把標題留著，否則一起丟掉。
            sections.append((level, title, ''))
            continue

        body = body.strip()
        if not body:
            sections.append((level, title, ''))
            continue

        # 逐段處理再接回去，保住自然段邊界（整章一次丟進去會讓段落黏在一起）
        blocks = []
        for block in re.split(r'\n\s*\n', body):
            block = block.strip()
            if not block:
                continue
            converted = finalize_block(block)
            if converted:
                blocks.append(converted)

        text = '\n\n'.join(blocks).strip()
        if drop_broken:
            text, dropped = _drop_broken_sentences(text)
            dropped_total += dropped

        sections.append((level, title, text.strip()))

    # 決定哪些章節要輸出：自己有內文，或底下任一子章節有內文
    keep = [False] * len(sections)
    for i, (level, _title, text) in enumerate(sections):
        if not text:
            continue
        keep[i] = True
        # 往回把所有祖先章節一併標記保留
        current = level
        for j in range(i - 1, -1, -1):
            if sections[j][0] < current:
                keep[j] = True
                current = sections[j][0]
                if current <= 1:
                    break

    parts = []
    for i, (level, title, text) in enumerate(sections):
        if not keep[i]:
            continue
        # 層級 1 是前言，不需要標題；其餘保留 Markdown 標題當作章節邊界
        if level > 1 and title:
            parts.append(f"{'#' * level} {title}")
        if text:
            parts.append(text)

    if not parts:
        return '', dropped_total, flags

    # 條目標題放在文本開頭（與 wikimedia/wikipedia 的慣例一致）
    body = _drop_lead_debris('\n\n'.join(parts))
    # 文檔組裝完要再正規化一次：刪掉碎片行會留下空行，
    # 各區塊接起來也可能產生多餘的空白
    doc = normalize_whitespace(_tidy_tables(f"{article_title}\n\n{body}"))
    # 逐字區塊（LaTeX、程式碼）的遮罩留到這裡才還原——整條清理鏈都不該碰它們。
    # 提早還原的話：孤兒括號規則吃掉 `\frac{a}{b}}` 的巢狀括號、殘骸行規則整行
    # 刪掉公式、空括號清理把 `f()` 變成 `f`、行首 `#` 讓 `#include` 變成清單項。
    # 落單圍欄要在還原遮罩**之前**清掉：這一步會再正規化一次空白，而程式碼的
    # 縮排此時還是遮罩字元，正規化碰不到它。順序反過來的話，剛還原成真空白的
    # 縮排立刻會被 `\n[ \t]+` 規則吃掉，Python 範例就全部貼到行首了。
    return unmask_verbatim_braces(_drop_orphan_fences(doc)), dropped_total, flags


# 「這篇進不進資料集」一律用簡體版判定。
#
# 長度門檻若各自量各自語言的字數，收錄與否就會取決於語言版本——翻譯詞的長度
# 本來就不同（沙特阿拉伯 5 字／沙烏地阿拉伯 6 字、内存 2 字／記憶體 3 字），
# 剛好卡在門檻上的條目只會進得了一邊（實測 `蠕虫蜥属` 簡體正文 30 字達標、
# 繁體 29 字落選）。
#
# 「先統一轉成同一變體再量」救不了：轉換表不是雙向可逆，實測 60,000 篇裡
# 把繁體轉回簡體後仍有 19% 與簡體版字數不同。唯一可靠的做法是讓兩邊看同一份
# 文字，所以非簡體的語言會另外組一次簡體文檔，只拿來做這個判定。
_CANONICAL_LANG = 'cn'


def _build_one(content, page, lang, convert_variant, skip_flags=None):
    """組出某個語言的文檔，回傳 (文本, 標題, 原始標題, 章節跳過決策)；不合格回傳 None"""
    # 轉換前先留下原始標題：OpenCC 會把「斯普林里奇」轉成「斯普林裡奇」，
    # 用轉換後的標題組出來的網址在維基百科上只能 301 轉址而非直接命中，
    # 而 url 欄位是 CC BY-SA 的署名依據，必須指向真正的原文。
    content = resolve_variants(content, lang)
    raw_first = HEADING_RE.match(content.lstrip().split('\n', 1)[0])
    raw_title = _strip_heading_number(raw_first.group(2), 1) if raw_first else page.get('title', '')

    # 簡繁轉換（維基官方表 + 台灣慣用詞白名單，理由見 wiki_text.convert_script）。
    # 注意：刻意不呼叫 pangu.spacing()，語料保持原貌。
    if convert_variant:
        content = convert_script(content, lang)

    # 文件層級先清一次殘留標記，段落層級就不必重複做
    content = strip_leftover_markup(content)

    # 條目標題 = 第一個 `# ` 標題（層級 1，不去編號）
    first = HEADING_RE.match(content.lstrip().split('\n', 1)[0])
    article_title = _strip_heading_number(first.group(2), 1) if first else page.get('title', '')
    article_title = drop_empty_brackets(article_title).strip()
    if not article_title:
        return None

    # 非條目命名空間（Module:、WikiProject: 等）是程式碼或專案頁，不是知識內容
    if NAMESPACE_RE.match(article_title):
        return None

    # 消歧義頁沒有知識內容（主要判定在解析階段讀 {{disambig}} 模板）
    if DISAMBIGUATION_TITLE_RE.search(article_title):
        return None

    text, _dropped, flags = build_document(content, article_title, skip_flags=skip_flags)
    return text, article_title, raw_title, flags


_IMAGE_MARK_RE = re.compile(IMAGE_MARK + r'(\d+)' + IMAGE_MARK)
_DOC_HEADING_RE = re.compile(r'(?m)^(#{2,6}) (.+)$')


def _drop_empty_cells(text):
    """收掉「拿掉圖片之後就空掉」的表格儲存格

    表格轉文字時空儲存格本來就會跳過（`convert_tables` 的 `if not cell`），
    但儲存格裡是圖片時，那時看到的是圖片位置標記——有內容。純文字版之後把
    標記拿掉，就留下 `1｜朱德｜…｜｜1886年12月1日` 這種空欄，整份資料集
    有 9,887 筆（0.667%），是最大的硬性缺陷。

    只在純文字版做：omni 版的標記會變成 `<image>`，那一欄有內容。
    圍欄內不碰——程式碼裡的 `｜` 是逐字內容。
    """
    if '｜' not in text:
        return text
    out, in_code = [], False
    for line in text.split('\n'):
        if line.lstrip().startswith('```'):
            in_code = not in_code
            out.append(line)
            continue
        if in_code or '｜' not in line:
            out.append(line)
            continue
        cells = [c.strip() for c in line.split('｜') if c.strip()]
        out.append('｜'.join(cells))
    # 這一步跑在 normalize_whitespace 之後，收尾要自己來：整列都空掉會留下
    # 空行，剝掉儲存格也可能在行尾留下空白（實測 55 筆「行尾空白」就是這樣來的）
    return re.sub(r'[ \t]+\n', '\n',
                  re.sub(r'\n{3,}', '\n\n', '\n'.join(out))).strip()


def _drop_empty_headings(text):
    """清掉「拿掉圖片之後就沒有內容」的章節標題

    整節只有圖片的章節（`政府` 的「地圖」是兩張世界地圖、`大英國協` 的
    「成員國」是成員國旗幟圖）在組裝時看起來有內容——圖片位置標記還在——
    純文字版之後才把標記拿掉，於是留下一個空標題（實測 20 萬篇裡 670 個）。

    只能在**純文字版**做。omni 版的標記會變成 `<image>`，那些章節有圖有說明，
    本來就該留著，連同標題一起，圖片才有上下文。
    """
    heads = list(_DOC_HEADING_RE.finditer(text))
    if not heads:
        return text
    levels = [len(h.group(1)) for h in heads]
    bodies = [text[h.end():(heads[k + 1].start() if k + 1 < len(heads) else len(text))]
              for k, h in enumerate(heads)]
    keep = [bool(b.strip()) for b in bodies]
    # 有內容的章節，它的所有祖先標題都要留著，否則層級會斷掉
    for i, ok in enumerate(keep):
        if not ok:
            continue
        current = levels[i]
        for j in range(i - 1, -1, -1):
            if levels[j] < current:
                keep[j] = True
                current = levels[j]
                if current <= 2:
                    break
    if all(keep):
        return text
    out, last = [], 0
    for i, h in enumerate(heads):
        if keep[i]:
            continue
        out.append(text[last:h.start()])
        last = h.end() + len(bodies[i])
    out.append(text[last:])
    return re.sub(r'[ \t]+\n', '\n',
                  re.sub(r'\n{3,}', '\n\n', ''.join(out))).strip()
# omni 版本裡代表「這裡有一張圖」的佔位符（對齊 MMC4／OBELICS 的交錯式慣例）
IMAGE_PLACEHOLDER = '<image>'
_LITERAL_IMAGE_TAG_RE = re.compile(r'(?i)<image\s*/?>')


# 佔位符前後各留一個空行，與文件其餘部分的分段一致；多出來的收掉。
# 只作用在佔位符附近，不碰其他空白——程式碼縮排必須原樣保留。
_FENCE_BLOCK_RE = re.compile(r'(?ms)^```.*?^```[ \t]*$')


def _normalize_outside_fences(text):
    """收斂空白，但圍欄內原樣保留

    `normalize_whitespace` 會刪行尾空白、壓縮連續空白、收掉多餘空行——對散文
    是對的，對程式碼是災難（縮排全沒了）。一度為了保住縮排乾脆不呼叫它，結果
    連續空行從 0 暴增到 4.7%、行尾空白 3,947 筆。分開處理才對：圍欄外照常
    正規化，圍欄內一個字元都不動。
    """
    if '```' not in text:
        return normalize_whitespace(text)
    out, last = [], 0
    for m in _FENCE_BLOCK_RE.finditer(text):
        out.append(normalize_whitespace(text[last:m.start()]))
        out.append('\n\n' + m.group(0) + '\n\n')
        last = m.end()
    out.append(normalize_whitespace(text[last:]))
    return re.sub(r'\n{3,}', '\n\n', ''.join(out)).strip()


def build_omni(text, image_bodies, page_title, lang):
    """把正文的圖片位置標記換成 `<image>`，並產出對應的圖片清單

    只收「標記還留在正文裡」的那些圖——章節被丟掉時，它的圖也不該出現在
    清單裡，否則 `<image>` 的數量會跟清單對不上。
    """
    from image_extractor import parse_usage, _file_url

    def prose(chunk):
        # 條目正文裡本來就可能出現字面的 `<image>`（講 HTML 標籤的條目就會）。
        # 原樣留著的話，佔位符的數量會比圖片清單多，交錯式格式最根本的不變量
        # ——「第 n 個 `<image>` 對應 images[n]」——就此失效。跳脫成實體寫法，
        # 語意不變而且不再與佔位符混淆。
        return _LITERAL_IMAGE_TAG_RE.sub('&lt;image&gt;', chunk)

    kept, out, last = [], [], 0
    for m in _IMAGE_MARK_RE.finditer(text):
        idx = int(m.group(1))
        info = None
        if 0 <= idx < len(image_bodies):
            try:
                parsed = parse_usage(image_bodies[idx], page_title, lang)
            except Exception:
                parsed = None
            if parsed:
                file_name, caption, alt = parsed
                info = {'url': _file_url(file_name), 'file_name': file_name,
                        'caption': caption, 'alt': alt}
        out.append(prose(text[last:m.start()]))
        if info is not None:
            # 佔位符獨佔一行：交錯式格式裡它代表「這裡插入一張圖」，
            # 黏在句子中間會被當成正文的一部分
            out.append(f'\n\n{IMAGE_PLACEHOLDER}\n\n')
            kept.append(info)
        last = m.end()
    out.append(prose(text[last:]))
    # 同樣不能再正規化：遮罩已還原，會吃掉程式碼縮排。佔位符前後各補一個
    # 換行就夠，多餘的空行由下面這條只作用在圍欄外的規則收掉。
    return _normalize_outside_fences(''.join(out)), kept


def process_page(page, lang='tw', convert_variant=True, min_length=MIN_DOC_LENGTH,
                 omni=False):
    """
    把一筆中間層頁面記錄轉成資料集記錄（不合格時回傳 None）。

    min_length 檢查的是「正文」長度，不含開頭的標題行，且一律在簡體版上計算，
    確保兩個語言收錄的條目集合完全一致。

    **章節要不要丟掉也一樣，只在簡體版上決定一次**。跳過清單是拿標題比對的，
    而同一個章節在兩個語言的寫法不同，正規化又收斂不到同一個字：
        繁體 `連結` → 正規化 `连结` → 在清單裡 → 丟掉
        簡體 `链结` → 正規化 `链结` → 不在清單裡 → 留著
    `连／链` 是兩個都合法的簡化寫法，zhconv 不會把它們併成一個，所以「補關鍵字」
    永遠補不完（`摩斯特`、`黃帝祭` 都是這樣一邊多一個章節）。改成算一次、兩邊共用，
    這一整類問題就不會再出現。
    """
    page_id = page['id']
    try:
        content = page['text']
        if not content.strip():
            return None

        # 先組簡體版：它同時是「收不收錄」與「哪些章節要丟」的基準。
        # 繁體本來就要多組一次簡體版來過門檻，所以沒有多花成本。
        canonical = _build_one(content, page, _CANONICAL_LANG, convert_variant)
        if canonical is None:
            return None
        gate_text, gate_title, _raw, skip_flags = canonical
        if len(gate_text) - len(gate_title) < min_length:
            return None

        if lang == _CANONICAL_LANG or not convert_variant:
            text, article_title, raw_title = gate_text, gate_title, _raw
        else:
            built = _build_one(content, page, lang, convert_variant,
                               skip_flags=skip_flags)
            if built is None:
                return None
            text, article_title, raw_title, _flags = built

        if omni:
            omni_text, images = build_omni(
                text, page.get('images') or [], article_title, lang)
            return {
                'id': page_id,
                'title': article_title,
                'url': _article_url(raw_title or article_title),
                'text': omni_text,
                'images': images,
            }
        return {
            'id': page_id,
            'title': article_title,
            'url': _article_url(raw_title or article_title),
            # 只剝標記，**不要**再呼叫 normalize_whitespace——此時逐字區塊的
            # 遮罩已經還原成真空白，再正規化一次會把程式碼的縮排吃掉。
            # 標記自成一行，`strip_image_marks` 連整行帶換行一起清掉，不留空行。
            'text': _drop_empty_headings(_drop_empty_cells(
                _normalize_outside_fences(strip_image_marks(text)))),
        }
    except Exception as e:
        print(f"處理條目 {page_id} 時出錯: {type(e).__name__}: {e}")
        return None


def _record_from_built(page, built, lang, omni):
    """將已組好的語言版本派生成 plain 或 omni 記錄。"""
    text, article_title, raw_title, _flags = built
    record = {
        'id': page['id'],
        'title': article_title,
        'url': _article_url(raw_title or article_title),
    }
    if omni:
        omni_text, images = build_omni(
            text, page.get('images') or [], article_title, lang)
        record['text'] = omni_text
        record['images'] = images
    else:
        record['text'] = _drop_empty_headings(_drop_empty_cells(
            _normalize_outside_fences(strip_image_marks(text))))
    return record


def process_page_variants(page, convert_variant=True,
                          min_length=MIN_DOC_LENGTH):
    """
    同時產生 tw/cn 的 plain + omni，避免同一頁重複組裝四次。

    回傳 ``{lang: {'plain': record, 'omni': record}}``；某語言不合格時
    該語言的兩個值都是 None。內容、排序與分片門檻都沿用
    ``process_page`` / ``process_directory_doc`` 的同一套邏輯。
    """
    empty = {
        'tw': {'plain': None, 'omni': None},
        'cn': {'plain': None, 'omni': None},
    }
    page_id = page.get('id', '?')
    try:
        content = page['text']
        if not content.strip():
            return empty

        # cn 同時是收錄門檻與章節去留的唯一基準；只組一次。
        canonical = _build_one(content, page, _CANONICAL_LANG, convert_variant)
        if canonical is None:
            return empty
        gate_text, gate_title, _raw, skip_flags = canonical
        if len(gate_text) - len(gate_title) < min_length:
            return empty

        built_by_lang = {_CANONICAL_LANG: canonical}
        if convert_variant:
            built_by_lang['tw'] = _build_one(
                content, page, 'tw', convert_variant, skip_flags=skip_flags)
        else:
            # 舊 API 在 convert_variant=False 時兩邊都沿用 canonical 文本，
            # 但 omni 圖說仍依傳入的 lang 處理。
            built_by_lang['tw'] = canonical

        result = {}
        for lang in ('tw', 'cn'):
            built = built_by_lang.get(lang)
            if built is None:
                result[lang] = {'plain': None, 'omni': None}
                continue
            result[lang] = {
                'plain': _record_from_built(page, built, lang, False),
                'omni': _record_from_built(page, built, lang, True),
            }
        return result
    except Exception as e:
        print(f"處理條目 {page_id} 時出錯: {type(e).__name__}: {e}")
        return empty


def _write_shard(records, path):
    """寫出一個 Parquet 分片"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = [('id', pa.string()), ('title', pa.string()),
              ('url', pa.string()), ('text', pa.string())]
    columns = {k: [r[k] for r in records] for k, _ in fields}
    # omni 版多一個圖片陣列：每張圖有網址、檔名、圖說、替代文字
    if records and 'images' in records[0]:
        image_type = pa.list_(pa.struct([
            ('url', pa.string()), ('file_name', pa.string()),
            ('caption', pa.string()), ('alt', pa.string()),
        ]))
        fields.append(('images', image_type))
        columns['images'] = [r.get('images') or [] for r in records]
    table = pa.Table.from_pydict(columns, schema=pa.schema(fields))
    pq.write_table(table, path, compression='zstd',
                   compression_level=PARQUET_COMPRESSION_LEVEL,
                   row_group_size=PARQUET_ROW_GROUP_SIZE,
                   use_dictionary=PARQUET_USE_DICTIONARY)


def _process_shard(shard_path, lang='tw', convert_variant=True, min_length=MIN_DOC_LENGTH, limit=None, omni=False):
    """處理一整個中間層分片，回傳 (記錄清單, 略過數)"""
    out, skipped = [], 0
    for page in iter_pages(None, shards=[shard_path]):
        rec = process_page(page, lang=lang, convert_variant=convert_variant,
                           min_length=min_length, omni=omni)
        if rec is None:
            skipped += 1
        else:
            out.append(rec)
        if limit and len(out) >= limit:
            break
    return out, skipped


def _process_shard_variants(shard_path, convert_variant=True,
                            min_length=MIN_DOC_LENGTH, limit=None):
    """一次處理分片，回傳 tw/cn × plain/omni 四組記錄。"""
    out = {(lang, mode): [] for lang in ('tw', 'cn')
           for mode in ('plain', 'omni')}
    skipped = {'tw': 0, 'cn': 0}
    accepted = 0
    for page in iter_pages(None, shards=[shard_path]):
        variants = process_page_variants(
            page, convert_variant=convert_variant, min_length=min_length)
        for lang in ('tw', 'cn'):
            plain = variants[lang]['plain']
            if plain is None:
                skipped[lang] += 1
                continue
            out[(lang, 'plain')].append(plain)
            out[(lang, 'omni')].append(variants[lang]['omni'])
        accepted += variants['cn']['plain'] is not None
        if limit and accepted >= limit:
            break
    return out, skipped


def _finalize_output_state(state):
    """寫完尾端 buffer，並將暫存分片改成 HF 檔名。"""
    if state['buffer']:
        path = os.path.join(
            state['output_dir'], f'_shard_{len(state["shards"]):05d}.parquet')
        _write_shard(state['buffer'], path)
        state['shards'].append(path)
        state['buffer'] = []
        state['bytes'] = 0

    final = []
    count = len(state['shards'])
    for i, path in enumerate(state['shards']):
        target = os.path.join(
            state['output_dir'], f'train-{i:05d}-of-{count:05d}.parquet')
        if os.path.exists(target):
            os.remove(target)
        os.rename(path, target)
        final.append(target)
    return final


def process_directory_variants(input_dir, output_dirs, num_workers=None,
                               max_files=None, convert_variant=True,
                               min_length=MIN_DOC_LENGTH,
                               shard_bytes=SHARD_TARGET_BYTES):
    """
    一次走訪中間層，同時寫出 tw/cn 的純文字與 omni 資料集。

    Args:
        output_dirs: ``{'tw': '/path/to/tw', 'cn': '/path/to/cn'}``；
            omni 會寫在各目錄的 ``omni/`` 子目錄。

    Returns:
        ``{(lang, mode): (files, total)}``
    """
    from tqdm import tqdm

    if set(output_dirs) != {'tw', 'cn'}:
        raise ValueError("output_dirs 必須同時提供 'tw' 與 'cn'")

    in_shards = shard_paths(input_dir)
    if max_files:
        in_shards = in_shards[:max(1, max_files // 5000)]
    if not in_shards:
        print(f"✗ {input_dir} 沒有 pages-*.jsonl 分片")
        return {}, 0
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    states = {}
    for lang in ('tw', 'cn'):
        for mode in ('plain', 'omni'):
            output_dir = (output_dirs[lang] if mode == 'plain'
                          else os.path.join(output_dirs[lang], 'omni'))
            os.makedirs(output_dir, exist_ok=True)
            states[(lang, mode)] = {
                'output_dir': output_dir,
                'buffer': [], 'bytes': 0, 'shards': [], 'total': 0,
            }

    print(f"輸入: {input_dir}（{len(in_shards)} 個分片）")
    print(f"輸出: tw/cn 純文字 + omni｜{num_workers} 個進程｜合併單一 pass")

    worker = partial(_process_shard_variants,
                     convert_variant=convert_variant,
                     min_length=min_length, limit=max_files)
    skipped = {'tw': 0, 'cn': 0}
    with multiprocessing.Pool(processes=num_workers) as pool:
        for groups, n_skipped in tqdm(
                pool.imap(worker, in_shards), total=len(in_shards),
                desc='轉換雙語分片'):
            for lang in ('tw', 'cn'):
                skipped[lang] += n_skipped[lang]
            for key, records in groups.items():
                state = states[key]
                for record in records:
                    state['buffer'].append(record)
                    state['bytes'] += len(record['text'].encode('utf-8'))
                    state['total'] += 1
                    if state['bytes'] >= shard_bytes:
                        path = os.path.join(
                            state['output_dir'],
                            f'_shard_{len(state["shards"]):05d}.parquet')
                        _write_shard(state['buffer'], path)
                        state['shards'].append(path)
                        print(f"\n  {key[0]}/{key[1]} 寫出 "
                              f"{os.path.basename(path)}"
                              f"（{len(state['buffer']):,} 筆）")
                        state['buffer'] = []
                        state['bytes'] = 0

    result = {}
    for key, state in states.items():
        final = _finalize_output_state(state)
        size = sum(os.path.getsize(p) for p in final)
        result[key] = (final, state['total'])
        print(f"✓ {key[0]}/{key[1]}: {state['total']:,} 筆，"
              f"{len(final)} 個分片，{size / 1024 / 1024:.1f} MB，"
              f"略過 {skipped[key[0]]:,} 篇")
    return result


def process_directory_doc(input_dir, output_dir, lang='tw',
                          num_workers=None, max_files=None, convert_variant=True,
                          min_length=MIN_DOC_LENGTH, shard_bytes=SHARD_TARGET_BYTES,
                          omni=False):
    """
    把中間層的分片 JSONL 轉成文檔級 Parquet 資料集。

    以「分片」為平行單位而不是「單篇條目」——後者要在進程間傳遞上百萬個
    小任務，光是 IPC 就吃掉大半時間。

    Returns:
        (輸出檔案清單, 記錄數)
    """
    from tqdm import tqdm

    in_shards = shard_paths(input_dir)
    if max_files:
        in_shards = in_shards[:max(1, max_files // 5000)]
    if not in_shards:
        print(f"✗ {input_dir} 沒有 pages-*.jsonl 分片")
        return [], 0

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"輸入: {input_dir}（{len(in_shards)} 個分片）")
    print(f"輸出: {output_dir}")
    print(f"語言: {'繁體中文' if lang == 'tw' else '簡體中文'}｜{num_workers} 個進程")

    worker = partial(_process_shard, lang=lang, convert_variant=convert_variant, omni=omni,
                     min_length=min_length, limit=max_files)

    shards = []
    buffer = []
    buffered_bytes = 0
    total = 0
    skipped = 0

    with multiprocessing.Pool(processes=num_workers) as pool:
        for records, n_skipped in tqdm(pool.imap(worker, in_shards),
                                       total=len(in_shards), desc='轉換分片'):
            skipped += n_skipped
            for record in records:
                buffer.append(record)
                buffered_bytes += len(record['text'].encode('utf-8'))
                total += 1

                if buffered_bytes >= shard_bytes:
                    path = os.path.join(output_dir, f'_shard_{len(shards):05d}.parquet')
                    _write_shard(buffer, path)
                    shards.append(path)
                    print(f"\n  寫出分片 {os.path.basename(path)}（{len(buffer):,} 筆）")
                    buffer = []
                    buffered_bytes = 0

    if buffer:
        path = os.path.join(output_dir, f'_shard_{len(shards):05d}.parquet')
        _write_shard(buffer, path)
        shards.append(path)
        print(f"\n  寫出分片 {os.path.basename(path)}（{len(buffer):,} 筆）")

    # 依 HF 慣例改成 train-00000-of-0000N.parquet（分片總數要全部寫完才知道）
    final = []
    for i, path in enumerate(shards):
        target = os.path.join(output_dir, f'train-{i:05d}-of-{len(shards):05d}.parquet')
        if os.path.exists(target):
            os.remove(target)
        os.rename(path, target)
        final.append(target)

    size = sum(os.path.getsize(p) for p in final)
    print(f"\n✓ 完成：{total:,} 筆記錄，{len(final)} 個分片，共 {size / 1024 / 1024:.1f} MB")
    print(f"  略過 {skipped:,} 個條目（命名空間頁、消歧義頁、stub、空內容）")
    return final, total


def main():
    import argparse

    parser = argparse.ArgumentParser(description='由中間層生成文檔級 Parquet 資料集')
    parser.add_argument('--input-dir', required=True, help='中間層目錄（含 pages-*.jsonl）')
    parser.add_argument('--output-dir', required=True, help='Parquet 輸出目錄')
    parser.add_argument('--lang', choices=['tw', 'cn'], default='tw')
    parser.add_argument('--workers', type=int, help='進程數')
    parser.add_argument('--max-files', type=int, help='只處理前 N 個檔案（測試用）')
    parser.add_argument('--min-length', type=int, default=MIN_DOC_LENGTH)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            pass

    args = parser.parse_args()
    process_directory_doc(
        args.input_dir, args.output_dir, lang=args.lang,
        num_workers=args.workers, max_files=args.max_files, min_length=args.min_length,
    )


if __name__ == '__main__':
    main()


def why_dropped(page, lang='tw', min_length=MIN_DOC_LENGTH):
    """診斷用：說明一筆中間層記錄為何沒有進資料集（QA 腳本使用）"""
    try:
        content = page['text']
        if not content.strip():
            return '空內容'
        content = resolve_variants(content, lang)
        content = convert_script(content, lang)
        content = strip_leftover_markup(content)
        first = HEADING_RE.match(content.lstrip().split('\n', 1)[0])
        title = _strip_heading_number(first.group(2), 1) if first else page.get('title', '')
        title = drop_empty_brackets(title).strip()
        if not title:
            return '無標題'
        if NAMESPACE_RE.match(title):
            return '非條目命名空間'
        if DISAMBIGUATION_TITLE_RE.search(title):
            return '消歧義（標題）'
        text, _dropped, _flags = build_document(content, title)
        if len(text) - len(title) < min_length:
            return f'正文不足 {min_length} 字'
        return '（應該有進去）'
    except Exception as e:
        return f'例外 {type(e).__name__}'
