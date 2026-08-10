"""
wikitext 解析

把 dump 的 wikitext 轉成語言中立的中間層文本。三件事跟一般的 wikitext
清洗工具不同：

1. **模板展開而非刪除** —— {{lang}}、{{flag}}、{{bd}}、{{coord}}、{{convert}}
   等模板渲染後是可見文字，整批刪掉會讓條目失去實質內容。
2. **表格轉文字而非丟棄** —— 表格佔可見內容約 15%。
3. **語言變體雙版本保留** —— 編者手寫的 -{zh-tw:…;zh-cn:…}- 是最準的地區
   用詞來源，中間層是 tw/cn 共用的，所以兩個版本都帶下去，由下游各自挑。
"""

import re
import bz2
from gensim.corpora.wikicorpus import extract_pages, filter_wiki


# 語言變體標記的保留用字元（Unicode 私有使用區，正文絕不會出現）。
# markdown 是 tw/cn 共用的，所以把兩個變體都帶到下一階段再選。
VARIANT_OPEN = '\ue000'
VARIANT_SEP = '\ue001'
VARIANT_CLOSE = '\ue002'
# 圖片在正文裡的位置標記：`\ue003{序號}\ue003`。
#
# omni（圖文交錯）版本要知道「這張圖出現在哪一段之後」，所以圖片語法被移除時
# 要在原地留一個記號。用私有區字元是因為它必須活過整條清理鏈——正文絕不會
# 出現這些字元，而 wiki_text 的私有區清除規則從 \ue015 才開始。
# 序號寫進標記裡，不能只靠出現順序：中間有章節被丟掉時，剩下的標記會對不上。
#
# 用 \ue015 而不是 \ue003——後者早就被 `_BARE_LT` 佔用（裸 `<` 交給 filter_wiki
# 前的替身），撞在一起會讓標記被還原成 `<`。保留區的分配見檔案開頭。
IMAGE_MARK = '\ue015'
# \u6b63\u6587\u88e1\u7684\u88f8 `<`\uff08`\u82e5 x < 0`\u3001`<1/3 \u5c3a\u5ea6`\uff09\u5728\u4ea4\u7d66 filter_wiki \u4e4b\u524d\u7684\u66ff\u8eab
_BARE_LT = '\ue003'
# `<!` 要放行，否則 filter_wiki 認不出 `<!-- 註解 -->`，編者的內部註解會變成正文
_BARE_LT_RE = re.compile(r'<(?!/?[A-Za-z])(?![!?])')


# ============================================================
# 行內模板展開
#
# 舊版把所有 {{...}} 整批刪掉，連帶刪掉模板「渲染後會顯示的正文」，
# 造成大量條目失去實質內容，例如：
#   斯普林里奇（{{langx|en|Spring Ridge}}）是…  → 斯普林里奇（）是…
#   董萍（{{bd|1923年|11月|2019年|10月21日}}）   → 董萍（）
#   {{flag|德國}} {{flag|西班牙}}（賽事參賽名單）  → 整列消失
# 依 dump 全域統計，lang/link-en/le/tsl/langx/flag 等模板出現數十萬次，
# 展開它們可以直接救回這些內容。
# ============================================================

# 純參考／導航類模板：展開為空字串（原本就不該進語料）
_DROP_TEMPLATES = {
    'reflist', 'refbegin', 'refend', 'r', 'rp', 'sfn', 'sfnp', 'harvnb', 'harvtxt',
    'notetag', 'noteta', 'notefoot', 'efn', 'cfn', 'refgt', 'ref', 'citation',
    'wayback', 'webarchive', 'dead link', 'dead-link', 'authority control',
    'catnav', 'commonscat', 'commons category', 'commons', 'wikisource',
    'see also', 'seealso', 'main', 'main article', 'further', 'redirect',
    '簡繁重定向', '简繁重定向', 'sfnref', 'clear', 'align',
    'toc', 'tocright', 'tocleft', '-', '!', 'anchor', 'anchors',
    # 頁首導航（hatnote）：「關於其他用法，請見…」這類提示不是條目內容。
    # 未知模板的保底展開會把它們的參數當成正文，讓「數學」條目開頭多出
    # 「Math、Maths」「數學 (消歧義)」兩行。
    'about', 'redirect', 'redirect2', 'otheruses', 'otheruseslist', 'other uses',
    'distinguish', 'for', 'hatnote', 'dablink', 'this', 'noteta', 'notetaa',
    'confused', 'seealso2', 'main other', 'pp-protected', 'pp', 'protection',
    '關於', '关于', '不是', '其他用法', '各地中文名', '大陸用詞',
    # 維護模板：{{unreferenced}}、{{stub}} 這類提示框也不是內容
    'unreferenced', 'refimprove', 'citation needed', 'fact', 'cn', 'stub',
    'expand', 'expand section', 'cleanup', 'update', 'outdated', 'orphan',
    'more citations needed', 'dead end', 'copyedit', 'npov', 'disputed',
    # Wikidata 取值模板：值存在 Wikidata 不在 dump 裡，我們解不出來。
    # 保底展開會把參數關鍵字當成正文，讓 2.73% 的條目前言變成
    # 「其市鎮面積為42.84平方公里，qualifier時人口數量為property人，…」。
    # 展開成空字串後，殘句修剪會把「時人口數量為人」這個子句切掉。
    'wikidata', 'wikidatalabel', 'wikidata list', 'wd', '#property', '#statements',
    '計算', '计算',
}

# 資料查詢模板的參數關鍵字，絕不會是條目正文
_PARAM_KEYWORDS = {
    'property', 'qualifier', 'raw', 'reference', 'datavalue', 'label',
    'sitelink', 'statement', 'entityid', 'unit',
}

# 取第 N 個位置參數（1-based）當作展開結果
# 清單模板：位置參數就是清單項，一項一行。整個丟掉的話
# `{{ubl|甲|乙|丙}}` 會變成空字串，把整份清單刪掉。
_REGION_WORD_TEMPLATES = {'地区用词', '地區用詞', '地区用語', '地區用語',
                          '地区用语', '地區用语'}
# 各地區參數可以互指（`hk=cn` 表示香港跟大陸一樣）
_REGION_KEYS_TW = ('tw', 'hant', 'hk', 'mo')
_REGION_KEYS_CN = ('cn', 'hans', 'sg', 'my')


def _pick_region_word(named):
    """從地區用詞模板取出繁簡兩種寫法，包成變體標記交給下游挑"""
    def resolve(keys, seen=()):
        for key in keys:
            value = (named.get(key) or '').strip()
            if not value:
                continue
            # `hk=cn` 這種互指，跟著指過去（防環）
            if value in named and value not in seen:
                return resolve((value,), seen + (value,))
            return value
        return ''

    tw = resolve(_REGION_KEYS_TW)
    cn = resolve(_REGION_KEYS_CN)
    if not tw and not cn:
        return ''
    if not tw:
        return cn
    if not cn or tw == cn:
        return tw
    return VARIANT_OPEN + tw + VARIANT_SEP + cn + VARIANT_CLOSE


_LIST_TEMPLATES = {
    'ubl', 'unbulleted list', 'plainlist', 'plain list', 'flatlist', 'flat list',
    'hlist', 'bulleted list', 'blist', 'ublist',
}


_TAKE_NTH = {
    'small': 1, 'big': 1, 'nowrap': 1, 'nobr': 1, 'nowraplinks': 1,
    'en': 1, 'zh': 1, 'ja': 1, 'ko': 1, 'de': 1, 'fr': 1, 'es': 1, 'ru': 1,
    'italic': 1, 'i': 1, 'b': 1, 'bold': 1, 'em': 1, 'strong': 1,
    'le': 1, 'link-en': 1, 'link-ja': 1, 'link-de': 1, 'link-fr': 1,
    'link-es': 1, 'link-ru': 1, 'link-ko': 1, 'link-it': 1, 'link-pt': 1,
    'flag': 1, 'flagcountry': 1, 'flagteam': 1, 'flagathlete': 1, 'nowiki': 1,
    'lang': 2, 'color': 2, 'font color': 2, 'text': 2,
    # 上下標渲染出來就是可見文字。一度被歸進「純圖示」整個丟掉，於是
    # `H{{sub|2}}O` 變成 `HO`、`x{{sup|2}}` 變成 `x`——化學式與數學式全毀。
    'sup': 1, 'sub': 1, 'su': 1,
    'tsl': 3, 'ill': 1, 'interlanguage link': 1,
    # {{Translink|en|Architecture_of_Israel|以色列建築}}：前兩個是語言碼與
    # 外文標題，第三個才是顯示用的中文
    'translink': 3, 'tsl2': 3,
}

# 只顯示旗幟圖示、沒有文字的模板
_ICON_ONLY = {'flagicon', 'flagdeco', 'fbicon', 'flagu'}

# 化學式／方程式模板把每個符號拆成獨立參數
# （`{{反應式|Rb|OH|+|H|F|→|Rb|F}}` 渲染成 `RbOH+HF→RbF`），
# 可見輸出是所有位置參數接起來。只取第一個會讓《氟化銣》的三條方程式
# 全部剩下一個 `Rb`。
_PROP_RE = re.compile(r'\bP\d+\b')

# 表格狀態模板：沒有參數，渲染出來是一個帶底色的詞。展開成空的話，
# 賽事成績、得獎紀錄、規格對照表會出現一整排空白儲存格（實測約 4 萬次引用）。
# 有位置參數時以參數為準（`{{yes|通過}}` → 通過）。
_STATUS_TEMPLATES = {
    'yes': '是', 'ya': '是', 'no': '否', 'na': '不適用', 'n/a': '不適用',
    'won': '獲獎', 'nom': '提名', 'nominated': '提名', 'lost': '未獲獎',
    'partial': '部分', 'maybe': '可能', 'unknown': '未知', 'dunno': '未知',
    'tba': '待定', 'tbd': '待定', 'tbc': '待定', 'pending': '待定',
    'ongoing': '進行中', 'draw': '和局', 'eliminated': '淘汰',
    'yes2': '是', 'no2': '否', 'some': '部分', 'depends': '視情況',
    'included': '包含', 'free': '免費', 'proprietary': '專有',
}

_MAGIC_LITERALS = {'=': '=', '!': '|', '!!': '||', '((': '{{', '))': '}}',
                   'bang': '!', 'equals': '=', 'pipe': '|'}

_JOIN_ALL_ARGS = {'反應式', '反应式', '化學式', '化学式', 'chem', 'chem2',
                  'ce', 'equation', '化学方程式', '化學方程式'}

_TEMPLATE_RE = re.compile(r'\{\{\s*([^|{}\n]+?)\s*((?:\|[^{}]*)?)\}\}')


_NAMED_KEY_RE = re.compile(r'[A-Za-z][A-Za-z0-9_\- ]{0,39}')

# Infobox 參數名 → 可讀標籤。
#
# 側邊資訊框本身不在行文裡（渲染成頁面右側的方塊），但裡面的生卒年、人口、
# 面積是事實，而正文有一半不會重複——實測 500 篇，infobox 裡帶數字的事實
# 有 51% 在正文找不到。整塊丟掉等於比官方的 wikimedia/wikipedia 資料集還少
# （那份走 HTML 抽取，infobox 內容本來就在裡面）。
#
# 只收對得上標籤的欄位。參數名很多是英文（`GDPPC_us`、`population_nonfarm`），
# 原樣寫進語料是在教模型垃圾 token，比不收更糟。
_INFOBOX_LABELS = {
    'birth_date': '出生日期', 'birth date': '出生日期', 'born': '出生',
    'birth_place': '出生地', 'birth place': '出生地',
    'death_date': '逝世日期', 'death date': '逝世日期', 'died': '逝世',
    'death_place': '逝世地', 'death place': '逝世地',
    'nationality': '國籍', 'citizenship': '國籍', 'ethnicity': '族裔',
    'occupation': '職業', 'alma_mater': '母校', 'education': '學歷',
    'spouse': '配偶', 'children': '子女', 'parents': '父母',
    'party': '政黨', 'office': '職位', 'predecessor': '前任',
    'successor': '繼任', 'term_start': '任期開始', 'term_end': '任期結束',
    'population_total': '人口', 'population': '人口', 'pop': '人口',
    'population_as_of': '人口統計時間', 'population_density': '人口密度',
    'population_urban': '城區人口', 'population_metro': '都會區人口',
    'area': '面積', 'area_total': '總面積', 'area_total_km2': '總面積',
    'area_land': '陸地面積', 'area_water': '水域面積', 'area_urban': '城區面積',
    'elevation': '海拔', 'elevation_m': '海拔', 'coordinates': '座標',
    'timezone': '時區', 'postal_code': '郵政編碼', 'postal code': '郵政編碼',
    'areacode': '電話區號', 'area_code': '電話區號',
    'country': '國家', 'region': '地區', 'province': '省份', 'district': '區',
    'county': '縣', 'city': '城市', 'capital': '首府', 'seat': '行政中心',
    'established': '成立', 'established_date': '成立日期',
    'founded': '成立', 'founder': '創辦人', 'foundation': '成立',
    'dissolved': '解散', 'opened': '啟用', 'closed': '停用',
    'language': '語言', 'languages': '語言', 'religion': '宗教',
    'currency': '貨幣', 'gdp': 'GDP', 'gdp_total': 'GDP',
    'website': '網站', 'web': '網站', 'homepage': '網站',
    'genre': '類型', 'label': '唱片公司', 'released': '發行日期',
    'director': '導演', 'producer': '製片', 'writer': '編劇',
    'starring': '主演', 'runtime': '片長', 'budget': '預算',
    'author': '作者', 'publisher': '出版社', 'isbn': 'ISBN',
    'developer': '開發者', 'platform': '平台', 'engine': '引擎',
    'industry': '產業', 'products': '產品', 'revenue': '營收',
    'employees': '員工人數', 'headquarters': '總部',
    'president': '校長', 'students': '學生人數', 'faculty': '教職員',
    'date': '日期', 'result': '結果', 'combatant1': '參戰方一',
    'combatant2': '參戰方二', 'commander1': '指揮官一',
    'commander2': '指揮官二', 'casualties1': '傷亡一', 'casualties2': '傷亡二',
    'length': '長度', 'width': '寬度', 'height': '高度', 'depth': '深度',
    'weight': '重量', 'speed': '速度', 'manufacturer': '製造商',
    'species': '物種', 'genus': '屬', 'family': '科', 'order': '目',
    'class': '綱', 'phylum': '門', 'kingdom': '界',
    # 生物分類階元的拉丁文寫法。這一組非補不可：分類框（Taxobox／
    # Speciesbox）的欄位名是 Lua 模組算出來的，模板頁裡沒有 `label = …`，
    # 自動抽取的對照表看不到它們（見 infobox_labels.py）。而中文維基的
    # 物種條目數以十萬計，少了這組就整批掛著 `regnum：植物界`。
    # 階元是一套封閉的固定詞彙，不是逐頁修補。
    'regnum': '界', 'superregnum': '總界', 'subregnum': '亞界',
    'divisio': '門', 'phylum_la': '門', 'subdivisio': '亞門', 'subphylum': '亞門',
    'classis': '綱', 'subclassis': '亞綱', 'superclassis': '總綱',
    'ordo': '目', 'subordo': '亞目', 'superordo': '總目', 'infraordo': '下目',
    'familia': '科', 'subfamilia': '亞科', 'superfamilia': '總科',
    'tribus': '族', 'subtribus': '亞族',
    'subgenus': '亞屬', 'sectio': '組', 'series': '系',
    'binomial': '二名法', 'binomial_authority': '命名者',
    'trinomial': '三名法', 'trinomial_authority': '命名者',
    'taxon': '學名', 'authority': '命名者', 'parent': '上級分類',
    'type_species': '模式種', 'type_genus': '模式屬',
    'synonyms': '異名', 'range_map': '分布圖', 'conservation_status': '保護狀況',
}

# 分類學欄位只有在生物分類框裡才是那個意思。`class` 在汽車框是車型級距、
# 在學校框是班級、在軍艦框是艦級，一律譯成「綱」會產出
# 「綱：中大型車」（別克君越）這種胡話；`kingdom` 在君主國框是國名不是「界」。
_TAXON_KEYS = {'species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom',
               'parent', 'series', 'sectio', 'authority'}
_TAXOBOX_RE = re.compile(
    r'(?:taxobox|automatic taxobox|speciesbox|生物分類|生物分类)')

# 一行可以寫多個參數（`{{Infobox|a=|b=x}}` 是合法的 wikitext），值必須切在
# 下一個「鍵 =」之前。不切的話空值參數會把後面整段吞進來：
#   `| class = | body style = 三廂四門` → 「綱：| body style = 三廂四門」
# 用 lookahead 而不是直接以 `|` 切，才不會拆壞 `[[中國|中华人民共和国]]`。
_INFOBOX_VALUE_CUT_RE = re.compile(
    r'\|(?=\s*[A-Za-z_一-鿿][\w\- 一-鿿]{0,30}\s*=)')

_SIDEBOX_RE = re.compile(
    r'(?:infobox|info box|taxobox|automatic taxobox|speciesbox|chembox|drugbox'
    r'|navbox|sidebar|ambox|campaignbox|succession box|starbox|geobox'
    r'|資訊框|资讯框|信息框|生物分類|生物分类|化學品|化学品)'
)

# 這些模板的內容是數學式，裡面的 `=` 是運算符不是參數指派。
# MediaWiki 本身也會誤判，編者的慣例寫法是 `{{math|1=E = mc²}}`，
# 我們直接整段當內容處理，避免公式消失。
#
# 只放數學式模板。{{chem}}、{{ipa}} 有真正的具名參數
# （`{{chem|link=碘离子|I|-}}`），放進來會讓 `link=碘离子` 洩漏成正文。
_NO_NAMED_ARGS = {'math', 'mvar', 'formula', 'sfrac', 'frac'}
# 就算在公式模板裡，這些鍵仍然是排版參數而非內容
_LAYOUT_KEY_RE = re.compile(r'(?i)(?:link|style|class|id|align|size|lang)\s*=')


# 參數要切在**頂層**的 `|`。內部連結的顯示文字也用 `|` 分隔，直接 str.split
# 會把它拆成兩半：`{{legend|#FFCCCC|[[中國省份|省份]]（23個）}}` 的可見文字
# 會變成半截的 `[[中國省份`。模板參數本身不含 `{{`（_TEMPLATE_RE 就排除了），
# 所以只要數 `[[` 與 `]]`。
_ARG_TOKEN_RE = re.compile(r'\[\[|\]\]|\|')


def _split_top_level(raw):
    parts, depth, last = [], 0, 0
    for m in _ARG_TOKEN_RE.finditer(raw):
        token = m.group(0)
        if token == '[[':
            depth += 1
        elif token == ']]':
            depth = max(0, depth - 1)
        elif depth == 0:
            parts.append(raw[last:m.start()])
            last = m.end()
    parts.append(raw[last:])
    return parts


def _group_arg_rows(raw, kept):
    """依原始碼的換行把參數分組成「列」

    清單類模板一列寫一行、一列好幾個參數：

        {{common taxon list|italic=yes
        |奎瓦病毒屬 |Cuevavirus |
        |滇絲病毒屬 |Dianlovirus |
        }}

    「奎瓦病毒屬」與「Cuevavirus」是同一列的兩欄。全部用 `\\n` 串起來的話，
    學名會落單成一個只有英文的段落——這正是「英數殘骸行」的主要來源
    （`Cuevavirus`、`Gag`、`SNK` 都是這樣掉出來的）。

    不列舉模板名：編者把參數排在同一行就代表它們同屬一列，這個訊號對所有
    模板都成立。分組結果必須跟 `kept` 完全一致才採用，否則（具名參數重排、
    `1=` 數字鍵）就退回原本的一參數一行。
    """
    rows, cur, pending = [], [], False
    for part in _split_top_level(raw):
        # 換行在內容前面就斷在這個參數之前，在後面就斷在下一個參數之前。
        # 不分前後一律當成「之前」的話，`|甲\n|乙` 會把甲自己切成一列。
        if '\n' in part[:len(part) - len(part.lstrip())]:
            pending = True
        value = part.strip()
        if value in kept:
            if pending and cur:
                rows.append(cur)
                cur = []
            pending = False
            cur.append(value)
        if '\n' in part[len(part.rstrip()):]:
            pending = True
    if cur:
        rows.append(cur)
    if [v for row in rows for v in row] != kept:
        return None
    return rows


def _split_args(raw, split_named=True):
    """把 |a|b|k=v 拆成 (位置參數, 具名參數)"""
    if not raw:
        return [], {}
    if not split_named:
        return [p.strip() for p in _split_top_level(raw)[1:]
                if not _LAYOUT_KEY_RE.match(p.strip())], {}
    pos, named = [], {}
    for part in _split_top_level(raw)[1:]:
        if '=' in part:
            k, v = part.split('=', 1)
            key = k.strip()
            # 數字鍵其實是位置參數的另一種寫法：{{lang|1=en|2=text}}。
            # 維基上什麼參數名都有（0=、999=），索引要夾在合理範圍內，
            # 否則 0= 會算出 -1 而覆蓋最後一個參數甚至拋 IndexError。
            #
            # 必須用 ASCII 數字判斷：Python 的 str.isdigit() 對上標字元也回傳
            # True（'1¹¹'.isdigit() is True），但 int() 不吃，全量跑到某篇條目
            # 就會整個解析中斷。
            if key.isascii() and key.isdigit():
                idx = int(key) - 1
                if 0 <= idx <= 32:
                    while len(pos) <= idx:
                        pos.append('')
                    pos[idx] = v.strip()
                continue
            # 具名參數的鍵是 ASCII 識別字，大小寫混用很常見
            # （Infobox 的 `Name`、`Badge_size`、`FIFA Rank`、`latNS`）。
            # 早期只認全小寫，結果 `latNS=N`、`Name=Denmark` 被當成位置參數，
            # 經未知模板的保底展開變成正文，黏在條目前言的第一句前面。
            #
            # 公式模板（{{math|E = mc²}}）的 "E" 也長得像參數名，靠 _NO_NAMED_ARGS
            # 在呼叫端跳過，不在這裡用鍵的長相硬拗。
            if _NAMED_KEY_RE.fullmatch(key):
                named[key.lower()] = v.strip()
                continue
        pos.append(part.strip())
    return pos, named


# 圖片語法再長也就是一段圖說，超過這個長度幾乎都是括號不平衡造成的錯位
_MAX_FILE_LINK_SPAN = 3000

# 直接在原字串上比對（re.I），不先做 s.lower()。土耳其語的 `İ` 小寫後會變成
# 兩個碼位（i + U+0307），副本與原字串的位置從此錯開一格，後面每個 `[[File:`
# 都對不上——圖片留在正文、也沒進圖片資料集。
_FILE_LINK_START_RE = re.compile(
    r'(?i)\[\[\s*(?:file|image|media|檔案|档案|文件|圖片|图片|圖像|图像|媒體|媒体)\s*:')


def _match_span(s, start, open_tok, close_tok, max_span=None):
    """
    從 start 的開括號往後找配對的收尾，回傳結束位置（配不到回傳 -1）。

    用 str.find 在標記之間跳躍，而不是逐字元檢查——後者在 C 層只做一次
    比較，卻要付出一次 Python 迴圈的代價。實測逐字元版本讓
    remove_file_links 佔掉整個解析的 38%（每頁數萬次 startswith）。
    """
    # 呼叫端原本都會在配對後拒絕過長區段；在這裡提前停止與
    # 原輸出等價，但避免每個沒收尾的開括號都重掃整個頁尾。
    n = len(s) if max_span is None else min(len(s), start + max_span)
    depth, i = 0, start
    while i < n:
        nxt_open = s.find(open_tok, i, n)
        nxt_close = s.find(close_tok, i, n)
        if nxt_close == -1:
            return -1
        if nxt_open != -1 and nxt_open < nxt_close:
            depth += 1
            i = nxt_open + len(open_tok)
        else:
            depth -= 1
            i = nxt_close + len(close_tok)
            if depth == 0:
                return i
    return -1


_IMAGE_MARK_RE = re.compile(IMAGE_MARK + r'(\d+)' + IMAGE_MARK)


_IMAGE_MARK_LINE_RE = re.compile(
    r'(?m)^[ \t]*(?:' + IMAGE_MARK + r'\d+' + IMAGE_MARK + r'[ \t]*)+\n?')


def strip_image_marks(s):
    """拿掉圖片位置標記（純文字版本用）

    自成一行的標記要連那一行一起清掉，只清字元的話會留下空行，把原本連續的
    清單項拆散（`- 幾何圖形` 與 `- 代數符號` 之間多一個空行）。
    """
    if IMAGE_MARK not in s:
        return s
    return _IMAGE_MARK_RE.sub('', _IMAGE_MARK_LINE_RE.sub('', s))


def remove_file_links(s, sink=None):
    """
    移除 [[File:…]] 圖片語法（連同圖說），用括號配對而不是正則。

    `sink` 給的話，每張圖的原始語法會依序收進去，並在原地留下位置標記
    `\ue003{序號}\ue003`——omni 版本靠它把圖片插回正確的位置。

    圖說不進正文——它是「這張圖是什麼」的說明，屬於圖片資料集。正文要的是
    條目本身的敘述。圖說連同圖片網址與所在章節都收在 wiki_images_dataset。

    原本用貪婪的 `\\[\\[File:.*\\]\\]`，一行裡只要有圖片，整行正文就被刪光：
        [[File:Example.jpg|thumb|圖說]]這座建築由[[建築師]]設計，於1990年落成。
        → 。
    改成非貪婪也不行——圖說裡常有 [[內部連結]]，會在中間就斷掉留下殘骸。
    """
    if '[[' not in s:
        return s
    out = []
    pos = 0
    i = s.find('[[')
    while i != -1:
        if _FILE_LINK_START_RE.match(s, i):
            end = _match_span(s, i, '[[', ']]', _MAX_FILE_LINK_SPAN)
            # 配不到 `]]` 或跨度大得不合理，就只跳過 `[[` 本身。
            # 原本無條件吃到配對處，括號不平衡時會把後面整篇吃掉——《鳥》的
            # 分類章節（7,800 字）就是這樣消失的。
            if end != -1 and end - i <= _MAX_FILE_LINK_SPAN:
                out.append(s[pos:i])
                if sink is not None:
                    out.append(f'{IMAGE_MARK}{len(sink)}{IMAGE_MARK}')
                    sink.append(s[i + 2:end - 2])
                pos = end
                i = s.find('[[', pos)
                continue
        i = s.find('[[', i + 2)
    out.append(s[pos:])
    return ''.join(out)


# `__NOTOC__`、`__TOC__`、`__NOEDITSECTION__` 是排版開關，渲染後不可見
_MAGIC_WORD_RE = re.compile(r'__[A-Z]{3,}__')

# 收尾標籤是必要的，而且跨度要設上限。原本寫成 `(?:</gallery>|\Z)`，
# 未閉合的 gallery 會一路吃到頁尾——後面的章節連同正文整段消失。
_GALLERY_RE = re.compile(r'(?is)<gallery\b[^>]*>(.{0,20000}?)</gallery>')
# 圖片參數（尺寸、對齊、連結目標）不是圖說
_GALLERY_PARAM_RE = re.compile(
    r'(?i)^(?:\d+px|x\d+px|left|right|center|centre|thumb|none|border|frameless'
    r'|alt\s*=.*|link\s*=.*|lang\s*=.*)$'
)


# gallery 內的一列一定以檔名開頭（`File:x.jpg`，命名空間前綴可省略）。
# 用它認出「這一列是一張圖」，而不是用「有沒有 `|`」——後者等於要求圖說存在。
_GALLERY_FILE_RE = re.compile(
    r'(?i)^\s*(?:(?:file|image|檔案|档案|文件|圖片|图片|圖像|图像)\s*:)?'
    r'[^|\n]{1,200}\.(?:jpg|jpeg|png|gif|svg|webp|tif|tiff|ogv|ogg|webm|mid|wav|pdf|djvu|xcf)'
    r'\s*(?:\||$)')


def convert_galleries(s, sink=None):
    """
    把 <gallery> 區塊轉成圖說文字，而不是整塊刪掉。

    gallery 的圖說在渲染後的頁面上是可見文字（「程序设计」的圖像展覽章節、
    「恒河」的藝術形象章節都靠它承載內容），整塊刪除等於丟掉一整節。
    每行的形態是 `File:名稱.jpg|圖說`，只有 `|` 後面的圖說是內容。

    但只有**成句**的圖說才留在正文。`[[File:…|thumb|西洋棋]]` 這種一般圖片的
    圖說一律不進正文（它們有自己的圖片資料集，gallery 裡的圖同樣收在裡面），
    gallery 沒理由用另一套標準——不然一整節就只剩「西洋棋」「卡坦島」這種
    飄在半空的單詞（`圖版遊戲` 的畫廊、`毗濕奴` 結尾的「十大化身」）。
    圖說一律轉成清單項，不做長度篩選。

    問題從來不是「圖說沒價值」，是它被當成一般段落輸出——於是 `圖版遊戲` 的
    畫廊變成「西洋棋」「卡坦島」兩個飄在半空的段落，`毗濕奴` 結尾掛著一句
    「十大化身」。gallery 本來就是「一張圖一列」的結構，用清單呈現才對得上。

    一度改成「太短就丟掉」，量過全量 41,500 頁、254,455 條圖說之後發現那會砍掉
    54%，其中不乏真內容：`中華民國第二屆國會眾議院議員` 的議員肖像集、
    `一世一元制` 整份君主列表（那篇的正文主體就寫在 gallery 裡）。刪除的理由
    只能是「沒有內容」，不能是「太短」。
    """
    # 直接用既有的忽略大小寫正則做 gate，避免幾乎每篇都建立
    # 一份完整的大寫副本。沒有合法收尾時提早回傳，與 sub 不匹配結果相同。
    if not _GALLERY_RE.search(s):
        return s

    def repl(match):
        captions = []
        for line in match.group(1).split('\n'):
            line = line.strip()
            # 沒有 `|` 就是「只寫檔名、不寫圖說」，那仍然是一張圖：它要進圖片
            # 資料集、要在 omni 版留下佔位符。以前用 `'|' not in line` 一併跳過，
            # 等於因為「沒有圖說」而丟掉圖片本身。
            if not line or not _GALLERY_FILE_RE.match(line):
                continue
            # 第一段是檔名，其後可能是尺寸／對齊參數，最後才是圖說。
            # 要切在**頂層**的 `|`：圖說裡的內部連結也用 `|` 分隔顯示文字，
            # 直接 str.split 會把 `[[明太祖|太祖]]` 攔腰切斷，連結目標整個丟掉、
            # 圖說變成 `太祖]]洪武帝`（`一世一元制` 的君主肖像集）。
            parts = [p.strip() for p in _split_top_level(line)[1:]]
            caption = next((p for p in reversed(parts)
                            if p and not _GALLERY_PARAM_RE.match(p)), '')
            # 位置標記自成一行，放在圖說之前——omni 版會換成 `<image>`，
            # 夾在 `* ` 與圖說中間的話清單項會被拆散
            if sink is not None:
                captions.append(f'{IMAGE_MARK}{len(sink)}{IMAGE_MARK}')
                sink.append(line)
            if caption:
                captions.append('* ' + caption)
        return ('\n' + '\n'.join(captions) + '\n') if captions else '\n'

    return _GALLERY_RE.sub(repl, s)


# 章節標題。MediaWiki 的層級是 2～6，之前用 `startswith('=====')` 一層層試，
# 最深只認到 5，`====== 魔塔外圍 ======` 會被當成 5 級再剝掉 5 個等號，
# 標題變成 `= 魔塔外圍`。改成正則直接讀等號數。
#
# 兩條規則，順序不能反：
#   1. 正常收尾 `== 標題 ==`，收尾之後還有東西就當成內文（維基上有人把標題和
#      正文寫在同一行：`====加入CBA====從2004年起…`）。
#   2. 漏收尾 `==標題`，整行都是標題。維基上這種寫法不少，不接受的話整行會
#      原封不動流進正文。
# 只寫第 2 條會出事：尾巴有雜訊的 `== 人口 ==。` 會把 `==` 一起吃進標題，
# 變成「人口 ==。」（波札那、帕馬森乾酪、前1580年代都中招）。
_WIKI_HEADING_RE = re.compile(r'^(={2,6})\s*(.+?)\s*\1\s*(.*)$')
_WIKI_HEADING_OPEN_RE = re.compile(r'^(={2,6})\s*(.*?)\s*=*\s*$')

# 純樣式的參數值：CSS 顏色、尺寸、對齊關鍵字。
# 只用來「跳過」而不是「否決」——後面還有位置參數時才往後拿。
_STYLE_VALUE_RE = re.compile(
    r'(?i)^(?:#[0-9a-f]{3,8}'
    r'|\d+(?:\.\d+)?\s*(?:px|em|rem|pt|%)'
    r'|left|right|center|centre|top|bottom|middle|none|auto|nowrap'
    r'|white|black|red|blue|green|yellow|orange|purple|gray|grey|silver'
    r'|gold|pink|brown|cyan|magenta|lime|navy|teal|maroon|olive|aqua'
    r'|fuchsia|violet|indigo|beige|khaki|tan|transparent)$')

# 維基自己掛的消歧義標記模板（含各語言／各類型的別名）
# 名稱後面必須緊接 `}}` 或 `|`：`{{disambiguation needed}}` 是掛在**正常條目**
# 內文裡的「這個連結需要消歧義」清理標記，用 `\b` 收尾會把整篇條目誤刪。
# 重定向頁：`#` 後面必須跟著 REDIRECT 關鍵字。單看 `#` 不行——那是有序清單
# 的行首標記，會把整篇條目誤判成重定向丟掉。
#
# 目標寫法不能限定成 `[[…]]`：編者也寫 `#redirect:目標`（冒號、沒有方括號），
# 實測 282 篇這樣的重定向頁會漏進資料集，正文變成「- redirect:休斯敦」。
# 頁面名稱魔術字（維基會把它們渲染成條目標題本身）
# 本專案保留的私有區（變體標記、逐字遮罩、圖片位置標記）
_RESERVED_PUA_RE = re.compile(r'[\ue000-\ue016]')
_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')
_PAGENAME_MAGIC_RE = re.compile(
    r'(?i)\{\{\s*(?:PAGENAME|PAGENAMEE|PAGENAMEBASE|BASEPAGENAME|BASEPAGENAMEE'
    r'|SUBPAGENAME|SUBPAGENAMEE|FULLPAGENAME|FULLPAGENAMEE|ROOTPAGENAME'
    r'|ARTICLEPAGENAME|NAMESPACE)\s*\}\}')
_REDIRECT_PAGE_RE = re.compile(
    r'(?i)^#\s*(?:REDIRECT|重定向|重新導向|重新定向)\s*[:：]?\s*(?:\[\[|\S)')
_DISAMBIG_TEMPLATE_RE = re.compile(
    r'\{\{\s*(?:disambig|disambiguation|dab|hndis|geodis'
    r'|letter-numbercombdisambig|numberdis'
    r'|消歧[义義]|人名消歧[义義]|地名消歧[义義])\s*(?:\||\}\})', re.I)

_IMAGE_FILE_RE = re.compile(r'(?i)\.(?:jpg|jpeg|png|gif|svg|webp|tif|tiff|ogg|ogv|webm)\s*$')

# 獨佔一行的未知模板要保留內容的門檻：夠長且有句讀才算成段文字
# 模板殼裡的程式碼圍欄。維基常用側欄模板裝範例程式
# （`{{Side box|text=<syntaxhighlight>…</syntaxhighlight>|below=C 的 Hello World}}`），
# 外殼是版面元素、裡面是內容，整塊丟掉等於把範例程式碼刪掉。
_WRAPPED_FENCE_RE = re.compile(r'(?s)```[^\n]*\n.*?\n```')
# 公式方塊：`{{Equation box 1|equation=<math>…</math>}}` 把要顯示的方程式放在
# 具名參數裡。跟程式碼一樣，公式沒有中文句讀，過不了「成段正文」那一關，
# 整個模板就展開成空字串——`相對論` 的愛因斯坦場方程就是這樣消失的。
_MATH_IN_VALUE_RE = re.compile(r'(?i)<math[\s>]')
_MIN_STANDALONE_PROSE = 30
_SENTENCE_MARK_RE = re.compile(r'[。！？，；、]')
# 句**末**標點。這是比長度可靠得多的「這是成段文字」訊號：導航模板的參數是
# 語言碼、開關、分類名，不會以句號收尾。
#
# 只用長度當門檻會刪掉真內容——`{{cquote|學問之道無他，求其放心而已矣，這是
# 治學的根本方法。}}` 只有 25 字，整段引文因此消失。刪除的理由必須是「沒有
# 內容」，不能是「太短」。長度條件保留成備援，讓沒有句末標點但確實成段的
# 文字（以引號收尾的引文）仍走原本那條路。
_PROSE_END_RE = re.compile(r'[。！？]')
# 行尾的來源標註（可能有好幾個接在一起）
# 只比對**一個**行尾的 ref，多個連在一起靠迴圈重複套用。
# 原本寫成 `(?:…|…)+\s*$`：外層 `+` 包住含 `.*?` 的選擇組是巢狀量詞，
# 遇到有很多 `<ref` 卻配不到結尾的長行會指數級回溯——實測讓 13 個 worker
# 各燒掉 3 小時 50 分 CPU、整批解析卡死在最後一個分塊。
# 內層跨度也設上限，不用無界的 `.*?`。
_TRAILING_REF_RE = re.compile(
    r'(?is)(?:<ref[^>]*>.{0,2000}?</ref\s*>|<ref[^>]*/>)\s*$')


def _strip_trailing_refs(text):
    for _ in range(4):
        stripped = _TRAILING_REF_RE.sub('', text).rstrip()
        if stripped == text:
            break
        text = stripped
    return text
# 參數裡帶標記的不是引文，是還沒展開的 wikitext。不擋的話，產生表格的模板會
# 把整張表原樣吐進正文（`語系`、`世界棒球經典賽` 的表格就這樣沒被轉換）。
_MARKUP_IN_ARG_RE = re.compile(r'\{\||\|-|\[\[|\{\{|</?[a-zA-Z]|\|')

_ID_LIKE_RE = re.compile(r'^[\d./\-|_:]{4,}$')
# 跨語言連結模板的語言碼（{{Translink|en|…}} 的第一個參數）
_LANG_CODE_RE = re.compile(r'[a-z]{2,3}(?:-[a-z]{2,4})?')


def _fallback_visible(pos, named=None):
    """
    未知模板的保底展開：取第一個位置參數，但排除看起來像內部識別碼的值。

    依 dump 統計，被放在「繫詞後面的數值位置」卻沒被展開的模板還有
    ipa(170)、math(120)、chem(34)、mvar(29)… 逐一列舉永遠會漏，
    因此改用這條通用規則；純參考／導航類模板已在前面攔下。

    沒有位置參數時再看具名參數：維基會把整段內容放進具名參數讓模板去折疊
    （`{{Math proof|proof=…}}` 收著整段證明），只看位置參數的話那段正文
    會靜默消失——`群` 條目的結合律證明連同十幾條公式就是這樣不見的。
    """
    # 具名參數裡的公式與程式碼是內容，但它們沒有中文句讀，過不了下面
    # 「成段正文」那一關。`{{Equation box 1|equation=<math>…</math>}}` 的
    # 愛因斯坦場方程、`{{Side box|text=<syntaxhighlight>…}}` 的範例程式
    # 都是這樣整條消失的。這一關要放在最前面：這種模板往往沒有位置參數，
    # 但也可能有（`|indent=:` 之類），不能只在 `not pos` 的分支處理。
    # 位置參數也要看。`{{markup|<syntaxhighlight>…</syntaxhighlight>|<math>…</math>}}`
    # 是「並排展示原始碼與渲染結果」的模板，內容全在位置參數裡——只看具名參數
    # 的話 `TeX` 條目的排版範例整塊消失。含公式／圍欄的那幾個位置參數全部留下，
    # 只留第一個會丟掉旁邊那條公式。
    hits = [v.strip() for v in list(pos or []) + list((named or {}).values())
            if _MATH_IN_VALUE_RE.search(v) or _WRAPPED_FENCE_RE.search(v)]
    if hits:
        return '\n\n'.join(hits) if len(hits) > 1 else hits[0]
    if not pos:
        if named:
            for value in named.values():
                value = value.strip()
                if len(value) >= 60 and _SENTENCE_MARK_RE.search(value):
                    return value
        return ''
    # 樣式參數（顏色、尺寸、對齊）不是要顯示的文字，跳過去找真正的內容。
    # {{Colors|#fff|green|顺时针循环线}} 直接取第一個位置參數會得到 `#fff`：
    # 不只丟掉「顺时针循环线」，`#` 開頭的行到了下游還會被當成 Markdown 標題
    # （「伊予鐵道」的表格列就變成了一個假章節）。
    idx = 0
    while idx < len(pos) and _STYLE_VALUE_RE.match(pos[idx].strip()):
        idx += 1
    if idx >= len(pos):
        return ''
    value = pos[idx].strip()
    if not value or len(value) > 120:
        return ''
    if _ID_LIKE_RE.match(value):          # 15/04/04/203/000 這種行政區代碼
        return ''
    if value.lower() in _PARAM_KEYWORDS:  # 資料查詢模板的參數關鍵字，不是正文
        return ''
    if _LANG_CODE_RE.fullmatch(value):    # 跨語言連結模板的語言碼，不是正文
        return ''
    # 檔名不是正文。`{{flagicon image|Flag of Brazil (1889-1960).svg|size=22px}}`
    # 的第一個位置參數是圖檔，保底取值會把它當顯示文字寫進條目。
    if _IMAGE_FILE_RE.search(value):
        return ''
    if not re.search(r'[\w一-鿿]', value):  # 沒有任何實質字元
        return ''
    return value


# 由 template_store 建立的「無參數模板」對照表。空的時候行為與過去相同。
_TEMPLATE_STORE = {}


_COUNTRY_ALIAS = {}
# 目前條目的 Wikidata 取值（{屬性: 值}）。由 md_converter 逐篇設定。
_WIKIDATA_VALUES = {}


def set_wikidata_values(values):
    """設定目前條目可用的 Wikidata 數值（見 wikidata_store.py）"""
    global _WIKIDATA_VALUES
    _WIKIDATA_VALUES = values or {}
# 旗幟模板：顯示「旗幟 + 國名」，國名要靠 Country data 的 alias 還原
# `{{fb|GER}}`、`{{fbw|JPN}}` 是足球隊模板，顯示的同樣是國名
_FLAG_TEMPLATES = {'flag', 'flagcountry', 'flagteam', 'flagathlete', 'flu', 'flagu',
                   'fb', 'fbw', 'fb-rt', 'fbu', 'nft', 'flagioc', 'flagiocteam'}


def set_template_store(store, country_alias=None, maintenance=None):
    """注入無參數模板、國家名稱與維護模板名單（見 template_store.py）"""
    global _TEMPLATE_STORE, _COUNTRY_ALIAS, _MAINT_TEMPLATES
    _TEMPLATE_STORE = store or {}
    _COUNTRY_ALIAS = country_alias or {}
    _MAINT_TEMPLATES = maintenance or frozenset()


_MAINT_TEMPLATES = frozenset()


def _rescue_verbatim(pos, named):
    """整族丟棄的模板裡，程式碼與公式仍是內容，撈出來

    `{{efn|縮排示例：<syntaxhighlight>…</syntaxhighlight>}}` 是註腳模板，外殼
    不進正文，但那段 Python 範例沒有別的地方可去（`Python` 條目的縮排示例就
    這樣整塊消失）。判準不是模板名，是「參數裡有沒有圍欄或公式」——引用模板、
    導航模板、維護模板的參數都不會有。
    """
    hits = [v.strip() for v in list(pos or []) + list((named or {}).values())
            if _WRAPPED_FENCE_RE.search(v) or _MATH_IN_VALUE_RE.search(v)]
    return '\n\n'.join(hits)
# 解析器函式的條件分支。`{{#if:條件|成立時|不成立時}}` 這類我們算不出條件，
# 但**留著原文比什麼都糟**——`合數` 的正文就掛著 `{{#ifexpr:…, }}`。
# 取「成立」那一支：那是編者預期的主要輸出，也是維基最常渲染的結果。
_BRANCH_FUNCS = ('#if:', '#ifeq:', '#ifexpr:', '#ifexist:', '#iferror:')


def _expand_template(match, standalone=False, introduced=False):
    name = match.group(1).strip().lower()
    base = name.split('/')[0]

    pos, named = _split_args(match.group(2), split_named=base not in _NO_NAMED_ARGS)

    # {{#expr:151 + 100 + 135}} 是解析器函式，算出來才是讀者看到的數字。
    # 不算的話《Pokémon GO》會變成「一共有種寶可夢被正式開放」。
    # 只認純算術，用受限的字元集把關，不讓任意運算式進到 eval。
    if name.startswith('#expr:'):
        return _eval_expr(name[6:])

    # 維護模板（訊息方塊）不是條目內容。名單從 dump 挖出來，見 template_store。
    if base in _MAINT_TEMPLATES:
        return _rescue_verbatim(pos, named)

    # 條件式解析器函式：取「成立」那一支，別把原始碼留在正文裡
    if any(name.startswith(fn) for fn in _BRANCH_FUNCS):
        return pos[0].strip() if pos else ''
    if name.startswith('#switch:'):
        # `{{#switch:值|a=甲|b=乙|預設}}`：算不出比對結果，取預設值
        # （沒有 `=` 的最後一個位置參數），沒有預設就取第一個分支
        plain = [p.strip() for p in pos if '=' not in p and p.strip()]
        return plain[-1] if plain else (pos[0].strip() if pos else '')

    # {{wikidata|property|P1082}}：值存在 Wikidata 不在 dump 裡。查得到就填，
    # 查不到才展開成空（讓殘句修剪處理）。不填的話「INSEE市鎮編碼為。」
    # 這個事實就永遠遺失了。
    if base == 'wikidata' and _WIKIDATA_VALUES:
        args = match.group(2) or ''
        props = _PROP_RE.findall(args)
        # `{{wikidata|qualifier|P1082|P585}}` 取的是 P1082 這筆聲明的 P585
        # 限定詞（通常是時間點），不是 P1082 本身
        if 'qualifier' in args.lower() and len(props) >= 2:
            return _WIKIDATA_VALUES.get(f'{props[0]}/{props[1]}', '')
        for prop in props:
            value = _WIKIDATA_VALUES.get(prop)
            if value:
                return value
        return ''

    # `{{=}}`／`{{!}}` 是用來在模板參數裡塞入字面符號的。dump 裡這些模板的
    # 內容是給編者看的錯誤提示，直接用會把「錯誤：已嵌入模板…」寫進正文。
    if name in _MAGIC_LITERALS:
        return _MAGIC_LITERALS[name]

    # 表格狀態模板。要用完整名稱比對——`n/a` 的 base 會被 split('/') 切成 `n`
    status = _STATUS_TEMPLATES.get(name) or _STATUS_TEMPLATES.get(base)
    if status:
        return pos[0] if pos and pos[0].strip() else status

    # {{sort|排序鍵|顯示文字}}：第一個是排序用的，第二個才是讀者看到的
    if base in ('sort', 'sortname', 'sortkey') and len(pos) > 1:
        return pos[1]

    if base in _JOIN_ALL_ARGS:
        return ''.join(pos)

    if base in _ICON_ONLY:
        return ''
    # 側邊資訊框整族：渲染成頁面右側的方塊，不在正文行文裡。
    # 用名稱前綴判斷而不是列舉，維基上有上千個 Infobox 變體。
    if _SIDEBOX_RE.match(base):
        return _rescue_verbatim(pos, named)
    if base in _DROP_TEMPLATES or base.startswith('cite ') or base.startswith('cite'):
        return _rescue_verbatim(pos, named)

    # {{lang-en|Text}}、{{lang-ja|…}}：語言碼寫在模板名裡，內容是第一個參數
    if base.startswith('lang-') or base == 'langx':
        return pos[1] if base == 'langx' and len(pos) > 1 else (pos[0] if pos else '')

    # {{flag|MLT}}／{{flag|PRC|name=中国}}：顯示的是國名，不是代碼。
    # 編者寫的 name= 最準，其次查 Country data 的 alias，都沒有才退回原值。
    if base in _FLAG_TEMPLATES and pos:
        if named.get('name'):
            return named['name']
        alias = _COUNTRY_ALIAS.get(pos[0].replace('_', ' ').strip().lower())
        if alias:
            return alias

    # {{地区用词|cn=硅|tw=矽|start={{langx|en|Silicon}}}}：中文維基用它處理
    # 兩岸用詞不同的條目名，渲染出來就是該地區的用詞。整個丟掉的話條目的
    # 第一句會失去主語——`硅`／`軟體` 的開頭都變成「，是一種…」。
    if base in _REGION_WORD_TEMPLATES:
        pick = _pick_region_word(named)
        if not pick:
            return ''
        start = (named.get('start') or '').strip()
        return f'{pick}（{start}）' if start else pick

    if base in _LIST_TEMPLATES:
        items = [p.strip() for p in pos
                 if p.strip() and not _LAYOUT_KEY_RE.match(p.strip())]
        return '\n'.join('* ' + it for it in items) if items else ''

    if base in _TAKE_NTH:
        idx = _TAKE_NTH[base] - 1
        if base == 'tsl' and len(pos) < 3:
            idx = min(1, len(pos) - 1)
        return pos[idx] if 0 <= idx < len(pos) else (pos[0] if pos else '')

    # {{bd|1923年|11月|2019年|10月21日}} → 1923年11月－2019年10月21日
    if base == 'bd':
        # `{{bd|出生年|出生月日|逝世年|逝世月日}}`。參數是**成對**的：
        # 前兩個是出生、後兩個是逝世，破折號只該出現在兩組之間。
        # 只給兩個參數時是在世的人（`{{bd|1989年|4月18日}}` → 1989年4月18日），
        # 硬加破折號會變成「1989年－4月18日」，看起來像已故。
        # 這是拿線上維基逐句比對才抓到的：那篇條目只有一句散文，一個字之差
        # 就讓整篇的覆蓋率變成 0%。
        parts = [p for p in pos if p and not p.startswith('catIdx')]
        if len(parts) >= 4:
            return f'{parts[0]}{parts[1]}－{parts[2]}{parts[3]}'
        if len(parts) == 3:
            return f'{parts[0]}{parts[1]}－{parts[2]}'
        return ''.join(parts[:2])

    # {{convert|5774|ft|m}} → 5774 ft
    if base in ('convert', 'cvt'):
        if len(pos) >= 2:
            return f'{pos[0]}{pos[1]}'
        return pos[0] if pos else ''

    # {{start date and age|2009|08|18}} → 2009年8月18日
    if base.startswith('start date') or base.startswith('end date') or \
            base in ('birth date', 'death date', 'birth date and age', 'death date and age'):
        nums = [p for p in pos if p.isdigit()]
        if len(nums) >= 3:
            return f'{nums[0]}年{int(nums[1])}月{int(nums[2])}日'
        if nums:
            return f'{nums[0]}年'
        return ''

    # {{Coord|8.1|N|134.66|W|…}} → 8.1°N 134.66°W
    if base == 'coord':
        nums = [p for p in pos if re.fullmatch(r'-?\d+(?:\.\d+)?', p)]
        dirs = [p for p in pos if p in ('N', 'S', 'E', 'W')]
        if len(nums) >= 2 and len(dirs) >= 2:
            return f'{nums[0]}°{dirs[0]} {nums[1]}°{dirs[1]}'
        return ' '.join(nums[:2])

    # {{val|1.5|e=6|u=m}} → 1.5×10^6 m
    if base == 'val':
        if not pos:
            return ''
        out = pos[0]
        if named.get('e'):
            out += f'×10^{named["e"]}'
        unit = named.get('u') or named.get('ul') or (pos[1] if len(pos) > 1 else '')
        return out + (unit if unit else '')

    # {{nihongo|中文|日文|羅馬字}} → 中文（日文）
    if base == 'nihongo':
        if not pos:
            return ''
        return f'{pos[0]}（{pos[1]}）' if len(pos) > 1 and pos[1] else pos[0]

    if base == 'nbsp':
        return ' '

    # {{chem|H|2|O}} → H2O：化學式的各段要接起來，只取第一段會變成 H
    if base in ('chem', 'chem2', 'chembox'):
        return ''.join(pos)

    # 未知模板的處理方式取決於它出現的位置，這是個結構性的區分：
    #
    #   獨佔一行 → 幾乎都是 hatnote／維護提示／側邊欄／資訊框，是導航元素，整個丟棄
    #              （{{otheruse}}、{{Not}}、{{Expand language}}、{{NoteTA}}、{{Portal}}…）
    #   行內     → 幾乎都是內容（{{ipa}}、{{math}}、{{chem}}、{{mvar}}…），
    #              保留第一個位置參數，那通常就是渲染後顯示的文字
    #
    # 模板表：dump 的 Template 命名空間裡就有內容。
    # `{{MLT}}` → `{{flag|MLT}}` → 馬爾他；不查表的話整個國名會消失。
    #
    # 順序很重要：**專門處理 > 模板表 > 保底**。表若放在最前面會蓋掉專門處理
    # ——`{{no}}` 在表格裡是「否」，模板表卻查到 `[[北安大略]]`。
    #
    # 查表用**完整名稱**而不是 base：子頁面模板 `{{NUMBEROFPOKEMONGO/1}}` 的
    # base 是 `numberofpokemongo`，那個模板又引用 9 個 `/n` 子頁面，用 base 查
    # 等於每輪膨脹 9 倍（《Pokémon GO》曾因此卡住一個 worker 超過 9 分鐘）。
    if not match.group(2) and name not in _DROP_TEMPLATES:
        body = _TEMPLATE_STORE.get(name)
        if body:
            return body

    # `{{#invoke:模組|函式名|參數…}}` 是 Lua 模組呼叫，第一個位置參數是**函式名**
    # 不是內容。不剝掉的話保底規則會把它當顯示文字：《利昂內爾·梅西》的
    # `{{#invoke:ilh|main|lang-code=en|馬克西米連諾·畢安庫奇|…}}` 變成
    # 「他的兩個表兄弟main和main也是職業足球運動員」，人名整個不見。
    if base.startswith('#invoke'):
        pos = pos[1:]

    # 用位置判斷而不是列舉模板名，才不會每出現一個新模板就漏一次。
    if standalone:
        # 但引文模板也獨佔一行，而它的第一個位置參數就是引文本身。整個丟掉的話，
        # 前一句的「某某曾說：」會變成有頭沒尾——實測 1.4% 的條目中招
        # （語義網、銅鼓、泛紫聯盟…）。
        #
        # 一樣不列舉模板名：導航模板的參數是短短的設定值（語言碼、開關、
        # 分類名），引文模板的參數是成段的句子。用這個差別區分。
        lead = pos[0].strip() if pos else ''
        if ((_PROSE_END_RE.search(lead)
             or (len(lead) >= _MIN_STANDALONE_PROSE and _SENTENCE_MARK_RE.search(lead)))
                and not _MARKUP_IN_ARG_RE.search(lead)):
            return lead
        # 程式碼與公式也是內容，但它們沒有中文句讀，過不了「成段正文」那一關。
        # 維基常用側欄模板裝範例程式（`{{Side box|text=<syntaxhighlight>…}}`），
        # 也用 `{{markup|<syntaxhighlight>…</syntaxhighlight>|<math>…</math>}}`
        # 並排展示原始碼與渲染結果——後者的內容全在**位置參數**裡，只看具名
        # 參數的話 `TeX` 條目的排版範例整塊消失。兩種參數都要看，而且含公式／
        # 圍欄的每一個都要留（只留第一個會丟掉旁邊那條公式）。
        hits = [v.strip() for v in list(pos or []) + list((named or {}).values())
                if _WRAPPED_FENCE_RE.search(v) or _MATH_IN_VALUE_RE.search(v)]
        if hits:
            return '\n\n'.join(hits)
        # 具名參數同樣可能裝著整段正文——`{{Math proof|proof=…}}` 把證明收在
        # `proof=` 裡折疊起來。只看位置參數的話，`群` 條目的結合律證明連同
        # 十幾條公式會靜默消失。
        for value in (named or {}).values():
            value = value.strip()
            if (_PROSE_END_RE.search(value)
                    or (len(value) >= _MIN_STANDALONE_PROSE
                        and _SENTENCE_MARK_RE.search(value))):
                return value
        # 有引導語指向它，就把位置參數裡的可見內容留下來（清單、分類群、
        # 獎牌統計都靠這條救回來）。純設定值的參數本來就會被下面的規則濾掉。
        if introduced:
            # 樣式值（顏色、尺寸、對齊）與版面參數都不是內容。少濾一種就會漏出來：
            # `{{legend|#ff0000|共和黨四次均勝}}` 曾讓圖說變成
            # 「#ff0000⏎共和黨四次均勝」，繁簡兩版還因此差了一筆圖片記錄。
            kept = [p.strip() for p in pos
                    if len(p.strip()) > 1
                    and not _LAYOUT_KEY_RE.match(p.strip())
                    and not _STYLE_VALUE_RE.match(p.strip())
                    and not _IMAGE_FILE_RE.search(p.strip())]
            if kept:
                rows = _group_arg_rows(match.group(2) or '', kept)
                if rows:
                    return '\n'.join('｜'.join(row) for row in rows)
                return '\n'.join(kept)
        return ''
    return _fallback_visible(pos, named)


# 跨度設上限：`{|` 若沒有對應的 `|}`，非貪婪比對會抓到下一個表格的結尾，
# 把中間的正文一起吞掉。上限刻意放得遠大於任何真實表格，只當失控時的煞車。
_TABLE_RE = re.compile(r'(?s)\{\|.{0,500000}?\n\s*\|\}')
_TABLE_OPEN_LINE_RE = re.compile(r'^\{\|[^\n]*\n')
_TABLE_CLOSE_LINE_RE = re.compile(r'\n\s*\|\}\s*$')
_TABLE_CAPTION_LINE_RE = re.compile(r'(?m)^\s*\|\+.*$')
_TABLE_ROW_RE = re.compile(r'(?m)^\s*\|-.*$')
_TABLE_CELL_RE = re.compile(r'(?m)^\s*[|!]{1,2}|\|\||!!')
_EQUALS_SPACE_RE = re.compile(r'\s*=\s*')
_WHITESPACE_RE = re.compile(r'\s+')
_TABLE_FILE_ONLY_RE = re.compile(
    r'(\d+px|x\d+px|left|right|center|centre|thumb|none'
    r'|[\w\s.\-]{1,255}\.(jpg|jpeg|png|gif|svg|webp))', re.I)
# 儲存格屬性：`style="background: #ccc" |內容`。
#
# 這裡刻意先用 str.split 切開再驗證，而不是寫一條大正則。
# `(?:key=value\s*)+\|` 這種「群組加號」形態在沒有結尾 `|` 的儲存格上會指數
# 級回溯——實測讓全量解析卡死在單一條目上超過 20 分鐘。
_ATTR_NAME_RE = re.compile(r'[a-zA-Z][a-zA-Z0-9\-]{0,19}')
# 表格儲存格會用到的 HTML／wiki 屬性名
_KNOWN_ATTR = (r'(?:style|width|height|align|valign|bgcolor|color|colspan|rowspan'
               r'|scope|class|id|abbr|nowrap|border|cellpadding|cellspacing|title'
               r'|dir|lang|sortable|data-sort-value)')
_ATTR_HEAD_START_RE = re.compile(r'(?i)^[\s!"\']*' + _KNOWN_ATTR + r'\s*=')
# 引號括起來的值（含沒收尾的那種：`style="background: {{x}}` 裡的模板被移除後
# 就會留下半個引號）。值裡可以有空白，所以要先抽掉再依空白切詞。
_QUOTED_RE = re.compile(r'"[^"\n]*"?|\'[^\'\n]*\'?')


def _is_cell_attr_head(head):
    """
    這一段是不是儲存格的樣式屬性（而不是內容）。

    維基上的寫法很雜，都得認得：
        style="background: #ccc"    帶引號、值裡有空白
        style="background: #pot     模板被移除後引號沒收尾
        align=right nowrap          混著沒有值的屬性
        bgcolor=                    有鍵沒有值
    但至少要有一個 `鍵=值`，否則單獨一格寫著 `Yes` 也會被當成屬性刪掉。
    """
    if not head or len(head) > 200:
        return False
    # 最可靠的訊號是「開頭就是已知的表格屬性名 + =」。維基上的屬性寫得再亂
    # （`rowspan=4 6`、`rowspan=4 bgcolor=聯盟 (馬來西亞)`、
    # `align =center; height: 30px;`），開頭這一段都是一樣的；而真正的儲存格
    # 內容幾乎不可能以 `rowspan=`、`style=` 起頭。
    if _ATTR_HEAD_START_RE.match(head):
        return True
    # 先把引號值抽掉，再把 `鍵 = 值` 的空白收掉，否則 `rowspan = 5` 會被切成
    # 三個詞而認不出來（實測 704 筆條目因此漏掉）
    masked = _EQUALS_SPACE_RE.sub('=', _QUOTED_RE.sub(' \x00 ', head))
    tokens = masked.split()
    if not tokens or len(tokens) > 12:
        return False
    has_pair = False
    for tok in tokens:
        if tok == '\x00':                       # 被抽掉的引號值
            continue
        if '=' in tok:
            name, _, _value = tok.partition('=')
            if not _ATTR_NAME_RE.fullmatch(name):
                return False
            has_pair = True         # 值可以是空的（`bgcolor=`）
        elif not _ATTR_NAME_RE.fullmatch(tok):
            return False
    return has_pair


_CELL_LEAD_RE = re.compile(r'^[\s|!]+')


def _strip_cell_attrs(cell):
    """
    去掉儲存格的樣式屬性，保留內容。

    屬性可能出現在幾種位置，都要處理：
        style="background: #ccc" |內容      一般儲存格
        ! colspan=2 | 內容                  表頭儲存格（! 之後還有屬性）
        width=200| align=right|1,068,888    多段屬性接連出現
    """
    # 分隔符殘留（`|||`、`| |` 會讓切出來的儲存格自己帶著前綴）
    cell = _CELL_LEAD_RE.sub('', cell)

    # 由左往右逐段剝掉屬性，最多 4 段（避免病態輸入無限繞）
    for _ in range(4):
        if '|' not in cell:
            break
        head, _sep, rest = cell.partition('|')
        if rest.startswith('|'):      # `||` 是儲存格分隔符，不是屬性
            break
        head = _CELL_LEAD_RE.sub('', head).strip()
        if _is_cell_attr_head(head):
            cell = _CELL_LEAD_RE.sub('', rest)
            continue
        break

    # 整格只有屬性沒有內容（`align=right`、`rowspan="2"`），不是資料
    stripped = cell.strip()
    if _is_cell_attr_head(stripped):
        return ''
    return cell


def _convert_table(match):
    """
    把 wikitable 轉成每列一行的純文字。

    表格佔可見內容約 15%（賽事成績、參賽名單、人口統計都在裡面），
    舊版整塊刪除等於丟掉這些資料。這裡保留儲存格文字，用「｜」分隔。
    """
    body = match.group(0)
    body = _TABLE_OPEN_LINE_RE.sub('', body)      # 表格開頭的樣式列
    body = _TABLE_CLOSE_LINE_RE.sub('', body)
    body = _TABLE_CAPTION_LINE_RE.sub('', body)   # 表格標題列

    lines = []
    for row in _TABLE_ROW_RE.split(body):
        cells = []
        pulled = []          # 從儲存格搬出來的多行程式碼區塊
        # 行首的分隔符可能是 `|`、`!`，也可能是換行後才寫的 `!!`／`||`，
        # 只吃一個字元會把第二個 `!` 留在儲存格內容裡。
        for raw in _TABLE_CELL_RE.split(row):
            cell = raw.strip()
            if not cell:
                continue
            cell = _strip_cell_attrs(cell).strip()   # 去掉 style="…" | 之類的屬性
            # 儲存格裡的程式碼圍欄要先轉成行內反引號。下一行會把整格壓成一行，
            # 圍欄的開頭標記就會跟內容黏在一起（```` ```bash {λf.f(b+2c)…} ````），
            # 那就不再是圍欄了——語言對照表（`ISWIM`、`Java和C++的对照`、
            # `C++类`）整欄的程式碼都是這樣失效的。
            cell = _take_cell_fences(cell, pulled)
            cell = _WHITESPACE_RE.sub(' ', cell)
            # 圖片尺寸、對齊參數、純檔名不是資料，不要留在表格文字裡
            if _TABLE_FILE_ONLY_RE.fullmatch(cell):  # 255 = MediaWiki 檔名上限
                continue
            if cell and cell not in {'-', '—', '–'}:
                cells.append(cell)
        if cells:
            lines.append('｜'.join(cells))
        # 搬出來的程式碼接在該列後面，各自成塊
        lines.extend(pulled)

    return ('\n' + '\n'.join(lines) + '\n') if lines else ''


_CELL_FENCE_RE = re.compile(r'(?s)```[^\n]*\n(.*?)\n?```')


def _take_cell_fences(cell, blocks):
    """處理儲存格裡的程式碼圍欄

    儲存格接著會被壓成一行，圍欄的開頭標記會跟內容黏在一起
    （```` ```bash {λf.f(b+2c)…} ````），那就不再是圍欄了——語言對照表
    （`ISWIM`、`Java和C++的對照`、`C++類`）整欄的程式碼都是這樣失效的。

    單行的程式碼用 `` ` `` 標起來留在格子裡，語意不變；多行的整塊搬出表格，
    接在該列後面自成圍欄。一度把多行的也用 ` / ` 串成一行，結果整個 class
    定義擠在一條反引號裡，比原本更難讀。
    """
    def repl(m):
        body = m.group(1).strip('\n')
        lines = [l for l in body.split('\n') if l.strip()]
        if not lines:
            return ''
        if len(lines) == 1:
            one = lines[0].strip()
            return one if '`' in one else f'`{one}`'
        blocks.append(m.group(0))
        return ''

    return _CELL_FENCE_RE.sub(repl, cell)


def convert_tables(text):
    """遞迴處理巢狀表格，由內而外轉換"""
    for _ in range(3):
        new = _TABLE_RE.sub(_convert_table, text)
        if new == text:
            break
        text = new
    return text


_LIST_PREFIX_RE = re.compile(r'^[ \t]*([*#:;]+)[ \t]*')


def _normalize_list_prefixes(s):
    """
    正規化行首的清單標記。

    wikitext 的行首標記是一整串 `*#:;` 的組合（`::#` 是「縮排兩層的有序
    列表」）。舊版只剝掉一個 `:`，剩下的 `#` 到了下游就被當成 Markdown
    標題——《香港甲組足球聯賽》的排名規則因此變成五個 h1 標題。

    這裡一次處理整串：純 `:`／`;` 是縮排或定義列表，攤成一般段落；
    含 `*` 或 `#` 的是清單，統一輸出 `* `（層級在下游本來就會攤平）。

    定義列表（`;詞條` 後面接 `:解釋`）要把兩行併成一句。拆開的話詞條會變成
    一個孤零零的單詞段落，跟後面的解釋完全脫節——`人類免疫缺陷病毒` 的
    `;Gag` ／ `::gag 基因產生…` 就是這樣變成一行只有「Gag」的。
    """
    out = []
    lines = s.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        m = _LIST_PREFIX_RE.match(line)
        if not m:
            out.append(line)
            continue
        body = line[m.end():].strip()
        if not body:
            continue                                  # 只有標記沒有內容
        prefix = m.group(1)
        if prefix.endswith(';') and i < len(lines):
            nm = _LIST_PREFIX_RE.match(lines[i])
            defn = lines[i][nm.end():].strip() if nm else ''
            if defn and set(nm.group(1)) == {':'}:
                out.append(body + ('' if body[-1] in '：:' else '：') + defn)
                i += 1
                continue
        out.append(body if set(prefix) <= {':', ';'} else '* ' + body)
    return '\n'.join(out)


# 資訊框的參數要按巢狀深度切，不能用正則找「行首的 `|鍵 =`」。
# `awards={{plainlist|⏎* 諾貝爾獎⏎* 科普利獎章⏎}}` 的收尾 `}}` 自成一行，
# 正則會在那裡斷掉，留下沒收尾的 `{{plainlist|`——模板展不開，模板名直接
# 漏進正文變成「awards：plainlist| * 諾貝爾獎 * 科普利獎章」。
_INFOBOX_TOKEN_RE = re.compile(r'\{\{|\}\}|\[\[|\]\]|\{\||\|\}(?!\})|\|')
_INFOBOX_KEY_RE = re.compile(r'^\s*([A-Za-z_一-鿿][\w\- 一-鿿]{0,30})\s*=')


def _split_infobox_params(body):
    """把資訊框內文依巢狀深度切成 (鍵, 值)，巢狀模板／連結裡的 `|` 不算分隔

    深度要**跨行保留**：巢狀模板本來就常常跨好幾行寫
    （`亳州市` 的 `|image = {{multiple image⏎ | total_width = 300⏎ …}}`），
    在換行時把深度歸零會把那些內部參數拆成頂層欄位，於是「基本資料」冒出
    `total_width：300`、`image_style：border:1px` 這種版面設定。

    但編者手誤留下沒收尾的 `[[`／`{{` 也很常見（`邢臺市` 寫著
    `|分類 = [[地級市` 就換行了），深度從此回不到 0，後面整個資訊框會被吞成
    同一個值。所以改成：先照深度切，切完再檢查——**只有在某個值自己
    括號不平衡時**，才把那一個值退回逐行切，錯誤就只波及那一列。
    """
    parts, depth, last = [], 0, 0
    for m in _INFOBOX_TOKEN_RE.finditer(body):
        token = m.group(0)
        if token in ('{{', '[[', '{|'):
            depth += 1
        elif token in ('}}', ']]', '|}'):
            depth = max(0, depth - 1)
        elif depth == 0:
            parts.append(body[last:m.start()])
            last = m.end()
    parts.append(body[last:])
    for part in parts[1:]:              # parts[0] 是模板名
        km = _INFOBOX_KEY_RE.match(part)
        if not km:
            continue
        key, value = km.group(1), part[km.end():]
        # 這個值自己括號不平衡（原始碼沒收尾），才退回逐行切
        if '\n' in value and (value.count('[[') != value.count(']]')
                              or value.count('{{') != value.count('}}')):
            lines = value.split('\n')
            yield key, lines[0]
            for line in lines[1:]:
                lm = _INFOBOX_KEY_RE.match(line.lstrip().lstrip('|'))
                if lm:
                    yield lm.group(1), line.lstrip().lstrip('|')[lm.end():]
            continue
        yield key, value


# 值裡的分項邊界（`<br>`、展開後的清單項目）。用 NUL 當臨時分隔符：正文不會
# 有這個字元，而且不佔用已分配的 PUA 區段。
_ITEM_SEP = '\x00'
# 連續的項目符號要一起吃掉：`{{plainlist|* A\n* B}}` 展開後第一行是 `* * A`
_INFOBOX_LIST_ITEM_RE = re.compile(r'(?m)^[ \t]*(?:[*#]+[ \t]*)+')


def _infobox_value_text(value):
    """把參數的值整理成一行可讀文字（清單模板展開成頓號並列）"""
    value = _INFOBOX_BR_RE.sub(_ITEM_SEP, value)
    # 先就地展開，才知道 `{{ubl|相對論|光電效應}}` 是兩個並列的值。
    # 留到後面整篇一起展開的話，展開出來的 `* ` 會黏在事實行中間。
    if '{{' in value:
        value = expand_inline_templates(value)
    value = _INFOBOX_LIST_ITEM_RE.sub(_ITEM_SEP, value)
    items = []
    for seg in value.split(_ITEM_SEP):
        seg = ' '.join(seg.split()).strip('、,，;；| \t')
        if seg and seg not in items:
            items.append(seg)
    return '、'.join(items)
# 值裡有這些就不是事實而是版面設定（圖片檔、尺寸、色碼）
# 值**整個**就是檔名／尺寸／色碼才跳過。用子字串搜尋的話，
# `known_for=1080px顯示器的發明`、`awards=使用example.png格式的標準`
# 會被整欄刪掉——那是真實內容，不是版面設定。
_INFOBOX_SKIP_VALUE_RE = re.compile(
    r'(?i)^(?:[\w.()\-–—,&+! ]{1,80}\.(?:jpg|jpeg|png|gif|svg|webp)'
    r'|\d+\s*x?\s*\d*\s*px|x\d+px|#[0-9a-f]{3,6})$')
# 純版面用的參數名。白名單外的欄位不再一律丟棄——`known_for`、`awards`、
# `workplaces` 都是實質內容，「英文參數名不在表裡」不是刪除的理由。
# 改成只擋掉確定是版面設定的鍵。
_INFOBOX_LAYOUT_KEY_RE = re.compile(
    r'(?i)^(?:image|img|photo|logo|map|flag|icon|picture|file|caption|alt'
    # `class` 不在這裡：HTML 的 class 屬性只出現在模板定義裡，條目寫的
    # `| class = ` 是內容（`大和型戰艦`、`中大型車`、分類階元的綱）。
    # `label`／`data` 也不在這裡——條目直接用通用 `{{Infobox}}` 時，
    # `labelN`／`dataN` 成對就是一條事實，下面會配對輸出。
    r'|size|width|height|align|valign|color|colour|bg|background|style|id'
    r'|border|float|module|header|above|below|subheader|sortkey'
    r'|order|nowrap|display|hidden|state|collapsible|autocollapse)'
    r'[_ ]?\d*$')
# `<br>` 是換行不是雜訊：`巴黎<br>倫敦` 是兩個值，換成頓號而不是整筆丟掉
_INFOBOX_BR_RE = re.compile(r'(?i)<br\s*/?>')


# 版面框：導覽列、警告框、選戰框、繼任框。它們跟事實資訊框共用命名慣例，
# 但內容是導覽元素——不分開的話「基本資料」會冒出 `list1：夏朝、商朝`、
# `group1：上古`（實測中間層有 11,035 個 `list1：`）。
_LAYOUT_BOX_RE = re.compile(
    r'(?i)(?:navbox|nav box|sidebar|side bar|ambox|imbox|cmbox|tmbox|ombox'
    r'|campaignbox|succession box|s-|navigation|導航|导航|導覽|导览)')


# 從 dump 的模板頁自動抽出的「參數 → 中文標籤」對照表（見 infobox_labels.py）。
# 沒設也能跑，只是標籤會退回英文原鍵。
# 動態標籤是條目自己填的值，過長就不是欄位名（是整段敘述）
_MAX_DYNAMIC_LABEL = 12
# 條目直接寫的通用資訊框列：`| label3 = 創辦人`
_GENERIC_LABEL_RE = re.compile(r'^label(\d+[a-z]?)$')
# 鍵尾的序號（`term_start2`、`office3`）。底線／空白一起吃掉。
_TRAILING_NUM_RE = re.compile(r'[ _]?\d+$')
_CJK_KEY_RE = re.compile(r'[一-鿿]')
_LABEL_STATIC = {}
_LABEL_DYNAMIC = {}
_LABEL_ALIAS = {}
_LABEL_RENDERED = {}
_LABEL_BY_BOX = {}
_LABEL_BY_BOX_DYNAMIC = {}


def set_infobox_labels(static=None, dynamic=None, alias=None,
                       by_box=None, by_box_dynamic=None, rendered=None):
    """注入資訊框標籤表（要在 fork 之前呼叫，worker 靠 copy-on-write 共用）"""
    global _LABEL_STATIC, _LABEL_DYNAMIC, _LABEL_ALIAS, _LABEL_RENDERED
    global _LABEL_BY_BOX, _LABEL_BY_BOX_DYNAMIC
    _LABEL_STATIC = static or {}
    _LABEL_DYNAMIC = dynamic or {}
    _LABEL_ALIAS = alias or {}
    _LABEL_BY_BOX = by_box or {}
    _LABEL_BY_BOX_DYNAMIC = by_box_dynamic or {}
    _LABEL_RENDERED = rendered or {}


def _label_key(key):
    """對照表的鍵一律「底線換空白、轉小寫」"""
    return key.replace('_', ' ').strip().lower()


def extract_infobox_facts(s, page_title=''):
    """
    從側邊資訊框抽出「對得上標籤」的事實，回傳一段 wikitext。

    產出的段落接在條目末尾，之後走既有的清理流程（模板展開、繁簡轉換、
    殘留標記清理），不必另外處理值裡的連結與模板。

    標籤的來源依序是：手寫對照表 → 條目自己提供的動態標籤 → 該模板頁寫的
    標籤 → 全域最常見的標籤 → 英文原鍵。查不到就保留原鍵，絕不因為
    「標籤不好看」而丟掉事實。
    """
    facts = []
    seen = set()
    for m in re.finditer(r'\{\{\s*([^|{}\n]{1,60})\s*\|', s):
        box = m.group(1).strip().lower()
        if not _SIDEBOX_RE.match(box) or _LAYOUT_BOX_RE.search(box):
            continue
        is_taxobox = _TAXOBOX_RE.search(box) is not None
        end = _match_span(s, m.start(), '{{', '}}', _MAX_TEMPLATE_SPAN)
        if end == -1 or end - m.start() > _MAX_TEMPLATE_SPAN:
            continue
        box_static = _LABEL_BY_BOX.get(_label_key(box), {})
        box_dynamic = _LABEL_BY_BOX_DYNAMIC.get(_label_key(box), {})

        # 先把整個框的參數收齊，動態標籤才查得到值：`| subdivision_type1 = 省`
        # 是 `| subdivision_name1 = 广东省` 這一列的欄位名，維基渲染出來是
        # 「省：广东省」，不是「subdivision_name1：广东省」。
        params = []
        by_name = {}
        for raw_key, raw_value in _split_infobox_params(s[m.start() + 2:end - 2]):
            key = _label_key(raw_key)
            params.append((key, raw_key.strip(), raw_value))
            by_name.setdefault(key, raw_value)

        consumed = set()          # 已經當成別人的標籤用掉的參數，不再自成一列
        rows = []

        # 條目直接用通用 `{{Infobox}}` 時，欄位名是條目自己寫的：
        # `| label3 = 創辦人 | data3 = 張三`。這跟模板頁裡的寫法一模一樣，
        # 配對輸出即可，兩個參數都不再各自成列。
        for key in list(by_name):
            num = _GENERIC_LABEL_RE.match(key)
            if not num:
                continue
            data_key = 'data' + num.group(1)
            if data_key not in by_name:
                continue
            label = _infobox_value_text(by_name[key])
            if not label or len(label) > _MAX_DYNAMIC_LABEL:
                continue
            consumed.add(key)
            rows.append((data_key, label, by_name[data_key]))
            consumed.add('__emitted__' + data_key)
        for key, shown_key, raw_value in params:
            if key in consumed or '__emitted__' + key in consumed:
                continue
            # 分類階元的鍵在別的框裡是別的意思（`class` 在汽車框是車型級距、
            # 在軍艦框是艦級）。那只代表「不能套用分類學譯名」，不代表這一欄
            # 沒有內容——以前整欄跳過，`class = 中大型車` 就這樣消失了。
            wrong_sense = key in _TAXON_KEYS and not is_taxobox
            if _INFOBOX_LAYOUT_KEY_RE.match(key.replace(' ', '_')):
                continue
            # 條目自己填的欄位名最準——`established_title=设立地级市` 就是
            # `established_date` 那一列在維基上真正渲染出來的標籤，勝過任何猜測。
            label = ''
            type_key = box_dynamic.get(key) or _LABEL_DYNAMIC.get(key)
            if type_key and type_key in by_name:
                dyn = _infobox_value_text(by_name[type_key])
                if dyn and len(dyn) <= _MAX_DYNAMIC_LABEL:
                    label = dyn
                    consumed.add(type_key)
            if not label and not wrong_sense:
                label = (_INFOBOX_LABELS.get(key.replace(' ', '_'))
                         or _INFOBOX_LABELS.get(key))
            if not label:
                label = box_static.get(key)
            # 鍵本身就是中文時，它就是欄位名，不必再去全域表猜。全域表是
            # 「這個參數名在所有模板裡最常見的標籤」，跨模板猜會錯得離譜——
            # `邢臺市` 的 `|面積 =` 被某個學校資訊框的表配成「校园面积」。
            if not label and _CJK_KEY_RE.search(key):
                label = key
            if not label:
                # 渲染反推要**贏過**全域最常見的標籤。全域表是「這個參數名在所有
                # 模板裡最常見的標籤」，跨模板猜；渲染反推是拿該模板的實際條目
                # 逐列比對值得到的，證據強得多。`亳州市` 只寫了 `leader_name`
                # 沒寫 `leader_title`（標籤由模板寫死），全域表猜成「主席」，
                # 渲染出來其實是「市委書記」。
                label = (_LABEL_RENDERED.get(key)
                         or _LABEL_STATIC.get(key)
                         or _LABEL_ALIAS.get(key))
            if not label:
                # 同一種欄位重複好幾列時，維基的寫法是在鍵尾加編號
                # （`term_start` / `term_start2` / `office3`）。去掉編號再查一次，
                # 不然「第二任期」這種欄位全部掛著英文原鍵。
                # 一定要先查完整的鍵：`area_total_km2` 的 `2` 是單位不是編號。
                base = _TRAILING_NUM_RE.sub('', key).strip()
                if base and base != key:
                    label = (_INFOBOX_LABELS.get(base.replace(' ', '_'))
                             or _INFOBOX_LABELS.get(base)
                             or box_static.get(base)
                             or _LABEL_STATIC.get(base)
                             or _LABEL_ALIAS.get(base))
            # 查不到就用原鍵當標籤，維持原始碼裡的寫法（`known_for`，
            # 不是正規化後的 `known for`）
            rows.append((key, label or shown_key, raw_value))

        for key, label, raw_value in rows:
            if key in consumed:
                continue
            value = _infobox_value_text(raw_value)
            if not value or _INFOBOX_SKIP_VALUE_RE.search(value):
                continue
            # 值就是條目標題的欄位（`name`、`title`、`官方名稱`）沒有帶進任何
            # 資訊——標題就在文件第一行。實測佔事實行的 3.0%。
            if page_title and value == page_title:
                continue
            # 去重要看「標籤＋值」。只看標籤的話，第二任期、第二個職位這種
            # 同標籤的不同事實會被當成重複刪掉。
            if (label, value) in seen:
                continue
            seen.add((label, value))
            facts.append(f'* {label}：{value}')
    if not facts:
        return ''
    return '\n\n== 基本資料 ==\n' + '\n'.join(facts) + '\n'


_COMMENT_RE = re.compile(r'(?s)<!--.*?(?:-->|\Z)')
# 自閉合的 `<ref name="x" />` 沒有收尾標籤，不能讓它跟遠處的 `</ref>` 配成一對
# ——`捷爾諾波爾州` 的前言變體標記裡有三個自閉合 ref，配錯之後整篇正文被吃掉，
# 簡體版的章節全部消失。跨度也要設上限，理由同其他所有配對規則。
_REF_BLOCK_RE = re.compile(r'(?s)<ref(?![^>]*/\s*>)[^>]*>.{0,20000}?</ref\s*>')
# 這一步跑在 `_fence_code_blocks` **之後**（就在 filter_wiki 之前），
# 所以認的是已經轉好的 ``` 圍欄與 $$／$ 公式。
# 行內公式也要撈：`參宿四` 的半徑推導寫在 `<ref group="note">` 裡，十條公式中
# 只有獨立成行的那條會變成 `$$…$$`，其餘因為同一行還有句號或粗體標記而成為
# 行內 `$…$`。只認 `$$` 的話那份推導只救回一條。
_VERBATIM_IN_REF_RE = re.compile(
    r'(?s)```[^\n]*\n.*?\n```|\$\$.+?\$\$|\$[^$\n]{1,300}\$')


def _keep_ref_verbatim(m):
    """刪掉 `<ref>` 外殼，但把裡面的程式碼與公式留下"""
    kept = [k.group(0) for k in _VERBATIM_IN_REF_RE.finditer(m.group(0))]
    return ('\n\n' + '\n\n'.join(kept) + '\n\n') if kept else ''


def strip_comments(s):
    """移除 HTML 註解，但**不碰逐字區裡的**

    MediaWiki 不會刪 `<pre>`／`<syntaxhighlight>`／`<nowiki>` 裡的註解——
    那是要照字面顯示的內容。無條件刪掉的話，`HTML` 條目示範註解寫法的
    `<pre><!-- This is a comment --></pre>` 會整塊消失，`Active Server Pages`
    的 ASP 範例也少掉解說用的註解行。

    但註解仍要**先於**其他處理清掉：編者常在註解裡放半截的標記
    （`聖雄甘地` 有 `<!-- {Critique of political economy}} -->`），那個多出來的
    `}}` 會讓大括號配對錯位，把 33,000 字的正文當成一個模板整段吃掉。
    所以先算出逐字區的範圍，只在範圍外動手。
    """
    if '<!--' not in s:
        return s
    spans = [(m.start(), m.end()) for m in _VERBATIM_RE.finditer(s)]
    if not spans:
        return _COMMENT_RE.sub('', s)
    # 註解要在**完整的文字**上比對，不能把文字切成段再逐段刪。
    # 切段刪的話，跨越逐字區邊界的註解只會被刪掉前半：`_COMMENT_RE` 的 `\Z`
    # 分支把前段吃到底，後段少了 `<!--` 就不匹配，尾巴留在正文裡——維基的
    # reflist 樣板註解本身就寫著 `<ref>` 與 `</ref>`，實測 11 篇條目因此
    # 留下「…using the <ref> and </ref> tags, and the template below-->」。
    def inside_verbatim(a, b):
        return any(lo <= a and b <= hi for lo, hi in spans)

    out, last = [], 0
    for m in _COMMENT_RE.finditer(s):
        if inside_verbatim(m.start(), m.end()):
            continue                   # 逐字區裡的註解是內容，原樣保留
        out.append(s[last:m.start()])
        last = m.end()
    out.append(s[last:])
    return ''.join(out)
# 分類連結（含 `|排序鍵`），大小寫都要認
# 開頭的 `:` 要一起認：`[[:Category:鄄城人]]` 是「連到分類頁」的寫法，渲染成
# 一個標籤為「Category:鄄城人」的連結。沒收掉的話那串前綴會當成正文留下來
# （鄄城縣、尿道下裂、喉鏡檢查術都有）。導覽元素不是條目內容，一律移除。
_CATEGORY_LINK_RE = re.compile(
    r'(?i)\[\[\s*:?\s*(?:Category|Template|分類|分类|模板)\s*:[^\]\n]{0,200}\]\]')

# 逐字區塊：裡面的大括號是 LaTeX 或程式碼，不是模板語法。
# 《法拉第電磁感應定律》的 `<math>\frac{a}{b}}</math>` 會讓配對錯位，
# 把後面 4,700 字的正文當成模板吃掉。先換成佔位字元，配對完再換回來。
# 跨度要設上限。標籤沒收尾時，非貪婪比對會一路找到很後面的另一個收尾標籤，
# 把中間整段正文都當成逐字內容遮起來——遮住的範圍裡模板不會展開，最後以
# `{{le|FTA接收器|FTA receiver}}` 的字面形式流進資料集（Linux內核、Python、
# Emacs 都是這樣）。上限放得遠大於任何真實的程式碼或公式區塊，只當煞車。
_MAX_VERBATIM_SPAN = 20000
_VERBATIM_RE = re.compile(
    r'(?is)(<(math|chem|ce|syntaxhighlight|source|pre|nowiki|templatedata)(?![\w.\-])[^>]*(?<!/)>)'
    r'(.{0,%d}?)(</\2\s*>)' % _MAX_VERBATIM_SPAN
)
# 逐字區塊裡要遮起來的字元，每個都對應一條會誤傷它們的下游規則：
#   { }  孤兒括號規則會把 LaTeX 的巢狀括號成串刪掉
#   |    殘骸行規則與表格切分會截斷公式和程式碼
#   <    會被當成 HTML 標籤，吃掉到下一個 > 為止的內容
#   #    行首的 `#` 是 wikitext 的有序清單，`#include` 會變成 `- include`
#   ( )  空括號清理會把 `f()` 變成 `f`
_LBRACE, _RBRACE = chr(0xe004), chr(0xe005)
# 逐字就是逐字：區塊裡的每個會被下游規則動到的字元都要遮起來。
#   { } | < #     括號配對、表格切分、HTML 標籤、行首有序清單
#   ( )           空括號清理（`f()` → `f`）
#   [ ]           連結語法（mhchem 的 `->[MnO2]` 會變成 `->MnO2`，意思就變了）
#   ' * = : ;     粗體標記（Python 的 `''` 空字串）、清單、標題、縮排、定義列表
_VERBATIM_MASK = tuple(zip("{}|<#()[]'*=:;>", (chr(0xe004 + n) for n in range(15))))
_VMASK = dict(_VERBATIM_MASK)
_VLT, _VGT = _VMASK['<'], _VMASK['>']
# 程式碼多遮一層空白：縮排在 Python 裡是語法的一部分，而清理鏈為了收斂排版
# 會把 `\n[ \t]+` 壓成 `\n`、把連續空白併成一個——整段程式碼會被推到行首，
# 縮排結構蕩然無存。只對程式碼標籤這麼做：數學式的空白該正規化就正規化。
# 全形空格（U+3000）也要遮。顏文字與 ASCII 藝術完全靠它排版——
# `2channel文字人物` 的 `　（　・∀・）` 不遮的話，空白正規化會把全形空格
# 收掉，`（　　　　）` 更會被「空括號」規則整個刪成空行，圖案就毀了。
_CODE_WS_MASK = ((' ', chr(0xe013)), ('\t', chr(0xe014)), ('\u3000', chr(0xe016)))
_CODE_TAGS = ('pre', 'syntaxhighlight', 'source')
_MASKED_WS = ''.join(mask for _raw, mask in _CODE_WS_MASK)
# 行內程式碼用的遮罩：逐字遮罩去掉 `{}|[]`（模板與連結在 <code> 裡照樣渲染），
# 再加上空白
_INLINE_CODE_MASK = tuple(
    (raw, mask) for raw, mask in _VERBATIM_MASK if raw not in '{}|[]'
) + _CODE_WS_MASK
_INLINE_BRACKET_MASK = tuple(
    (raw, mask) for raw, mask in _VERBATIM_MASK if raw in '[]')


# 逐字區塊在輸出裡要有明確邊界，否則跟行文混在一起分不出來，殘留標記的檢查
# 也沒辦法把它們排除（`<pre>` 裡的 `log<sub>2</sub>k` 是頁面上真的會顯示的
# 文字，不是殘骸）。程式碼用 ``` 圍欄、公式用 $…$，都是模型輸出的通用寫法。
#
# `<nowiki>` 的內容是「照字面顯示」的文字，只脫標籤不刪內容——原本整塊被
# filter_wiki 吃掉，留下「用 代替，例如。」這種斷句。
_CODE_BLOCK_RE = re.compile(
    r'(?is)<(syntaxhighlight|source|pre)(?![\w.\-])([^>]*(?<!/))>(.{0,20000}?)</\1\s*>')
# `<nowiki/>` 是自閉合的（維基用它來斷開解析），不能當成開始標籤——
# 否則會一路配到後面某個 `</nowiki>`，把中間的正文整段當成逐字內容。
# 《統一資源標識符》就這樣被吃掉 2,436 字，裡面的模板與 ref 全部沒被處理。
_NOWIKI_RE = re.compile(r'(?is)<nowiki(?![\w.\-])[^>]*(?<!/)>(.{0,20000}?)</nowiki\s*>|<nowiki(?![\w.\-])[^>]*/>')
# `<code>` 是普通 HTML 標籤，不是內容不解析的擴充標籤——MediaWiki 在裡面照樣
# 展開模板與連結。把它當逐字區塊遮起來的話，`<code>{{le|SCHED_DEADLINE|…}}</code>`
# 會以模板原文的形式流進資料集（Linux內核、Python、Emacs 都中招）。
# 這裡只換成行內反引號保住「這是程式碼」的訊息，內容照常送去展開。
_CODE_TAG_RE = re.compile(r'(?is)<code(?![\w.\-])[^>]*(?<!/)>(.{0,4000}?)</code\s*>')
# 「這段內容裡有沒有 wiki／HTML 標記」。刻意不含括號與冒號——那些在一般
# 中文句子裡太常見，拿來判斷會把普通文字誤認成標記示範。
_MARKUP_CHAR_RE = re.compile(r"[{}\[\]|<>*#=]|''")
_LANG_ATTR_RE = re.compile(r'(?i)\blang\s*=\s*["\']?([\w+#.-]{1,20})')


def _wrap_inline_code(m):
    body = m.group(1).strip()
    # 空的 `<code></code>` 會變成夾在中文裡的一對反引號（`總是有愛在隔離`
    # 的「飄``移」），內容本來就沒有就整個丟掉。
    if not body or '`' in body:
        return body
    # `<code>` 裡常常就是尖括號本身（C 的 `<cmath>`、`<math.h>` 標頭檔）。
    # 不遮起來的話下游的 HTML 標籤規則會把它整個吃掉，只剩一對空反引號。
    # 行內程式碼也要遮，但**不能遮大括號與管線**：`<code>` 是普通 HTML 標籤，
    # MediaWiki 在裡面照樣展開模板，遮起來的話 `<code>{{le|SCHED_DEADLINE|…}}</code>`
    # 會以模板原文流進資料集。只遮「散文清理規則會動到」的字元就夠：
    #   ( )      空括號清理把 `f()` 變成 `f`
    #   '        粗體標記（Python 的 `''` 空字串）
    #   空白      空白正規化壓掉 `a  b` 的對齊
    for raw, mask in _INLINE_CODE_MASK:
        body = body.replace(raw, mask)
    # 方括號同理：`arr[i]` 的括號會被殘留標記清理吃掉。但內容裡真有 `[[` 時
    # 那是內部連結，遮起來就展不開了，只好讓連結優先——程式碼裡出現陣列索引
    # 遠比出現維基連結常見，所以預設保護方括號。
    if '[[' not in body:
        for raw, mask in _INLINE_BRACKET_MASK:
            body = body.replace(raw, mask)
    return f'`{body}`'


def _fence_code_blocks(s):
    """把程式碼區塊換成 ``` 圍欄，nowiki 只脫標籤"""
    def code(m):
        lang = _LANG_ATTR_RE.search(m.group(2) or '')
        # 只剝換行，不要剝遮罩空白——那是第一行的縮排。
        # `<pre>⏎  The quick brown fox` 的兩個空格被剝掉後，整段程式碼的
        # 首行就沒有縮排了（TeX 的排版範例、GTK 的 C 範例都中招）。
        body = m.group(3).strip('\n')
        # 內容是空的就連標記一起丟。只留一個孤立的 ``` 會讓圍欄配不成對，
        # 下游也看不出那裡本來有東西（`湘西土家族苗族自治州` 就出現孤立圍欄）。
        if not body.strip(_MASKED_WS).strip():
            return ''
        return f"\n```{lang.group(1) if lang else ''}\n{body}\n```\n"

    s = _CODE_BLOCK_RE.sub(code, s)
    def nowiki(m):
        body = (m.group(1) or '').strip()
        # nowiki 包住的如果是整張表格，那是資料不是「照字面顯示的範例」——
        # 維基用這招讓 `{{row numbers}}` 這類模板去加工表格內容
        # （`語系` 的語系列表、`世界棒球經典賽` 的歷屆賽事都是）。
        # 還原遮罩交回給 convert_tables，不然整張表會以原始語法流進資料集。
        plain = unmask_verbatim_braces(body)
        if '{|' in plain and '|}' in plain:
            return plain
        # nowiki 的內容是「照字面顯示」的文字，多半是在示範 wiki 標記本身。
        # 用行內反引號標起來，讀者看得出那是字面值，殘留標記的檢查也才能把它
        # 排除——否則 `<nowiki><div>Hi</div></nowiki>` 會被算成殘留 HTML。
        #
        # 但編者也常拿 nowiki 來**擋繁簡轉換**，包住的就是一般文字。那種情況
        # 加反引號等於憑空在句子中間插標記：`葡式蛋撻` 的「粵港澳稱葡`撻`」，
        # 而且只有繁體那一支有（簡體那一支沒包），繁簡兩版就此對不上。
        # 裡面根本沒有標記字元的，就是普通文字。
        if not body or '`' in body or not _MARKUP_CHAR_RE.search(body):
            return body
        return f'`{body}`'

    # `<code><nowiki>…</nowiki></code>` 是常見的巢狀寫法，兩層各包一次會變成
    # 雙層反引號，所以內容已經被標記過就不再包。
    # nowiki 要先處理：`<code><nowiki>…</nowiki></code>` 是常見寫法，
    # 反過來的話兩層各包一次會變成雙層反引號。
    s = _NOWIKI_RE.sub(nowiki, s)
    return _CODE_TAG_RE.sub(_wrap_inline_code, s)


def _mask_verbatim_braces(s):
    """
    把逐字區塊（LaTeX、程式碼）的大括號換成佔位字元。

    一度改成「數學／化學式裡照樣展開模板」，理由是以為 `{{=}}` 會殘留下來。
    那是誤判：`黎曼ζ函式`、`分數`、`恒生指數` 的原始碼一個 `{{=` 都沒有，
    輸出裡的 `{{` 全部來自 LaTeX 的巢狀括號。反過來，在數學式裡跑模板展開
    會主動毀掉公式——`\frac{{a}}{b}` 的 `{{a}}` 會被當成未知模板展開成空字串，
    變成 `\frac{b}`。逐字區塊就是逐字，一律遮起來。
    """
    def repl(m):
        body = m.group(3)
        for raw, mask in _VERBATIM_MASK:
            body = body.replace(raw, mask)
        if m.group(2).lower() in _CODE_TAGS:
            for raw, mask in _CODE_WS_MASK:
                body = body.replace(raw, mask)
        return m.group(1) + body + m.group(4)

    return _VERBATIM_RE.sub(repl, s)


def unmask_verbatim_braces(s):
    """還原逐字區塊的大括號與管線符號。資料集階段清完殘留標記後才呼叫。"""
    for raw, mask in _VERBATIM_MASK + _CODE_WS_MASK:
        s = s.replace(mask, raw)
    return s


def _unmask_verbatim_braces(s):
    return unmask_verbatim_braces(s)

# 配對出來卻大得離譜的區塊幾乎都是原始碼裡的括號不平衡造成的錯位，
# 不是真的模板。超過這個長度就當成沒配對處理，寧可留雜訊也不吃掉正文。
_MAX_TEMPLATE_SPAN = 20000


def _looks_like_template(body):
    """
    配對到的區塊是真的模板，還是括號錯位圈到的正文？

    模板的參數區塊裡不會有空行，正文會。條目原始碼裡的裸大括號很多
    （J 語言的程式碼範例、數學式），只靠長度擋不住，用空行數量判斷更準。
    """
    return len(body) <= _MAX_TEMPLATE_SPAN and body.count('\n\n') <= 1


def _skip_param_block(s, i):
    """
    從未閉合的 `{{模板名` 開始，跳過它後面整段參數。

    沒有 `}}` 就無從得知模板到哪裡結束，只丟掉模板名所在的這一行，剩下的
    `|參數 = 值` 交給 drop_orphan_params。

    試過用「吃到第一個空行」「吃到第一行沒有 `=` 的」當結束界線，兩種都會
    在別的條目上失控——《聖雄甘地》被吃掉 21,000 字、《法拉第電磁感應定律》
    被吃掉 4,700 字。殘留幾行參數是可以接受的，刪掉正文不行，所以界線收到
    一行為止。
    """
    line_end = s.find('\n', i)
    return len(s) if line_end == -1 else line_end


# 已經轉換成 `｜` 分隔的表格列；模板外殼（`{{名稱|` 與收尾的 `}}`）
_CONVERTED_ROWS_RE = re.compile(r'(?m)^[^\n]*｜[^\n]*$')
# 成段的正文：有句讀、夠長的行
_PROSE_LINE_RE = re.compile(r'(?m)^[^\n]{40,}[。！？][^\n]*$')


def _looks_like_wrapped_content(body):
    """模板外殼裡包的是內容（表格列、成段正文或程式碼），不是純參數設定"""
    if len(_CONVERTED_ROWS_RE.findall(body)) >= 3:
        return True
    if _WRAPPED_FENCE_RE.search(body):
        return True
    return len(_PROSE_LINE_RE.findall(body)) >= 2
_TEMPLATE_SHELL_RE = re.compile(r'^\{\{[^|{}\n]{0,60}\|?|\}\}$')


def remove_template_blocks(s):
    """
    用大括號配對移除展開後仍殘留的模板，取代原本受限於單行的正則迴圈。

    `{{...}}` 可以跨行也可以巢狀，逐行的 `{{(.*?)}}` 只吃得到最內層又不跨行的
    那一種，於是《路德维希·范·贝多芬》的註腳正文、《公历》的計算說明會漏出來，
    尾巴還帶著一個孤零零的 `}}`。

    配對成功的整段丟掉（那是沒能展開的模板）。配不到對的要看位置：
    自成一行的（`{{Infobox economy` 少了收尾，《中华人民共和国经济》真的就這樣寫）
    連同那一行的模板名一起丟掉，剩下的參數行由 drop_orphan_params 收尾；
    夾在句子中間的只丟大括號，後面的文字仍是正文。

    掃描用 str.find 在標記之間跳躍而不是逐字元，理由同 _match_span。
    """
    if '{{' not in s and '}}' not in s:
        return s
    out = []
    pos = 0
    i = 0
    n = len(s)
    while i < n:
        nxt_open = s.find('{{', i)
        nxt_close = s.find('}}', i)
        if nxt_open == -1 and nxt_close == -1:
            break
        # 落單的 `}}`（前面沒有開括號）直接丟掉
        if nxt_open == -1 or (nxt_close != -1 and nxt_close < nxt_open):
            out.append(s[pos:nxt_close])
            pos = i = nxt_close + 2
            continue

        end = _match_span(s, nxt_open, '{{', '}}', _MAX_TEMPLATE_SPAN)
        if end != -1 and _looks_like_template(s[nxt_open:end]):
            body = s[nxt_open:end]
            # 維基會把大段內容塞進模板參數讓模板去加工，整段丟掉等於刪內容：
            #   {{row numbers|<nowiki>{| … |}</nowiki>}}  替表格列編號
            #   {{Math proof|proof=…}}                    折疊整段證明
            # 前者裡面是轉換好的表格列，後者是成段的正文與公式。兩種都要保住
            # 內容、只丟外殼（`群` 的結合律證明、`語系` 的 25 個語系都靠這條）。
            if _looks_like_wrapped_content(body):
                out.append(s[pos:nxt_open])
                out.append(_TEMPLATE_SHELL_RE.sub('', body))
                pos = i = end
                continue
            out.append(s[pos:nxt_open])
            pos = i = end
            continue

        # 沒配對（或圈到的範圍不像模板）：自成一行的連模板名一起丟，
        # 夾在句子中間的只丟掉大括號本身
        line_start = s.rfind('\n', 0, nxt_open) + 1
        if not s[line_start:nxt_open].strip():
            out.append(s[pos:nxt_open])
            pos = i = _skip_param_block(s, nxt_open)
        else:
            out.append(s[pos:nxt_open])
            pos = i = nxt_open + 2
    out.append(s[pos:])
    return ''.join(out)


# 模板參數行：`| country = 中華人民共和國`。表格在這一步之前已經轉成 `｜`
# 分隔的文字了，所以還留著半形 `|` 前綴的就是沒收尾的模板漏出來的參數。
_PARAM_LINE_RE = re.compile(r'(?m)^[ \t]*\|[ \t]*[A-Za-z_][\w\- ]{1,39}[ \t]*=[^\n]*$')


def drop_orphan_params(s):
    """丟掉未閉合模板漏出來的參數行（要連續兩行以上才算，避免誤傷表格）"""
    if '|' not in s:
        return s
    spans = [m.span() for m in _PARAM_LINE_RE.finditer(s)]
    if not spans:
        return s
    out, keep_from, run = [], 0, []
    for start, end in spans + [(None, None)]:
        if run and (start is None or s[run[-1][1]:start].strip()):
            if len(run) >= 2:                     # 連續參數行 → 是模板殘骸
                out.append(s[keep_from:run[0][0]])
                keep_from = run[-1][1]
            run = []
        if start is not None:
            run.append((start, end))
    out.append(s[keep_from:])
    return ''.join(out)


# 判斷模板是否自成一行時的搜尋窗。維基的行再長也不會超過這個量級。
_LINE_WINDOW = 4096

# 化學式（mhchem 的 <chem>／<ce>）也是 LaTeX，跟數學式一樣要標分隔符。
# 只認 <math> 的話，`植物` 的光合作用方程式、`催化劑` 的反應式、`氚` 的衰變式
# 都會以裸 LaTeX 的形式散在行文裡，看不出是公式。
# 標籤名後面不能接 `.`／`-`／字母：C 的標頭檔 `<math.h>`、`<chem.h>` 在程式
# 相關條目裡很常見，用 `\b` 會把它當成公式的開啟標籤，一路配到後面真正的
# `</math>`，把中間整段（含表格）都吞進逐字區塊——`C++ Technical Report 1`
# 的 23 個數學函式表就這樣沒被轉換。
_MATH_TAG_RE = re.compile(
    r'(?is)<(math|chem|ce)(?![\w.\-])[^>]*(?<!/)>(.{0,20000}?)</\1\s*>'
    r'|<(?:math|chem|ce)(?![\w.\-])[^>]*/>')


# 公式保留 LaTeX 原始碼，並補上標準分隔符：行內 `$…$`、獨佔一行 `$$…$$`。
#
# 只剝標籤留裸 LaTeX 有兩個毛病。一是公式的起訖消失了，混在行文裡分不出來
# （`x = \mathrm{Re}\{z\} \,為實部`）；二是獨佔一行的公式會變成「沒有中文、
# 沒有句讀、含大括號」的行，正好命中殘骸行規則被整行刪掉——《歐拉公式》的
# `e^{ix} = \cos x + i\sin x` 就是這樣整條消失的。
# `$…$` 是 LaTeX 與模型輸出的通用寫法，補上之後兩個問題一起解決。
_MATH_DISPLAY_ATTR_RE = re.compile(r'(?i)display\s*=\s*["\']?block')


def _keep_math(match):
    """保留公式的 LaTeX 原始碼，去掉標籤，補上 $ 或 $$ 分隔符"""
    body = re.sub(r'\s+', ' ', match.group(2) or '').strip()
    if not body:
        return ''
    text = match.string
    line_start = text.rfind('\n', 0, match.start()) + 1
    line_end = text.find('\n', match.end())
    if line_end == -1:
        line_end = len(text)
    # 維基用行首的 `:` 縮排來表示獨立公式；`display=block` 也是同一個意思
    standalone = (not text[line_start:match.start()].strip(' \t:*#')
                  and not text[match.end():line_end].strip())
    block = standalone or _MATH_DISPLAY_ATTR_RE.search(
        text[match.start():match.end()])
    if block:
        return f'$${body}$$'
    return f'${body}$'


# 句子收尾的標點。行尾是這些字元就代表句子結束了，下一行是新的一段。
_LINE_END_PUNCT = frozenset('。！？；：…，、」』）】》〉”’"\'.!?;:,)]}>')
# 行首出現這些就不是續行，而是另一種結構（清單、標題、表格、圍欄、縮排）
_LINE_START_STRUCT = re.compile(r'^\s*(?:[*#:;=|!]|```|\{\||\|\}|｜|\{\{)')
_NEWLINE_RUN_RE = re.compile(r'\n+')


def _join_wrapped_prose(s):
    """把編者折行折斷的句子接回同一行

    wikitext 的**單一**換行不會分段，只有空行才會。但我們是逐行處理再用空行
    串起來，於是編者為了原始碼好讀而折的行，到輸出就變成兩個段落，句子攔腰
    斷開——實測平均每五篇條目就有一處：

        …然而經濟學家卻一面倒的認為自由貿易才是增進社會整體利益的方式
        。依據美國經濟學會的調查…

    只在「上一行沒有以任何句讀收尾」時才接，所以編者刻意分開的兩個完整句子
    不會被併起來。這一步必須跑在 `\n+ → \n` 之前——空行一旦被收掉，就再也
    分不出「折行」與「分段」了。
    """
    out = []
    in_code = False
    for line in s.split('\n'):
        cur = line.rstrip()
        # 圍欄**內部**逐字保留：不只是標記那兩行，中間每一行都不能接。
        # 少了這個狀態機，`def f():` 會跟 `    return 1` 併成一行，程式碼全毀。
        if in_code:
            out.append(cur)
            if cur.strip() == '```':
                in_code = False
            continue
        if cur.lstrip().startswith('```'):
            in_code = True
            out.append(cur)
            continue
        nxt = cur.lstrip()
        if (out and out[-1] and cur and nxt
                and out[-1][-1] not in _LINE_END_PUNCT
                and not out[-1].endswith('>')
                and not _LINE_START_STRUCT.match(out[-1])
                and not _LINE_START_STRUCT.match(cur)
                and '｜' not in out[-1] and '｜' not in cur
                and '```' not in out[-1]):
            sep = ' ' if out[-1][-1].isascii() and out[-1][-1] not in '-–—' else ''
            out[-1] = out[-1] + sep + nxt
            continue
        out.append(cur)
    return '\n'.join(out)


def _collapse_blank_lines(s):
    """收掉多餘空行，但圍欄裡的空行是程式碼的一部分

    原本是 `re.sub('\\n+', '\\n', s)`，一視同仁地把連續換行壓成一個，套到程式碼
    上就是把區塊之間的空行全部抹掉——實測 27% 的程式碼區塊只差在這個
    （`C♯` 的 `string status = string.Empty;` 與 `public string Status` 之間
    本來有一個空行）。空行在程式碼裡是可讀性的一部分，Python 的定義之間更是慣例。
    """
    if '```' not in s:
        return _NEWLINE_RUN_RE.sub('\n', s)
    out, in_code = [], False
    for line in s.split('\n'):
        stripped = line.strip()
        if in_code:
            out.append(line)
            if stripped == '```':
                in_code = False
        elif stripped.startswith('```'):
            in_code = True
            out.append(line)
        elif stripped:
            out.append(line)
        # 圍欄外的空行直接丟掉，等同原本 `\n+ → \n` 的效果
    return '\n'.join(out)


# gap 裡這些東西到最後都會消失，判斷公式是否相鄰時要先扣掉
_VANISHING_GAP_RE = re.compile(r"(?s)''+|<!--.*?-->|&nbsp;|[ \t]+$")


def _keep_math_all(s):
    """逐條轉換公式，順手拆開黏在一起的分隔符

    兩個行內公式緊鄰時，輸出會產生貼在一起的 `$$`，讀起來變成行間公式的分隔
    符，兩條公式就此合成一條壞掉的（`楊輝三角形` 的 `$n = F_{2i+2}…,\\,$$k =
    F_{2i}…,\\,$` 是實例）。

    判斷條件必須看**已經產生的輸出**，不能看原始碼裡是不是 `</math><math>`：
    中間可能隔著 `''`、`{{nbsp}}` 這類會被清掉的標記，原始碼上看不出相鄰。
    """
    out, last, tail = [], 0, ''
    for m in _MATH_TAG_RE.finditer(s):
        gap = s[last:m.start()]
        if gap:
            out.append(gap)
            # gap 整段都是「最後會消失的標記」（`''`、註解、&nbsp;）時，不要讓它
            # 蓋掉 tail——那樣就看不出前一條公式其實緊貼著下一條。
            if _VANISHING_GAP_RE.sub('', gap).strip():
                tail = gap
        piece = _keep_math(m)
        # 判斷「相鄰」要看 gap **清乾淨之後**還剩什麼。中間夾著 `''`（粗體標記）、
        # `<!--註解-->`、`&nbsp;` 時，原字串上看起來不相鄰，但那些標記在下游會被
        # 清掉，最後還是黏成 `$$`（`<math>a</math>''<math>b</math>` → `$a$$b$`）。
        if piece.startswith('$') and tail.endswith('$'):
            piece = ' ' + piece
        if piece:
            out.append(piece)
            tail = piece
        last = m.end()
    out.append(s[last:])
    return ''.join(out)

# 解析器函式 {{#expr:…}} 的算術求值。字元集限死成數字與四則運算，
# 不符合就原樣放棄，不讓任意運算式進到 eval。
_SAFE_EXPR_RE = re.compile(r'[\d\s+\-*/().]{1,200}')


def _eval_expr(expr):
    expr = expr.strip()
    if not _SAFE_EXPR_RE.fullmatch(expr):
        return ''
    try:
        value = eval(expr, {'__builtins__': {}}, {})       # noqa: S307 - 字元集已限制
    except Exception:
        return ''
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


# 展開後的文本相對原文的長度上限。模板互相引用時會指數級膨脹，這是煞車。
_MAX_EXPANSION_RATIO = 4


# 中日韓字元。折行處兩側只要有一邊是中文就不補空格，補了反而在句子中間留白。
_CJK_CLASS = '\u2e80-\u9fff\uf900-\ufaff\ufe30-\ufe4f\uff00-\uffef'
_WRAP_NEXT_CJK_RE = re.compile(f'[ \t]*\n[ \t]*(?=[{_CJK_CLASS}])')
_WRAP_PREV_CJK_RE = re.compile(f'(?<=[{_CJK_CLASS}])[ \t]*\n[ \t]*')
_WRAP_RE = re.compile(r'[ \t]*\n[ \t]*')


def _join_wrapped_lines(s):
    """把模板參數裡的折行接回同一行"""
    if '\n' not in s:
        return s
    s = _WRAP_NEXT_CJK_RE.sub('', s)
    s = _WRAP_PREV_CJK_RE.sub('', s)
    return _WRAP_RE.sub(' ', s)


# 模板後面緊接著繫詞＝它渲染的是主語
_LEADS_SENTENCE_RE = re.compile(r'^[是為爲]')
# 「名稱」類參數，依可信度排序
_SUBJECT_KEYS = ('zh', '中文名', '中文名稱', '名稱', '名称', 'name', 'title',
                 'label', 'binomial', 'taxon', '學名', '学名')


def _subject_from_params(match):
    """從模板參數裡取出它渲染出來的名稱（給句首主語用）"""
    pos, named = _split_args(match.group(2))
    for key in _SUBJECT_KEYS:
        value = (named.get(key) or '').strip()
        if value and not _LAYOUT_KEY_RE.match(value):
            return value
    return _fallback_visible(pos, named)


def expand_inline_templates(text, passes=8):
    """由內而外展開模板（多跑幾輪以處理巢狀）"""
    limit = max(len(text) * _MAX_EXPANSION_RATIO, 100000)
    for _ in range(passes):
        current = text

        def repl(m):
            # 找所在行的頭尾要設搜尋窗。`rfind('\n', 0, m.start())` 是從模板位置
            # 一路往回掃到檔頭——O(位置)，而一頁有數千個模板、要跑 8 輪，整體
            # 變成 O(n²)。《Pokémon GO》單篇因此卡住一個 worker 超過 9 分鐘。
            # 超出窗就代表這一行很長，前面必定有內容，本來就不是 standalone。
            lo = max(0, m.start() - _LINE_WINDOW)
            hi = min(len(current), m.end() + _LINE_WINDOW)
            line_start = current.rfind('\n', lo, m.start()) + 1
            if line_start == 0 and lo > 0:
                return _join_wrapped_lines(_expand_template(m, standalone=False))
            line_end = current.find('\n', m.end(), hi)
            if line_end == -1:
                line_end = hi
            before = current[line_start:m.start()].strip()
            after = current[m.end():line_end].strip()
            standalone = not before and not after
            # 前一行以冒號收尾＝作者寫了引導語在指向這個模板，那它就是內容
            # 而不是導航元素。不認這個訊號的話，`{{Column|2|*[[非洲開發銀行]]…}}`
            # 這類包住清單／表格的模板會整個被丟掉，留下
            # 「中華人民共和國是下列國際組織的成員：」這種有頭沒尾的句子
            # （實測 1.33% 的條目中招）。
            introduced = False
            if standalone:
                prev_end = current.rfind('\n', lo, line_start - 1) + 1 if line_start else 0
                prev = current[prev_end:max(prev_end, line_start - 1)].strip()
                # 引導語後面常常緊接著來源標註（`本屬包括以下物種：<ref>…</ref>`），
                # 那一行就不是以冒號收尾了。先把尾巴的 ref 拿掉再判斷——不然
                # 生物分類條目的物種清單全部接不到（實測是剩餘案例的大宗）。
                prev = _strip_trailing_refs(prev).strip()
                introduced = prev.endswith(('：', ':'))
            expanded = _expand_template(m, standalone=standalone,
                                        introduced=introduced)
            # 模板後面緊接著「是／為」時，它渲染出來的就是這句話的主語。
            # 展開成空字串會讓條目第一句變成「，是一種化學元素」——
            # `{{臺灣植物|zh=杜鵑花屬}}是杜鵑花科的一個屬` 就是這樣掉的。
            # 用位置判斷而不是列舉模板名：分類群、地區用詞、各種名稱模板
            # 都套用同一條規則。
            if not expanded.strip() and _LEADS_SENTENCE_RE.match(
                    current[m.end():m.end() + 2]):
                expanded = _subject_from_params(m)
            # 行內模板的展開結果不該帶換行：原始碼裡的換行只是編者把長參數折行
            # （`{{lang|fr|Annales de chimie et de⏎physique}}`），MediaWiki 渲染成
            # 同一段。照原樣輸出的話一個句子會被切成兩段，第二段以 `》（1789年…`
            # 這種標點開頭（`物理化學`、`Su-25攻擊機` 都是）。
            # standalone 的展開結果本來就可能是多列清單，不能碰。
            # 含圍欄的也不能碰：`{{efn|…<syntaxhighlight>…}}` 寫在句子中間，
            # 接合換行會把整段程式碼壓成一行。
            if standalone or '```' in expanded:
                return expanded
            return _join_wrapped_lines(expanded)

        new = _TEMPLATE_RE.sub(repl, current)
        if new == current:
            break
        # 互相引用的模板會指數級膨脹，超過上限就停在上一輪的結果
        if len(new) > limit:
            break
        text = new
    return text



# 變體標記本體不含大括號，巢狀靠重跑幾輪處理
_VARIANT_BLOCK_RE = re.compile(r'-\{([^{}]*)\}-', re.DOTALL)
_VARIANT_PASSES = 8


def resolve_variant_markers(s):
    """
    處理 -{zh-tw:…;zh-cn:…}- 語言變體標記。

    這些標記是維基編者手寫的各地用詞對照（例如 zh-tw:賽局理論／zh-cn:博弈论、
    zh-cn:反馈／zh-tw:回報），品質遠高於任何自動轉換。

    中間層是 tw/cn 共用的，這一階段不能選邊，所以把兩個變體都保留下來，
    用私有區字元包起來（正文絕不會出現這些字元），交給下游依語言挑選。

    寫成模組層級函式是為了讓圖片圖說也走同一套規則——圖片擷取原本自己寫了
    一份較弱的清理邏輯，漏掉 {{PAGENAMEBASE}}、{{Tsl}} 等模板。
    """
    # 一次掃過整份文字換掉所有標記，再重跑幾輪處理巢狀。
    #
    # 原本是「找到一個就用切片重建整篇」，k 個標記就重建 k 次，成本 O(k·n)——
    # 實測 2,000 個變體區塊（20 萬字）要 281 ms，而條目裡的變體標記動輒上百個。
    if '-{' not in s:
        return s
    for _ in range(_VARIANT_PASSES):
        new_s = _VARIANT_BLOCK_RE.sub(lambda m: _resolve_variant(m.group(1)), s)
        if new_s == s:
            break
        s = new_s
    return s


def _resolve_variant(body):
    """把變體標記內容轉成保留雙版本的形式"""
    # 去掉開頭的旗標（例如 -{H|zh-cn:…}- 的 H|）
    body = re.sub(r'^\s*[A-Za-z\-]{1,3}\s*\|', '', body.strip())

    pairs = {}
    for part in body.split(';'):
        m = re.match(r'\s*zh-(hans|hant|cn|tw|hk|mo|sg|my)\s*:\s*(.*)', part, re.DOTALL)
        if m:
            pairs.setdefault(m.group(1), m.group(2).strip())

    if not pairs:
        # 沒有變體資訊，只是用來抑制轉換的 -{文字}-，直接取內容
        return body

    def pick(order):
        for key in order:
            if pairs.get(key):
                return pairs[key]
        return ''

    tw = pick(('tw', 'hant', 'hk', 'mo'))
    cn = pick(('cn', 'hans', 'sg', 'my'))

    if not tw and not cn:
        return ''
    if not tw:
        return cn
    if not cn or tw == cn:
        return tw
    # 內容跨行（圍欄、清單、表格）時，標記要自己獨佔一行。黏在一起的話
    # 下游所有「行首是不是 ```」的狀態機都認不出來——`\ue000```js` 不算圍欄開頭，
    # 於是整段程式碼被當成一般段落，每行之間被撐開成空行（`JavaScript` 繁簡
    # 兩版因此差了 7 行）。行內的短詞維持原樣，不要平白多出換行。
    if '\n' in tw or '\n' in cn:
        return (f'\n{VARIANT_OPEN}\n{tw}\n{VARIANT_SEP}\n{cn}\n{VARIANT_CLOSE}\n')
    return VARIANT_OPEN + tw + VARIANT_SEP + cn + VARIANT_CLOSE


class WIKIParse(object):

    KEYWORDS = [
        'Template', 'Category', 'Wikipedia',
        'File', 'Topic', 'Portal',
        'MediaWiki', '模块', 'Draft', 'Help'
    ]
    
    def __init__(self, input_file, markdown=False):
        try:
            bz2_file = bz2.open(input_file)
            self.wiki_content = extract_pages(bz2_file)
        except Exception as e:
            raise RuntimeError(e)

        self.markdown = markdown
        self.nl = '\n\n' if markdown else '\n'

    def __is_not_word(self, word):
        word_items = word.split(':')
        if len(word_items) > 1 and \
                word_items[0] in self.KEYWORDS:
            return True
        return False

    def __is_redirect(self, text):
        """重定向頁：`#REDIRECT [[目標]]`

        原本只看「開頭是不是 `#`」，但 `#` 在 wikitext 是**有序清單**的行首標記，
        於是任何以編號清單開頭的正常條目都被當成重定向整篇丟棄。要求完整形態
        （關鍵字 + 連結）才算。
        """
        return _REDIRECT_PAGE_RE.match(text.lstrip()) is not None

    def __is_disambiguation(self, text):
        """
        消歧義頁：維基自己掛的標記模板就是權威答案。

        原本只在資料集階段用「開頭是『XXX可以指：』」這種措辭去猜，猜不準：
        全量掃 50 萬頁，帶標記模板的有 13,405 頁（2.68%），措辭啟發式只抓得到
        其中一小部分，還會誤判（「町」「南瓜」是正常條目，開頭卻也寫著
        「可以指：」）。改讀模板之後兩邊都對，而且繁簡兩版必然一致——
        判斷發生在語言中立的中間層，不受轉換後字數變化影響。

        `set index`／`shipindex` 這類索引條目不算在內，它們有實質內容。
        """
        return _DISAMBIG_TEMPLATE_RE.search(text) is not None

    def __clean_synonym(self, s):
        """處理語言變體標記（實作見模組層級的 resolve_variant_markers）"""
        return resolve_variant_markers(s)

    def __clean_template(self, s):
        # 处理 {{Le|文本1|文本2|文本3}} 类型模板，优先保留第三个参数(通常是中文解释)，否则保留第一个参数
        def le_replacement(match):
            text1 = match.group(1)
            text3 = match.group(3) if match.group(3) else None
            return text3 if text3 else text1

        # 先处理Le模板
        le_template = r'{{Le\|(.*?)\|(.*?)(?:\|(.*?))?}}'
        s = re.sub(le_template, le_replacement, s)

        # 處理所有 link-xx 類模板，保留第二個參數（顯示文字）
        def link_lang_replacement(match):
            # {{link-de|顯示文字|原文}}，保留顯示文字
            return match.group(1)
        s = re.sub(r'{{link-[a-z]{2}\|([^|{}]+)\|([^|{}]+)}}', link_lang_replacement, s)

        # 其餘沒能展開的模板：用大括號配對移除，跨行與巢狀都吃得到
        return remove_template_blocks(s)

    def __clean_wiki_links(self, s):
        """自定义的链接清理函数，保留链接文本"""
        # 处理 [[链接|显示文本]] 格式，保留显示文本
        s = re.sub(r'\[\[([^\|\]]+)\|([^\]]+)\]\]', r'\2', s) #繁體
        
        # 处理 [[链接]] 格式，保留链接文本
        s = re.sub(r'\[\[([^\]]+)\]\]', r'\1', s)
        
        # 处理外部链接 [http://... 显示文本] 格式
        s = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', s)
        
        # 移除单独的外部链接 [http://...]
        s = re.sub(r'\[https?://[^\s\]]+\]', '', s)
        
        # 移除其他wiki标记但不使用filter_wiki
        s = re.sub(r"'''([^']+)'''", r'\1', s)  # 粗体
        s = re.sub(r"''([^']+)''", r'\1', s)    # 斜体
        s = _REF_BLOCK_RE.sub('', s)  # 移除引用（此方法目前未被呼叫，保留備查）
        s = re.sub(r'<ref[^>]*/?>', '', s)  # 移除单独的ref标签
        s = re.sub(r'<!--.*?-->', '', s, flags=re.DOTALL)  # 移除注释
        s = re.sub(r'<[^>]+>', '', s)  # 移除其他HTML标签
        
        return s

    def __clean(self, s, sink=None, page_title=''):
        # 註解要最先拿掉。編者常在註解裡放半截的標記
        # （《聖雄甘地》有 `<!-- {Critique of political economy}} -->`），
        # 那個多出來的 `}}` 會讓後面的大括號配對錯位，把 33,000 字的正文
        # 當成一個模板整段吃掉。
        s = strip_comments(s)
        # 逐字遮罩要在資訊框抽取**之前**。抽取靠 `{{`／`}}` 配對找出資訊框的
        # 範圍，而 LaTeX 的巢狀括號會湊出假的 `}}`：`黄金分割率` 的
        # `| 連分數=<math>1 + \cfrac{1}{1 + \cfrac{1}{1 + \ddots}}</math>` 尾端
        # 那個 `}}` 收掉的是兩層單括號，配對卻當成模板收尾，資訊框從那裡被
        # 切斷，後面的欄位連同公式一起消失。
        s = _mask_verbatim_braces(s)
        # 側邊資訊框整塊丟棄前，先把裡面的事實抽出來接到文末
        s += extract_infobox_facts(s, page_title)
        # 遮罩之後才換圍欄：圍欄裡的內容已經受保護
        s = _fence_code_blocks(s)
        s = self.__clean_synonym(s)
        # 先把帶正文的行內模板展開，再交給下面的規則移除剩餘模板，
        # 否則 {{lang}}、{{flag}}、{{bd}} 這類模板的可見內容會一起被刪掉
        s = expand_inline_templates(s)
        # 模板展開完要再挑一次變體：模板本體裡也寫著 -{zh-cn:…;zh-tw:…}-，
        # 只在展開前挑的話那些標記會原封不動流進正文——`桃花源` 的
        # 「桃花源記旁證-{zh-cn:》; zh-tw:〉; zh-hk:》;}-」出現了 9 次。
        s = self.__clean_synonym(s)
        # 表格轉成文字列（要在下面的表格移除規則之前做，否則資料會整塊消失）
        s = convert_tables(s)
        s = self.__clean_template(s)

        # convert_tables 沒能處理的殘餘表格。跨度要設上限：`{|` 若沒有對應的
        # `|}`，非貪婪比對會一路找到很遠的地方，把中間的正文一起刪掉。
        s = re.sub(r':*\{\|[\s\S]{0,20000}?\|\}', '', s)
        # 表格語法已經處理完，還留著 `|參數 = 值` 的就是沒收尾的模板漏出來的
        s = drop_orphan_params(s)
        s = remove_file_links(s, sink)
        s = convert_galleries(s, sink)
        # EasyTimeline／樂譜／地圖等區塊裡是繪圖指令不是內容
        # （山東省條目原本會洩漏 `id:barra value:rgb(0.7,0.9,0.7)` 這種圖表定義）
        s = re.sub(r'(?is)<(timeline|score|mapframe|maplink|graph|imagemap)\b[^>]*>.*?</\1>', '', s)
        s = re.sub(r'(?is)<(timeline|score|mapframe|maplink|graph|imagemap)\b[^>]*/>', '', s)
        # <math> 裡是公式，是內容不是標記。交給 filter_wiki 會整段刪掉，
        # 留下「其滿足：」「任意矩陣，是斜對稱矩陣。」這種斷句（實測 1.375%
        # 的條目中招）。這裡先脫掉標籤保留 LaTeX 原始碼，句子才完整。
        # （大括號已由 _mask_verbatim_braces 遮起來，filter_wiki 不會誤判。）
        s = _keep_math_all(s)
        # <poem> 裡是詩詞正文，只脫標籤不刪內容
        # （原本把 `<poem style=…>` 整塊刪掉，古典詩詞條目會整段消失）
        s = re.sub(r'(?is)</?poem\b[^>]*>', '\n', s)
        # 行為開關（__NOTOC__、__TOC__、__NOEDITSECTION__…）不是文字
        s = _MAGIC_WORD_RE.sub('', s)
        # 分類連結要自己清掉，不能等 filter_wiki——gensim 只認大寫 `Category:`，
        # 小寫的 `[[category:史学|Z浙]]` 會落到一般連結規則，把**排序鍵**當成
        # 顯示文字留在正文結尾（《浙東史學》結尾就多了一個 `Z浙`）。
        # 實測 0.93% 的頁面有小寫寫法。
        s = _CATEGORY_LINK_RE.sub('', s)
        s = re.sub(r'(.){{([^{}\n]*?\|[^{}\n]*?)}}',
                   r'\1[[\2]]', s)  # 修正: 原為 \\1 \\2 (字面反斜線), 導致正文殘留 \1\2
        
        # filter_wiki 會把「< 到下一個 >」當成 HTML 標籤整段刪掉，正文裡的
        # 裸 `<` 因此會吃掉後面一大片內容——「米诺克斯」的圖說寫著
        # `<1/3 尺度`，結果整篇後半（6 個章節）消失；數學條目的 `若 x < 0`
        # 也是同樣情形。先把不是標籤開頭的 `<` 換成佔位字元，過完再換回來。
        # 註腳裡的程式碼與公式要先撈出來。filter_wiki 會把整個 `<ref>…</ref>`
        # 刪掉，而編者會把「這個方法的定義」整段放進註腳——`Smalltalk` 條目的
        # 五段集合類別實作全在 `<ref>` 裡，一刪就整批消失。判準不是標籤名，
        # 是「裡面有沒有圍欄或公式」：真正的引用不會有。跟 `{{efn}}` 同源。
        s = _REF_BLOCK_RE.sub(_keep_ref_verbatim, s)
        s = _BARE_LT_RE.sub(_BARE_LT, s)
        s = filter_wiki(s)
        s = s.replace(_BARE_LT, '<')
        # 逐字區塊的大括號**不在這裡還原**，一路遮到資料集階段
        # strip_leftover_markup 跑完為止（見 md_to_dataset._build_one）。
        #
        # 原本在這裡還原，之後 LaTeX 的大括號就跟殘留的模板括號分不出來，
        # `remove_leftover_templates` 的孤兒括號規則（`\{\{+|\}\}+`）會把
        # 巢狀括號成串刪掉：
        #     原始碼 \overset{\underset{\mathrm{t=p^{-s}}}{}}{=}
        #     輸出   \overset{\underset{\mathrm{t=p^{-s{{=}
        # LaTeX 的巢狀括號本來就大量產生 `}}`，全站約兩萬篇有數學式的條目都會
        # 被改壞，而且規則是「刪除」，任何殘留檢查都看不到。
        
        # 內容被移除後只剩符號的空清單項要丟掉，但一定要錨在行首：
        # 原本寫 `\* *\n`，任何**行尾**的星號都會命中，連同換行一起刪掉，
        # 把下一列黏上來——「咖啡因」的含量表用 `135*` 標註腳，結果四列黏成
        # 一列：`咖啡，沖濾｜240 mL｜135咖啡，脫咖啡因｜240 mL｜5…`。
        s = re.sub(r'(?m)^[ \t]*\*[ \t]*$\n?', '', s)
        s = re.sub(r"'{2,}", '', s)
        # 折行接合要在空行還在的時候做。下一行的 `\n+ → \n` 會把空行一起收掉，
        # 到那時就分不出「編者折行」與「真的分段」了。
        s = _join_wrapped_prose(s)
        s = _collapse_blank_lines(s)
        s = _normalize_list_prefixes(s)
        # 固定字面字串不需要進 regex 引擎；str.replace 與此規則逐字等價。
        s = s.replace('\n==', '\n\n==')
        # 折行折在句號前面時（`…的方式⏎。依據美國經濟學會…`），句號要接回上一句。
        # 原本換成 `。\n` 只做了一半：句號歸位了，但後面那半句仍然另起一行，
        # 到下游就變成獨立的一段。整行接回去才對——wikitext 的單一換行不分段。
        #
        # 但不能黏到圍欄標記後面。`<pre>…</pre>。` 的句號會接到收尾的 ``` 上，
        # 變成 ```` ```。 ````，圍欄就配不成對了（`全球資訊網` 的 HTML 範例因此
        # 整塊失去圍欄標記）。
        s = re.sub(r'(?<!```)\n。', '。', s)
        return s

    def __clean_surrogates(self, text):
        """清理文本中的代理對字符"""
        if isinstance(text, str):
            # 沒命中時 Pattern.sub 直接回傳原 str，不必每篇先複製成
            # UTF-8 bytes 再 decode；命中時與 errors='ignore' 一樣移除 surrogate。
            return _SURROGATE_RE.sub('', text)
        return text

    def __fresh(self, word, text):
        def update(cn):
            return str(int(cn) + 1)

        def form_line(catalog, title, level):
            if catalog:
                level = len(catalog) - catalog.count('0') + 1
                catalog = [c for c in catalog if c != '0']
                line = '.'.join(catalog) + ' ' + title
            else:
                line = title + self.nl

            if self.markdown:
                line = '#' * level + ' ' + line
            return line

        fresh_text = form_line(None, word, 1)
        prev_item_line = False
        # counters[0] 對應 `==`（層級 2），依此類推到 `======`（層級 6）
        counters = ['0'] * 5
        # 程式碼圍欄裡是逐字內容，一行就是一行。維基的一般行要用空行隔開
        # （`self.nl` 是 `\n\n`），但同一條規則套到程式碼上會把每一行都撐開成
        # 一個段落——`Trie`、`記憶體洩漏` 的 C 範例整段變成隔行的
        # `#include <stdio.h>` ⏎⏎ `#include <stdlib.h>`，縮排與區塊結構全毀。
        #
        # 只認**配得成對**的圍欄。維基原始碼裡本來就躺著編者留下的落單 ```
        # （`湘西土家族苗族自治州` 正文中間就有一個），把它當成開頭的話，
        # 後面整篇正文都會被當成程式碼，段落分隔全部消失。
        lines = text.split('\n')
        code_lines = set()
        code_close = set()
        opener = None
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if opener is None:
                if stripped.startswith('```'):
                    opener = idx
            elif stripped == '```':
                code_lines.update(range(opener, idx + 1))
                code_close.add(idx)
                opener = None
        for index, line in enumerate(lines):
            item_line = False
            # 只有圖片位置標記的行：用單一換行，不要打斷前後的清單。
            # 補空行的話，剝掉標記後 gallery 的清單項之間會多出空行。
            if _IMAGE_MARK_LINE_RE.fullmatch(line + '\n'):
                fresh_text += line + '\n'
                continue
            if index in code_lines:
                fresh_text += line + '\n'
                # 收尾的圍欄後面補一個空行跟下一段隔開。要認「是不是收尾那一行」，
                # 不能只看內容是不是 ```——沒有語言標註時開頭那行也長這樣，
                # 於是圍欄一開始就多出一個空行（`<pre>` 的區塊全部中招）。
                if index in code_close:
                    fresh_text += '\n'
                continue
            head = _WIKI_HEADING_RE.match(line)
            trailing = head.group(3).strip() if head else ''
            if not head:
                head = _WIKI_HEADING_OPEN_RE.match(line)
            if head:
                level = len(head.group(1))
                idx = level - 2
                counters[idx] = update(counters[idx])
                for j in range(idx + 1, len(counters)):
                    counters[j] = '0'
                line = form_line(counters[:idx + 1], head.group(2), level)
                # 標題後面同一行還有正文，接在標題下方而不是併進標題
                if trailing:
                    line = line.rstrip('\n') + '\n' + trailing
            elif line.startswith('***'):
                line = '  * ' + line[3:].strip()
                item_line, prev_item_line = True, True
            elif line.startswith('**'):
                line = ' * ' + line[2:].strip()
                item_line, prev_item_line = True, True
            elif line.startswith('*') or line.startswith('#'):
                line = '* ' + line[1:].strip()
                item_line, prev_item_line = True, True
            else:
                pass

            if not item_line and prev_item_line:
                fresh_text += '\n'
                prev_item_line = False

            nl = '\n' if item_line else self.nl
            fresh_text += line + nl
        return fresh_text

    def parse(self, content):
        word, text, ID = content

        if self.__is_not_word(word) or \
           self.__is_redirect(text) or \
           self.__is_disambiguation(text):
            return None, None, []

        # 空頁不處理。判準是「有沒有內容」，不是「有多短」——原本以原始
        # wikitext 30 字為界，把「李某某是中國演員。」這種完整的一句話條目
        # 也擋掉了（見 md_to_dataset.MIN_DOC_LENGTH 的說明）。
        if not text.strip():
            return None, None, []
        # 原文若本身就含我們保留的私有區字元，遮罩還原時會把它變成別的字元
        # （`\ue007` 會變成 `<`）。全量掃過：494 萬頁裡有 30 頁真的中招。
        # 進來就清掉，讓保留區永遠只有我們自己寫的標記。
        text = _RESERVED_PUA_RE.sub('', text)
        # 頁面名稱魔術字要先換成條目標題。展開成空字串的話，條目的第一句會
        # 失去主語——`{{PAGENAME}}是杜鵑花科的一個屬。` 變成「是杜鵑花科的
        # 一個屬。」，實測 0.48% 的條目中招，`矽`、`軟體`、`脫氧核糖核酸`
        # 這些主要條目都在裡面。
        text = _PAGENAME_MAGIC_RE.sub(lambda _m: word, text)
        images = []
        text = self.__clean(text, images, word)
        
        # 在處理前先清理代理字符
        word = self.__clean_surrogates(word)
        text = self.__clean_surrogates(text)
        
        text = self.__fresh(word, text)

        return ID, text, images
