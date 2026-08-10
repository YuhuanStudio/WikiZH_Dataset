"""
文本正規化

唯一原則：

    **只移除「確定是標記」或「確定是空殼」的東西，絕不刪除自然語言內容。**

任何「看到某個關鍵字就刪一整段」的規則都不該存在——該做的是把值救回來
（模板展開見 wiki_parser.py），救不回來時也只切掉壞掉的那一小塊。

這條原則是有代價的：清理規則必須寫得精準，不能圖方便。以下都是實測踩過的
反例，註解裡保留它們是為了避免再次退回那種寫法：

    「公式為E = mc²，其中c是光速。」        → 「公式為」
    「他擁有Ph.D.學位，任職於U.S. Steel。」  → 「他擁有PhD學位，任職於US Steel。」
    「該公司AT&T與C#語言。」                → 「該公司ATT與C語言。」
    「決賽於溫布萊球場舉行。」               → 整句消失
    「[[File:x.jpg|圖說]]這座建築…落成。」   → 「。」
    snake_case_name                       → snakecasename
"""

import html
import re
import unicodedata

import zhconv
from tw_vocab import to_tw_vocab
from wiki_parser import VARIANT_OPEN, VARIANT_SEP, VARIANT_CLOSE, remove_file_links


# ============================================================
# 語言變體
# ============================================================

_VARIANT_RE = re.compile(
    re.escape(VARIANT_OPEN) + r'(.*?)' + re.escape(VARIANT_SEP) + r'(.*?)' + re.escape(VARIANT_CLOSE),
    re.DOTALL,
)
_VARIANT_LEFTOVER_RE = re.compile('[' + VARIANT_OPEN + VARIANT_SEP + VARIANT_CLOSE + ']')

def resolve_variants(text, lang):
    """
    挑選維基編者手寫的地區用詞（-{zh-tw:賽局理論;zh-cn:博弈论}-）。

    這是品質最高的一層轉換，優先於任何自動轉換。
    """
    idx = 1 if lang == 'tw' else 2
    return _VARIANT_LEFTOVER_RE.sub('', _VARIANT_RE.sub(lambda m: m.group(idx), text))


# 不做繁簡轉換的區段：圍欄程式碼、行內反引號、行內與行間公式。
# 順序有意義——圍欄要先配，否則圍欄裡的反引號會先被行內規則吃掉。
# 行間公式 `$$…$$` 也要排在行內 `$…$` 前面。
# 跨度設上限：落單的 ``` 或 `$$` 每出現一次就會掃到文末，不設限就是 O(n²)。
_NO_CONVERT_RE = re.compile(
    r'(?ms)^```.{0,50000}?^```[ \t]*$'   # 圍欄程式碼區塊
    r'|`[^`\n]{1,2000}`'                 # 行內程式碼
    r'|\$\$.{0,20000}?\$\$'              # 行間公式
    r'|\$[^$\n]{1,2000}\$')              # 行內公式


_CJK_RE = re.compile(r'[一-鿿]')
# LaTeX 的痕跡：`\frac`、`\text`、上下標的 `^{`／`_{`、換行的 `\\`
_LATEX_HINT_RE = re.compile(r'\\[A-Za-z]+|[\^_]\{|\\\\')


def _is_verbatim(span):
    """這一段真的是逐字區塊，還是 `$` 配錯對圈到的散文

    `$…$` 是行內公式的寫法，但 `$` 在中文條目裡更常是**貨幣符號**或
    `$MFT`、`$Volume` 這種 NTFS 檔名。兩個這樣的 `$` 之間夾著的整段散文
    會被當成公式，於是**完全不做繁簡轉換**——`反恐精英：全球攻勢` 裡
    「$250,000」到「$1,000,000」之間那段就這樣留著「举办了赛事系列」，
    `NTFS` 整篇更是大半沒轉。

    判準量出來的：全庫 250,956 個 `$…$` 區段裡 7,359 個含中文，其中只有
    126 個（1.7%）帶 LaTeX 記號——那些是 `\text{光照}` 這類化學反應式，
    是真公式；其餘 7,233 個全是貨幣與檔名。所以：**含中文又沒有 LaTeX
    記號的，就不是公式**。

    圍欄程式碼與行內程式碼有明確的起訖標記，不受這個問題影響，一律逐字。
    """
    if not span.startswith('$') or span.startswith('$$'):
        return True
    if not _CJK_RE.search(span):
        return True
    return bool(_LATEX_HINT_RE.search(span))


def convert_script(text, lang):
    """
    繁簡轉換，用維基百科自己的轉換表（見 zhconv.py）。

    不用 OpenCC 的理由是量出來的：逐字對照網站 zh-tw 版，OpenCC 的偏差是
    0.96%，錯的清一色是維基那張表涵蓋、OpenCC 沒有的台灣慣用詞與異體字
    （臺灣/台灣、發佈/發布、鏈接/連結、二噁英/戴奧辛、肖邦/蕭邦、溼/濕）。
    表打底之後再疊一層 tw_vocab 的 IT 詞白名單：維基表偏保守，會漏掉
    網絡→網路、視頻→影片、用戶→使用者、內存→記憶體、計算機→電腦。

    刻意不用 OpenCC 的 `s2twp` 補這一段——它在百科語境的誤轉率極高，
    12 句一般散文測試中錯了 7 句（大力支持→大力支援、運動會項目→運動會
    專案、重整程序→重整程式、文件由政府發布→檔案由政府釋出），
    而維基表 0 誤轉。白名單只收在一般語境不會有其他意思的詞。

    **程式碼與公式不轉**，與 MediaWiki 自己的轉換器一致：LanguageConverter
    把 `<code>`、`<pre>`、`<syntaxhighlight>`、`<math>` 標成不轉換區。轉了會
    改掉程式的語意——`Console.WriteLine("简体字符串")` 變成 `"簡體字串"`
    （字串內容被改寫）、`// 构造函数` 的 `函数` 被台灣詞表換成 `函式`、
    LaTeX 下標 `x_{数据}` 也跟著變。逐字區塊就是逐字。
    """
    # 四個不轉區段的語法一定包含 ` 或 $。大多數一般條目兩者都沒有，
    # 先用 C 層的 str 搜尋可避免每篇都讓複雜正則掃完全文。
    if '`' not in text and '$' not in text:
        return _convert_prose(text, lang)
    if not _NO_CONVERT_RE.search(text):
        return _convert_prose(text, lang)

    out, pos = [], 0
    while True:
        m = _NO_CONVERT_RE.search(text, pos)
        if not m:
            break
        if _is_verbatim(m.group(0)):
            out.append(_convert_prose(text[pos:m.start()], lang))
            out.append(m.group(0))
            pos = m.end()
        else:
            # 這個 `$…$` 不是公式。只跳過開頭那一個 `$`，從它之後重新找——
            # 整段跳過的話，貨幣的 `$` 會把後面真正的公式一起吃掉。
            out.append(_convert_prose(text[pos:m.start() + 1], lang))
            pos = m.start() + 1
    out.append(_convert_prose(text[pos:], lang))
    return ''.join(out)


def _convert_prose(text, lang):
    converted = zhconv.convert(text, lang)
    return to_tw_vocab(converted) if lang == 'tw' else converted


# ============================================================
# 殘留標記清理
# ============================================================

_MAX_MARKUP_SPAN = 20000


def _looks_like_markup(span):
    """配對出來的這一段是標記，還是括號錯位圈到的正文"""
    return len(span) <= _MAX_MARKUP_SPAN and span.count('\n\n') <= 1


def _remove_balanced(text, open_tok, close_tok):
    """
    用括號配對移除區塊，正確處理巢狀。

    正則做不到這件事：`\\{\\{.*?\\}\\}` 遇到 `{{外層|{{內層}}}}` 會在內層的
    `}}` 就收尾，於是正文留下 `}|time=}}` 這種殘骸（實測 0.36% 的條目中招）；
    改成貪婪又會把兩個相鄰模板之間的正文一起吃掉。

    掃描用 str.find 在標記之間跳躍，不逐字元檢查（逐字元版本在解析階段
    佔掉近四成時間）。
    """
    if open_tok not in text:
        return text
    out = []
    pos = 0
    i = text.find(open_tok)
    olen, clen = len(open_tok), len(close_tok)
    while i != -1:
        depth, j, n = 0, i, len(text)
        end = -1
        while j < n:
            nxt_open = text.find(open_tok, j)
            nxt_close = text.find(close_tok, j)
            if nxt_close == -1:
                break
            if nxt_open != -1 and nxt_open < nxt_close:
                depth += 1
                j = nxt_open + olen
            else:
                depth -= 1
                j = nxt_close + clen
                if depth == 0:
                    end = j
                    break
        # 沒有對應的結尾，或圈到的範圍大得不像標記（模板的參數區塊裡不會有
        # 空行，正文會），都保留原文不亂刪——括號配對一旦錯位，無條件刪除
        # 會讓整段內容消失。
        if end == -1 or not _looks_like_markup(text[i:end]):
            i = text.find(open_tok, i + olen)
            continue
        out.append(text[pos:i])
        pos = end
        i = text.find(open_tok, pos)
    out.append(text[pos:])
    return ''.join(out)


_ORPHAN_BRACE_RE = re.compile(r'\{\{+|\}\}+')


def remove_leftover_templates(text):
    """
    移除 stage-1 沒能展開的殘留模板（含巢狀），以及配對不到的孤兒括號。

    上游把外層模板部分處理掉後，正文裡會留下落單的 `}}`：
        「…以每個樂章皆充滿了強烈的節奏律動為特色}}。」
    `{{` 與 `}}` 不會出現在自然語言裡，配對不到就直接清掉。
    """
    return _ORPHAN_BRACE_RE.sub('', _remove_balanced(text, '{{', '}}'))


def remove_leftover_tables(text):
    """移除 stage-1 沒能轉換的殘留表格（含巢狀）"""
    return _remove_balanced(text, '{|', '|}')
_CATEGORY_RE = re.compile(r'\[\[\s*(?:Category|分類|分类)\s*:[^\]]*\]\]', re.I)
_MAGIC_WORD_RE = re.compile(r'__[A-Z]+__')
_HTML_COMMENT_RE = re.compile(r'(?s)<!--.*?-->')
# ref 標籤。除了正常寫法，還要涵蓋維基上手誤造成的「沒有閉合角括號」版本
# （全量 155 萬篇跑出來的實例）：
#     另有一個已關閉出口<ref name="MTA-ReviewAC-2015"。   ← 開標籤沒閉合
#     沒能順利挺過。</ref他在2020年去世。                   ← 孤兒閉標籤
#
# 這兩種要分開處理：前者要連屬性一起清掉，後者只能清掉標籤本身——
# 如果對孤兒標籤也套「清到下一個句號」，會把後面真正的句子一起吃掉。
_REF_RE = re.compile(
    r'(?is)'
    r'<ref[^>\n]*>.{0,20000}?</ref\s*>'  # 正常的成對標籤（跨度設上限，
                                             #   否則沒收尾的 <ref> 會配到很遠的 </ref>）
    r'|<ref[^>\n]*/>'                    # 自閉合
    r'|<ref(?![a-zA-Z])[^>\n]{0,120}?(?=[。，\n]|$)'   # 開標籤缺 >，連屬性清掉
    r'|</?ref(?![a-zA-Z])'               # 孤兒標籤，只清標籤本身
)
_HTML_TAG_RE = re.compile(r'<[^>\n]{1,200}>')
_EXTERNAL_LINK_RE = re.compile(r'\[(?:https?|ftp)://[^\s\]]+(?:\s+([^\]\n]{0,300}))?\]')
_GLUED_URL_RE = re.compile(r'(?<=[一-鿿])(?:https?|ftp)://[^\s，。、；：）」』\]]+')
_WIKI_LINK_PIPED_RE = re.compile(r'\[\[([^\[\]|]*)\|([^\[\]|]*)\]\]')
_WIKI_LINK_RE = re.compile(r'\[\[([^\[\]|]*)\]\]')

# 具名實體（`&phi;`）。刻意不含 `&amp;`／`&lt;`／`&gt;`——那三個在上面單獨處理，
# 避免把 `&amp;lt;` 這種雙重轉義的內容錯誤地解成標籤。
_NAMED_ENTITY_RE = re.compile(r'&(?!amp;|lt;|gt;|#)[a-zA-Z][a-zA-Z0-9]{1,30};')
_HTML_ENTITIES = {
    '&lt;': '<', '&gt;': '>', '&amp;': '&', '&quot;': '"', '&apos;': "'",
    '&nbsp;': ' ', '&#39;': "'", '&mdash;': '—', '&ndash;': '–',
    '&ldquo;': '“', '&rdquo;': '”', '&hellip;': '…', '&middot;': '·',
}

# 只在「整行都是模板參數」時才刪。v1 是無條件刪掉任何 key=value 到行尾，
# 結果把含有等號的正文（數學式、化學式）整段砍掉。
#
# 參數名不能含空白（最多允許一個，例如 `image caption`），否則
# 「He said A=B and C=D clearly here.」會被當成 key「He said A」= value。
# 必須有 `|` 前綴。沒有前綴時 `A T = − A`（反對稱矩陣的公式）會被當成
# 參數行刪掉——公式現在會保留 LaTeX，這種誤判就會直接吃掉內容。
# 沒有 `|` 的參數殘骸由解析階段的 drop_orphan_params 與 _skip_param_block 處理。
_PARAM_LINE_RE = re.compile(
    r'(?m)^\s*\|\s*[a-zA-Z_][a-zA-Z0-9_\-]*(?:\s[a-zA-Z0-9_\-]+)?\s*=\s*[^\n]*$'
)


# 只由標記符號組成、沒有任何實質文字的行。
#
# 巢狀模板被上游部分處理後會留下孤兒殘骸：
#     }|time=}}
#     }} }}
#     |style2= style="width:5px;"|head2=yes|
#
# 判準：沒有中文、含有 { } | 之一、且**整行沒有句讀**。
# 英文正文幾乎一定有逗號或句號（"He said A=B and C=D clearly here."），
# 而模板殘骸不會有——用這個區分比「有沒有英文單字」可靠得多，
# 因為殘骸裡常常就帶著 time、style、head 這類參數名。
# 用 lookahead 確認條件，主體不做可回溯的 `.{0,200}X.{0,200}`
_DEBRIS_LINE_RE = re.compile(
    r'(?m)^(?![^\n]*[一-鿿])(?![^\n]*[.,!?？。，！])(?=[^\n]*[{}|])[^\n]{0,400}$'
)


# 圖片語法沒被 remove_file_links 涵蓋的殘骸：檔名後面直接跟說明文字
# （`MobileHero.jpg|中華民國經濟部舉辦的比賽`），以及表格分隔線 `---- ---- ----`
#
# 檔名部分寫成有界的「詞（空白詞）*」而不是 `[\w\s.()\-]+`——後者遇到長行
# 會回溯到爆炸。
_IMAGE_CAPTION_RE = re.compile(
    r'[\w.()\-–—,&+!\x27]{1,60}(?:[ \t][\w.()\-–—,&+!\x27]{1,60}){0,15}'
    r'\.(?:jpg|jpeg|png|gif|svg|webp|ogg|ogv|webm|pdf)[ \t]*\|', re.I)
_RULE_LINE_RE = re.compile(r'(?m)^[\s\-–—_=|｜]{4,}$')

# remove_file_links 沒攔到的圖片語法會留下 File:／檔案: 前綴
_FILE_PREFIX_RE = re.compile(r'(?m)^\s*(?:File|Image|Media|檔案|档案|文件|圖片|图片|圖像|图像)\s*:\s*', re.I)

# 前綴連著檔名黏在表格列裡（《中央線快速》的 `File:JR area YAMA.png東京｜…`），
# 不在行首所以上面那條攔不到。「前綴 + 檔名 + 圖片副檔名」不可能是自然語言，
# 出現在哪都可以清掉。檔名寫成有界的「詞（空白詞）*」避免回溯爆炸。
_FILE_TOKEN_RE = re.compile(
    r'(?:File|Image|Media|檔案|档案|文件|圖片|图片|圖像|图像)\s*:\s*'
    r'[\w.()\-–—,&+!\x27]{1,60}(?:[ \t][\w.()\-–—,&+!\x27]{1,60}){0,15}'
    r'\.(?:jpg|jpeg|png|gif|svg|webp|ogg|ogv|webm|pdf|tif|tiff)', re.I)

# 上面兩條圖片殘骸正則命中時，必定先出現這個副檔名。用簡單、
# 沒有可變量詞的正則當 gate，並保留 re.I 對 Unicode 大小寫的原始語意。
_IMAGE_EXTENSION_RE = re.compile(
    r'\.(?:jpe?g|png|gif|svg|webp|ogg|ogv|webm|pdf|tiff?)', re.I)


def _is_navbox_line(line):
    """
    導航框殘骸：用半形 | 串起來的一排連結（表格已轉成全形｜，半形連續出現
    就是 navbox）。「中國歷史年表 | 中國歷史事件列表 | 中國君主列表…」

    刻意用純 Python 而不是正則。原本寫成
        (?:\\s*\\|\\s*[^\\n|]{1,40}){2,}
    這種巢狀量詞，遇到管線很多的長行會指數級回溯——實測讓單一頁面把一個
    worker 卡住超過 12 分鐘，整批轉換看起來像「多核沒生效」。
    """
    if '|' not in line or any(c in line for c in '。！？'):
        return False
    parts = [p.strip() for p in line.split('|')]
    if len(parts) < 3:
        return False
    return all(0 < len(p) <= 40 for p in parts)


def drop_image_debris(text):
    """移除圖片檔名殘骸、表格分隔線與導航框列"""
    if _IMAGE_EXTENSION_RE.search(text):
        text = _IMAGE_CAPTION_RE.sub('', text)
        text = _FILE_TOKEN_RE.sub('', text)
    text = _FILE_PREFIX_RE.sub('', text)
    if '|' in text:
        text = '\n'.join('' if _is_navbox_line(l) else l for l in text.split('\n'))
    return _RULE_LINE_RE.sub('', text)


# 行內的模板參數殘骸：`label2=末端裸子植物 |sublabel2=Acrogymnospermae`、
# `width="30pt"`。行首的由 _PARAM_LINE_RE 處理，這裡補行中的。
_INLINE_PARAM_RE = re.compile(r'(?:\|\s*)?[a-zA-Z_][a-zA-Z0-9_\-]{0,24}\s*=\s*"[^"\n]{0,60}"'
                              r'|\|\s*[a-zA-Z_][a-zA-Z0-9_\-]{0,24}\s*=\s*[^\s|\n]{0,40}')


def drop_markup_debris(text):
    """移除只剩標記符號的殘骸行，以及行中的模板參數殘骸"""
    if not re.search(r'[{}|=]', text):
        return text
    text = _DEBRIS_LINE_RE.sub('', text)
    return _INLINE_PARAM_RE.sub('', text)


# 註腳裡的程式碼與公式要撈出來。編者會把「這個方法的定義」整段放進 `<ref>`
# ——`Smalltalk` 條目的五段集合類別實作全在註腳裡，外殼一刪就整批消失。
# 判準不是標籤名，是「裡面有沒有圍欄或公式」：真正的引用不會有。
_FENCE_IN_REF_RE = re.compile(r'(?s)```[^\n]*\n.*?\n```|\$\$.+?\$\$')


def _drop_ref(m):
    kept = _FENCE_IN_REF_RE.findall(m.group(0))
    return ('\n\n' + '\n\n'.join(kept) + '\n\n') if kept else ''


def strip_leftover_markup(text):
    """清掉 stage-1 漏網的 wiki 標記，不碰自然語言內容"""
    text = _HTML_COMMENT_RE.sub('', text)
    text = _REF_RE.sub(_drop_ref, text)
    text = remove_leftover_templates(text)
    text = remove_leftover_tables(text)
    text = _PARAM_LINE_RE.sub('', text)
    text = drop_markup_debris(text)
    text = drop_image_debris(text)
    text = _CATEGORY_RE.sub('', text)
    text = remove_file_links(text)

    # 外部連結保留顯示文字：[http://x 文字] → 文字
    text = _EXTERNAL_LINK_RE.sub(lambda m: m.group(1) or '', text)
    # 緊貼中文的裸網址是參考文獻殘骸（「2347.69萬人https://…html，其中」），
    # 前後有空白的網址則可能是條目在示範網址（例如「統一資源定位符」），予以保留
    text = _GLUED_URL_RE.sub('', text)
    # 內部連結保留顯示文字
    for _ in range(3):
        new = _WIKI_LINK_PIPED_RE.sub(r'\2', text)
        new = _WIKI_LINK_RE.sub(r'\1', new)
        if new == text:
            break
        text = new

    text = _HTML_TAG_RE.sub('', text)
    text = _MAGIC_WORD_RE.sub('', text)
    for entity, char in _HTML_ENTITIES.items():
        text = text.replace(entity, char)
    # 數字型 HTML 實體
    text = re.sub(r'&#(\d{1,6});', lambda m: chr(int(m.group(1))), text)
    # 具名實體不只上面那十幾個：希臘字母（`&phi;`、`&lambda;`）、數學符號、
    # 各種破折號都會出現在圖說與正文裡。手列一定會漏，交給標準函式庫。
    # `&amp;` 已在上面處理過，這裡不會把 `&amp;lt;` 二次解碼成 `<`。
    if '&' in text:
        text = _NAMED_ENTITY_RE.sub(
            lambda m: html.unescape(m.group(0)), text)
    return text


# ============================================================
# Markdown 行內標記 → 純文字
# ============================================================

_BOLD_RE = re.compile(r"'''(.+?)'''|\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"''(.+?)''")
_MD_LINK_RE = re.compile(r'\[([^\]\n]*)\]\([^)\n]*\)')
_LIST_ITEM_RE = re.compile(r'(?m)^([ \t]*)[\*\-\+•]+[ \t]+(.*)$')
_NUM_ITEM_RE = re.compile(r'(?m)^([ \t]*)\d+[.)][ \t]+(.*)$')
_DEF_ITEM_RE = re.compile(r'(?m)^[;:][ \t]*(.*)$')


# 列表項被壓在同一行的情形：`- 前993年 *前933年 *前873年 …`。
# 來自模板（Columns-list）或表格儲存格裡的清單被攤平成一行。
# 判準要嚴：`*` 前後都有內容、同一行至少出現兩次，而且該行不是表格列
# （表格列已經用全形｜分隔，拆行會破壞欄位對齊）。單獨一個 `*` 可能是
# 乘號或註腳標記，不動。
# `;` 也要涵蓋：wikitext 的 `;詞條` 是定義列表，被壓平後會留下同樣的形狀
# （`長宏邨`、`葵涌道` 的交通資訊「- 港鐵：美孚站 ;巴士 ;專線小巴 ;紅色小巴」，
# 實測 0.8% 的條目中招）。判別關鍵是**分號前面有空白**——正常標點的分號是
# 黏在前一個字上的（`came; she left`），只有清單被壓平才會空一格。
_GLUED_LIST_RE = re.compile(r'(?<=[^\s*;])[ \t]+[*;][ \t]*(?=[^\s*;])')


def split_glued_list_items(text):
    """把黏在同一行的列表項拆成各自一行"""
    if ' *' not in text and ' ;' not in text:
        return text
    out = []
    for line in text.split('\n'):
        # 有句讀的是正文（`計算 a *b 的值時 *要注意`），不是被壓平的清單。
        # 但「一行裡黏了三個以上的 `*`」再怎麼看都是清單被壓平了，句讀擋不住——
        # `連鎖超市列表`、`鯉魚門` 整份店鋪清單擠成一行
        # （`*沃爾瑪 好又多（Trust-mart）… *家樂福 *大潤發 *物美商業 …`），
        # 因為裡面有全形逗號就整條規則放行，輸出就是一行幾千字的殘骸。
        glued = len(_GLUED_LIST_RE.findall(line))
        if '｜' in line or glued < 2 or (glued < 3 and any(c in line for c in '。，！？；')):
            out.append(line)
            continue
        parts = []
        for part in _GLUED_LIST_RE.split(line):
            part = part.strip().lstrip('*;').strip()
            if part.startswith('- '):
                part = part[2:].strip()
            if part:
                parts.append(part)
        # 被壓平的清單，項目是詞組（「家樂福」「專線小巴」）；數學式的「項目」
        # 卻是單一符號（`a * b * c * d` 會被切成 a、b、c、d，乘號全部消失）。
        # 過半是單字元就不是清單，原樣留著。
        singles = sum(1 for p in parts[1:] if len(p) <= 1)
        if len(parts) < 2 or singles * 2 > len(parts) - 1:
            out.append(line)
            continue
        out.extend('- ' + p for p in parts)
    return '\n'.join(out)


def markdown_inline_to_text(text):
    """
    去掉行內格式標記，保留文字內容。

    列表一律正規化成 `- `，不像 v1 把列表壓成行內的 `•`——那既破壞結構，
    也讓 21% 的資料帶著 `•` 這種訓練雜訊。
    """
    text = _BOLD_RE.sub(lambda m: m.group(1) or m.group(2), text)
    text = _ITALIC_RE.sub(r'\1', text)
    text = _MD_LINK_RE.sub(r'\1', text)
    # 反引號不剝掉：解析階段用它標記 nowiki 的字面內容，是刻意留下的邊界
    text = split_glued_list_items(text)
    text = _LIST_ITEM_RE.sub(lambda m: '- ' + m.group(2).strip(), text)
    text = _NUM_ITEM_RE.sub(lambda m: '- ' + m.group(2).strip(), text)
    text = _DEF_ITEM_RE.sub(r'\1', text)
    return text


# ============================================================
# 空白與標點正規化
# ============================================================

# 含 LRM/RLM 等方向控制字元（U+200E/200F）——純文字語料裡是雜訊
_INVISIBLE_RE = re.compile(r'[​-‏‪-‮⁠-⁤﻿­]')
_SPACE_LIKE_RE = re.compile(r'[  -   　]')
_CTRL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
# 私有使用區：來源文本本身就會出現（維基用 PUA 表示罕用字與自造符號），
# 這些字元在任何字型下都是方框，對語料沒有價值。
#
# U+E000–U+E015 是本專案自己保留的控制區段（語言變體標記、逐字區塊的字元
# 遮罩），要排除在外——這些遮罩得一路活到文檔組裝完才還原，被這條規則清掉
# 的話公式與程式碼會少掉大括號、管線與括號。還原之後不該有殘留，
# validate_full 的「私有區字元」檢查就是這道防線。
_PUA_RE = re.compile(r'[\ue017-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]')

# 全形標點旁的空白（只處理全形，ASCII 標點在英文裡需要空格）
_CJK_PUNCT = '。！？；：、，）」』】》〉'
_CJK_PUNCT_OPEN = '（「『【《〈'


def normalize_whitespace(text):
    """統一各種空白字元，收斂多餘空行，但保留段落結構"""
    text = _CTRL_RE.sub('', text)
    text = _PUA_RE.sub('', text)
    text = _INVISIBLE_RE.sub('', text)
    text = _SPACE_LIKE_RE.sub(' ', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def normalize_punctuation(text):
    """
    收斂重複標點與標點旁的贅空白。

    刻意不碰 ASCII 標點與符號：`Ph.D.`、`U.S. Steel`、`AT&T`、`C#`、
    `admin@example.com`、`E = mc²` 都是有意義的內容。
    """
    # 重複的全形標點
    text = re.sub(r'。{2,}', '。', text)
    text = re.sub(r'，{2,}', '，', text)
    text = re.sub(r'、{2,}', '、', text)
    text = re.sub(r'；{2,}', '；', text)
    text = re.sub(r'！{2,}', '！', text)
    text = re.sub(r'？{2,}', '？', text)
    # 標點相鄰時的冗餘（，。 → 。）
    text = re.sub(r'[，、；]+(?=[。！？])', '', text)
    # 全形標點與中文之間不該有空白
    text = re.sub(r'([' + _CJK_PUNCT + r'])[ \t]+(?=[一-鿿])', r'\1', text)
    # 清單標記後面的那個空白不能吃掉。繁體轉換會把 `’89` 的撇號轉成 `』`
    # （維基自己的標點慣例），這條規則接著把 `- 』89` 收成 `-』89`，清單標記
    # 就此失效——`山水情` 的獲獎清單、`村瀨步` 的廣播節目清單都因此繁簡兩版
    # 項目數對不上。
    text = re.sub(r'(?m)(?<!^-)(?<!^\*)(?<!^•)[ \t]+(?=[' + _CJK_PUNCT + r'])', '', text)
    text = re.sub(r'([' + _CJK_PUNCT_OPEN + r'])[ \t]+', r'\1', text)
    # 句首殘留的標點
    text = re.sub(r'(?m)^[，、；：。]+\s*', '', text)
    return text


_EMPTY_BRACKETS = [
    (re.compile(r'（([^（）]*)）'), '（', '）'),
    (re.compile(r'\(([^()]*)\)'), '(', ')'),
    # 全形配半形。原文本來就這樣寫的很少，多半是括號裡的內容來自模板，
    # 模板被移除後只剩兩個對不起來的括號：
    #   [句號] 「句號（)，也稱作句點」  [豐田Supra] 「豐田Supra（)，是…」
    (re.compile(r'（([^（）()]*)\)'), '（', ')'),
    (re.compile(r'\(([^（）()]*)）'), '(', '）'),
    (re.compile(r'「([^「」]*)」'), '「', '」'),
    # 彎引號也要收：簡體版用的是 `“”`，漏掉的話同一個空引號繁體刪了、簡體留著
    # （`鳴海繪里香` 的表格列尾巴多一個 `｜“”`）
    (re.compile(r'“([^“”]*)”'), '“', '”'),
    (re.compile(r'‘([^‘’]*)’'), '‘', '’'),
    (re.compile(r'【([^【】]*)】'), '【', '】'),
    (re.compile(r'《([^《》]*)》'), '《', '》'),
]


# 括號裡「只剩這些」才算空殼：空白，加上模板被清空後留下的連接標點。
# 判準不能寫成「沒有 \w 字元」——被引號框起來的符號本身就是內容，圖例、
# 對照表整篇都靠它：`- 「*」：打榜中`、`- 「/」表示未有相關資料`（`鄭基高`）。
# 那條舊規則會把符號連同引號一起抹掉，剩下「-：打榜中」，讀者再也不知道
# 那個記號是什麼。而且它只在繁體發生——`“*”` 轉成繁體才變成 `「*」`，
# 簡體的彎引號不在括號表裡，於是繁簡兩版的清單項數對不上。
_BRACKET_FILLER_RE = re.compile(r'^[\s，、；;,]*$')


def drop_empty_brackets(text):
    """
    只移除「裡面沒有任何內容」的括號。

    v1 會移除所有不含中文的括號，把（61.99%）、（1948-1956）、（SABC）、
    （Infinity）、（なつみ）這些原文資訊靜默刪掉。
    """
    for _ in range(2):
        changed = False
        for pattern, _o, _c in _EMPTY_BRACKETS:
            new = pattern.sub(
                lambda m: '' if _BRACKET_FILLER_RE.match(m.group(1)) else m.group(0), text)
            if new != text:
                text, changed = new, True
        if not changed:
            break
    # 模板展開成空字串後留下的連續分隔標點：（Le Chambon，；Lo Chambon）
    text = re.sub(r'[，、;；]\s*(?=[，、;；])', '', text)
    text = re.sub(r'([（(])\s*[，、;；]\s*', r'\1', text)
    text = re.sub(r'\s*[，、;；]\s*(?=[）)])', '', text)
    return text


def finalize_block(text):
    """段落層級的收尾（標記已在文件層級清過，這裡不重複做）"""
    # 安全網：語言變體標記若因表格切分等原因斷成單邊，殘留的私有區字元
    # 絕不能流進資料集
    text = _VARIANT_LEFTOVER_RE.sub('', text)
    text = markdown_inline_to_text(text)
    text = unicodedata.normalize('NFC', text)
    text = normalize_whitespace(text)
    text = drop_empty_brackets(text)
    text = normalize_punctuation(text)
    return text.strip()


def clean_block(text):
    """單一段落的完整清理流程（含標記清理，供測試與單獨使用）"""
    return finalize_block(strip_leftover_markup(text))


# ============================================================
# 不具知識價值的章節
# ============================================================

# 短關鍵字（≤2 字）只認完全相同的標題，長關鍵字才允許前綴比對。
# 用子字串比對會誤刪實質內容：[烏西與烏克蘭其他地區的文化差異]、[理論來源]、
# [天然資源 > 石油及天然氣]、[翻譯說明] 都曾被當成參考章節整段丟掉。
SKIP_SECTION_KEYWORDS = [
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

_EXACT_ONLY_LEN = 2


def _canon_section(title):
    """章節標題正規化：統一成簡體再比對"""
    return _SECTION_CANON.convert(title.strip().lower())


# 要用**完整的 cn 表**（字元表 + 地區用詞表），不能只用 zh2Hans。
#
# 這些規則跑在已經轉換過的文字上，而兩邊轉出來的詞不一樣：同一個
# `== 鏈結 ==`，簡體版被 zh2CN 轉成「链接」（命中清單、丟棄），繁體版留著
# 「鏈結」，只用 zh2Hans 正規化會得到「链结」（不在清單、保留）。
# 結果就是繁體版多出「## 鏈結」「## 網站鏈結」這些章節（劉伯明、荔景、
# 南京大學歷史學院…）。完整表會把「鏈結」也正規化成「链接」，兩邊一致。
_SECTION_CANON = zhconv.get_converter('cn')
_SKIP_KEYWORDS_CANON = sorted(
    {_SECTION_CANON.convert(k.strip().lower()) for k in SKIP_SECTION_KEYWORDS if k.strip()})
_SKIP_EXACT_CANON = frozenset(
    k for k in _SKIP_KEYWORDS_CANON if len(k) <= _EXACT_ONLY_LEN)
_SKIP_PREFIX_CANON = tuple(
    k for k in _SKIP_KEYWORDS_CANON if len(k) > _EXACT_ONLY_LEN)


def should_skip_section(title_path):
    """
    參考資料、外部連結這類章節整段不要（逐層判斷，任一層命中就跳過）。

    比對前把兩邊都正規化成同一字體。關鍵字清單原本是手工列繁簡兩種寫法，
    只要漏一種就會兩個語言版本不一致——`連結` 在清單裡、簡體的 `连结` 不在
    （清單只有 `链接`），於是繁體版正確丟掉「外部連結」章節，簡體版卻留著。
    實測 4,369 筆條目兩版結構不同。
    """
    for part in title_path:
        title = _canon_section(part)
        if not title:
            continue
        if title in _SKIP_EXACT_CANON or title.startswith(_SKIP_PREFIX_CANON):
            return True
    return False
