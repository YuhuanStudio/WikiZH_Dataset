"""行為黃金案例：每個修好的缺陷都留一個案例，改動後必跑

壓力測試守的是「會不會卡死」，這支守的是「輸出對不對」。加它的直接原因是
我自己踩過的一次退步：接合折行的規則沒有排除圍欄**內部**，於是
`def f():` 跟 `    return 1` 被併成一行，整份程式碼毀掉——而當時所有殘留
字元檢查、結構不變量都是綠的，是人工翻輸出才發現的。

每個案例都對應一個實際壞過的條目，註解裡寫著是哪一篇。

用法：python qa/cases.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import md_to_dataset as md
import template_store
import wiki_parser as wp

PARSED = os.environ.get('WIKIZH_PARSED_DIR', 'parsed/202608')
# 條目要夠長才會被收錄，案例前後補一段中性文字
PAD = '這是前導段落，長度足夠讓條目通過收錄門檻的限制條件設定。\n\n'
TAIL = '\n\n這是結尾段落，同樣需要足夠的長度才會被收錄進資料集。'


# 有些案例必須是「整篇條目」而不是被夾在前後文之間（重定向只認開頭的 `#`），
# 開頭寫 `#RAW#` 就不補前後文。
RAW_MARK = '#RAW#'


def build_omni_record(body, title='測試', lang='tw'):
    """回傳 omni 版記錄（正文帶 `<image>` 佔位符 + 圖片陣列）"""
    parser = wp.WIKIParse.__new__(wp.WIKIParse)
    parser.markdown = True
    parser.nl = '\n\n'
    _id, doc, images = parser.parse((title, PAD + body + TAIL, '1'))
    if doc is None:
        return None
    return md.process_page({'id': '1', 'text': doc, 'images': images},
                           lang=lang, omni=True)


def build(body, title='測試', lang='tw'):
    parser = wp.WIKIParse.__new__(wp.WIKIParse)
    parser.markdown = True
    parser.nl = '\n\n'
    if body.startswith(RAW_MARK):
        source = body[len(RAW_MARK):]
    else:
        source = PAD + body + TAIL
    _id, doc, _images = parser.parse((title, source, '1'))
    if doc is None:
        return ''
    # 一定要走 process_page：繁簡轉換與變體挑選在那裡做，
    # 直接呼叫 build_document 會漏掉，`{{〈}}` 的兩個變體就會一起留下來。
    record = md.process_page({'id': '1', 'text': doc}, lang=lang)
    if not record:
        return ''
    text = record['text']
    # 去掉標題行與前後補的段落，只留案例本身
    inner = text.split('\n', 2)[2] if text.count('\n') >= 2 else text
    inner = inner.replace(PAD.strip(), '').replace(TAIL.strip(), '')
    return inner.strip()


# (名稱, 輸入 wikitext, 必須出現的字串, 絕對不能出現的字串)
CASES = [
    # Trie／記憶體洩漏：一般行用空行分段的規則套到程式碼上，每行變成一段
    ('程式碼保留縮排與換行',
     '<syntaxhighlight lang="py">\ndef f():\n    if x:\n        return 1\n</syntaxhighlight>',
     ['```py\ndef f():\n    if x:\n        return 1\n```'],
     ['def f():    if x:', 'def f():\n\n']),
    # TeX／GTK：產生圍欄時把兩端的遮罩空白也剝掉，第一行的縮排就沒了
    ('程式碼首行的縮排要保留',
     '<pre>\n  The quick brown fox jumps over\nthe lazy dog.\n</pre>',
     ['```\n  The quick brown fox jumps over\nthe lazy dog.\n```'],
     ['```\nThe quick']),
    # 沒有語言標註時，開頭的圍欄也長得像收尾，一度讓每個 <pre> 多一個空行
    ('無語言標註的圍欄不多空行',
     '<pre>\nhello world\nsecond line\n</pre>',
     ['```\nhello world\nsecond line\n```'],
     ['```\n\nhello']),
    # 全球資訊網：`</pre>。` 的句號黏到收尾圍欄上，圍欄就配不成對了
    ('圍欄後的句號不黏上來',
     '<pre>\n<a href="x">Home</a>\n</pre>。',
     ['```\n<a href="x">Home</a>\n```'],
     ['```。']),
    # 經濟學／中國歷史：編者折行讓一句話被切成兩段，句號孤零零落在下一段
    ('折行的句子接回同一段',
     '經濟學家認為自由貿易才是增進整體利益的方式\n。依據調查已達成共識。',
     ['方式。依據調查'],
     ['方式\n\n。']),
    ('完整的兩句不會被併起來',
     '這是完整的一段話。\n這也是完整的一段話。',
     ['這是完整的一段話。\n\n這也是完整的一段話。'],
     ['一段話。這也是']),
    ('空行分段不會被併起來',
     '1990年代的發展\n\n這個時期的內容說明。',
     ['1990年代的發展\n\n這個時期的內容說明。'],
     ['發展這個時期']),
    # 絲狀病毒科：同一原始碼行的參數是同一列，不能一個參數一段
    ('清單模板依原始碼換行分列',
     '本科包括以下屬：\n{{common taxon list|italic=yes\n|奎瓦病毒屬 |Cuevavirus |\n'
     '|滇絲病毒屬 |Dianlovirus |\n}}',
     ['奎瓦病毒屬｜Cuevavirus', '滇絲病毒屬｜Dianlovirus'],
     ['奎瓦病毒屬\n\nCuevavirus']),
    # 圖版遊戲／一世一元制：圖說被當成一般段落，變成飄在半空的單詞
    ('圖庫圖說轉成清單且解析連結',
     '<gallery>\nFile:A.jpg|西洋棋\nFile:B.jpg|[[明太祖|太祖]]洪武帝\n</gallery>',
     ['- 西洋棋', '- 太祖洪武帝'],
     ['太祖]]', '\n\n西洋棋']),
    # 人類免疫缺陷病毒：`;詞條` 與 `:解釋` 分家，詞條變成孤零零一行
    ('定義列表併成一句',
     ';Gag\n:gag基因產生55kD的蛋白p55。',
     ['Gag：gag基因產生55kD的蛋白p55。'],
     ['Gag\n\ngag基因']),
    # 查詢字串／辛巴威元：引號裡的符號被當成空括號，連引號一起刪掉
    ('引號裡的符號不會被刪',
     '圖例：\n* “*”：打榜中\n* 空格編碼為“+”或“%20”。',
     # 繁體版的彎引號會轉成 `「」`，重點是符號本身還在
     ['「*」：打榜中', '「+」'],
     ['- ：打榜中', '- 空格編碼為或']),
    # 臺灣盲文：表格是在介紹括號／引號本身；內部為空不代表它是模板殘骸。
    # 簡體轉換又會把 `『 』` 變成 `“ ”`，舊規則只刪簡體版，造成欄數不一致。
    ('表格裡展示的空括號符號要保留',
     '符號對照如下：\n\n印刷體｜「 」｜『 』｜（ ）｜〔 〕｜｛ ｝\n\n點字｜1｜2｜3｜4｜5',
     ['印刷體｜「」｜『』｜（）｜〔 〕｜｛ ｝',
      '點字｜1｜2｜3｜4｜5'],
     []),
    # 相對論：`{{Equation box|equation=<math>…</math>}}` 把要顯示的方程式放在
    # 具名參數裡；公式沒有中文句讀，過不了「成段正文」那一關就整條消失
    ('公式方塊裡的方程式要保留',
     "愛因斯坦場方程：{{Equation box 1\n|indent=:\n|title='''Einstein'''\n"
     "|equation=<math>G_{\\mu\\nu} = 8\\pi G T_{\\mu\\nu}</math>\n|border colour=#50C878\n}}",
     ['$G_{\\mu\\nu} = 8\\pi G T_{\\mu\\nu}$'],
     []),
    # 楊輝三角形：兩條相鄰的行內公式黏成 $$，讀起來變成行間公式
    ('相鄰公式不會黏成行間分隔符',
     '推導：<math>a+b</math><math>c+d</math>相等。',
     ['$a+b$ $c+d$'],
     ['$$']),
    # 桃花源：模板本體裡的變體標記在展開後沒人處理，整段洩漏進正文
    ('模板展開後的變體標記會被挑選',
     '近人陳寅恪在{{〈}}桃花源記旁證{{〉}}中博引諸多地理著作考證。',
     ['〈桃花源記旁證〉'],
     ['-{', 'zh-cn:']),
    # 物理化學／Su-25：行內模板的參數折行，把一個句子切成兩段
    ('模板參數的折行會接回來',
     '歷史期刊有《{{lang|fr|Annales de chimie et de\nphysique}}》創刊於1789年。',
     ['《Annales de chimie et de physique》'],
     ['physique\n']),
    # 惠州公交L1路：空事實規則的字數預算讓繁簡兩版判斷不同——`公交车` 轉繁後是
    # `公車`，同一行簡體 21 字保留、繁體 20 字被丟掉，引導語就此消失
    ('引導語不會被當成空欄位丟掉',
     '* L1A路目前配置23台公交车，详细情况如下：\n\n00736D｜00806D｜01286D',
     ['詳細情況如下：'],
     []),
    ('沒有值的欄位還是要丟掉',
     '* 成立：\n* 網站：www.example.gov',
     ['網站：www.example.gov'],
     ['- 成立：']),
    # 連鎖超市列表／鯉魚門：整份清單被壓成一行，因為行裡有全形逗號就放行不切
    ('壓平的清單會被切開',
     '* 沃爾瑪 好又多（Trust-mart）（後併入沃爾瑪） *家樂福 *大潤發 *物美商業 *超市發',
     ['- 家樂福', '- 大潤發', '- 物美商業'],
     ['*家樂福']),
    ('有句讀的正文不會被誤切',
     '這句話裡 *有兩個 *標記，但它是一段完整的正文敘述而不是清單。',
     ['這句話裡 *有兩個 *標記'],
     ['- 有兩個']),
    # C♯：`\n+ → \n` 把程式碼區塊之間的空行一起收掉（27% 的區塊只差在這個）
    ('程式碼裡的空行要保留',
     '<syntaxhighlight lang="cs">\nstring status;\n\npublic string Status\n{\n    get;\n}\n</syntaxhighlight>',
     ['string status;\n\npublic string Status'],
     ['string status;\npublic string']),
    # 計算機程式：範例程式裝在 {{Side box|text=…}} 裡，側欄被整塊丟掉時陪葬
    ('模板裡的程式碼不會跟著外殼消失',
     '{{Side box|style=width:30em\n|text=<syntaxhighlight lang="c">int main(void) {\n    return 0;\n}</syntaxhighlight>\n|below=C的範例\n}}',
     ['```c', 'int main(void) {', '    return 0;'],
     []),
    # 電腦程式：範例程式排在前言位置，前言的「沒句讀又不到 12 字就是碎片」
    # 判準把 `#include <stdio.h>` 連同開頭的圍欄一起刪掉，後面每組圍欄跟著錯位
    ('前言位置的程式碼區塊完整保留',
     '{{Side box|text=<syntaxhighlight lang="c">#include <stdio.h>\nint main(void) {\n    return 0;\n}</syntaxhighlight>|below=C範例}}\n\n程式是一組指示電腦執行工作的指令集合。',
     ['```c\n#include <stdio.h>\nint main(void) {'],
     []),
    # ISWIM／Java和C++的对照：儲存格會被壓成一行，圍欄的開頭標記跟內容黏在
    # 一起就不再是圍欄（```` ```bash {λf.f(b+2c)…} ````），整欄程式碼失效
    ('表格儲存格裡的程式碼轉成行內反引號',
     '{| class="wikitable"\n! AE !! ISWIM\n|-\n'
     '| <syntaxhighlight lang="bash">{λf.f(b+2c)}</syntaxhighlight>\n'
     '| <syntaxhighlight lang="sml">let x = M; L</syntaxhighlight>\n|}',
     ['`{λf.f(b+2c)}`｜`let x = M; L`'],
     ['```bash {']),
    ('儲存格裡的多行程式碼整塊搬出表格',
     '{| class="wikitable"\n! C++ !! Java\n|-\n'
     '| <syntaxhighlight lang="cpp">class Foo {\npublic:\n    int x;\n};</syntaxhighlight>\n'
     '| 只支持類別\n|}',
     ['```cpp\nclass Foo {\npublic:\n    int x;\n};\n```'],
     [' / public:', '`class Foo {']),
    # 長宏邨／葵涌道：`;詞條` 定義列表被壓平，留下黏在一起的 ` ;`
    ('壓平的定義列表會被切開',
     '交通如下：\n* 港鐵：美孚站 ;巴士 ;專線小巴 ;紅色小巴',
     ['- 巴士', '- 專線小巴', '- 紅色小巴'],
     [';巴士']),
    ('一般分號標點不會被誤切',
     '這是一段正常的中文句子；裡面有分號；但分號前面沒有空白。',
     ['句子；裡面有分號；但分號'],
     ['- 裡面有分號']),
    # 山水情／村瀨步：繁體轉換把 `’89` 的撇號轉成 `』`（維基自己的標點慣例），
    # 「全形標點前不留空白」的規則接著把 `- 』89` 收成 `-』89`，清單標記失效
    ('清單標記後的空白不會被吃掉',
     "* ’89上海文化藝術節優秀成果獎\n* 廣播電影電視部1988年優秀影片獎",
     ['\n- '],
     ['-』', '-」']),
    # 摩斯特／黃帝祭：`連結`（繁）正規化成 `连结` 命中跳過清單，`链结`（簡）
    # 正規化還是 `链结` 不在清單裡——`连/链` 兩種簡化寫法收斂不到同一個字。
    # 章節跳過改成只在簡體版決定一次，兩邊共用。
    ('章節跳過的決定與語言無關',
     # 連結節要放中間：build() 會在最後補一段結尾文字，放最後會讓那段散文
     # 落進連結節裡，就不再是「純連結」了
     '== 連結 ==\n* 官方Facebook粉絲頁\n* 官方網站\n\n== 成員 ==\n* 溫力銘\n* 陳銘澤',
     ['## 成員'],
     ['## 連結', '官方Facebook粉絲頁']),
    # 以下五項來自 codex 的深度 review，都已重現確認
    # `#` 是有序清單的行首標記，只看「開頭是不是 #」會把整篇條目當成重定向丟掉
    ('有序清單開頭的條目不算重定向',
     '這是一篇條目的前言說明文字，長度足夠通過收錄門檻的限制條件。\n\n== 章節 ==\n#第一項\n#第二項',
     ['- 第一項', '- 第二項'],
     []),
    # {{sub}}／{{sup}} 渲染出來是可見的上下標，一度被歸進「純圖示」整個丟掉
    ('上下標模板不會被丟掉',
     '水的化學式是H{{sub|2}}O，而葡萄糖是C{{sub|6}}H{{sub|12}}O{{sub|6}}，指數寫成x{{sup|2}}。',
     ['H2O', 'C6H12O6', 'x2'],
     ['HO', 'CHO']),
    ('清單模板的項目不會被丟掉',
     '成員包括：\n{{ubl|甲某人|乙某人|丙某人}}',
     ['- 甲某人', '- 乙某人', '- 丙某人'],
     []),
    # <code> 裡是逐字內容，但 MediaWiki 在裡面照樣展開模板，兩者都要成立
    ('行內程式碼逐字保留但模板照樣展開',
     '呼叫 <code>f()</code> 與 <code>arr[ i ]</code>，參見 <code>{{le|SCHED_DEADLINE|排程}}</code>。',
     ['`f()`', '`arr[ i ]`', '`SCHED_DEADLINE`'],
     ['`f`', '`arr[i]`']),
    # nowiki 的內容先被逐字遮罩後，判斷器曾看不到真正的 `[[`／`<ref>`，
    # 因而沒加反引號；資料集收尾一還原，字面範例就被誤報成殘留標記。
    ('nowiki 裡的標記要保留逐字邊界',
     '以下展示原始寫法：<nowiki>[[目標|顯示文字]]</nowiki>。',
     ['`[[目標|顯示文字]]`'],
     []),
    ('nowiki 裡的 HTML 實體要保留逐字邊界',
     'HTML 實體寫成 <nowiki>&amp;yen;</nowiki>。',
     ['`&amp;yen;`'],
     []),
    # Windows Metafile：資訊框有一個沒收尾的 `<code>`。資訊框事實附到文末後，
    # 它曾配到別欄位的 `</code>`，把整篇正文當成行內程式碼遮起來。
    ('沒收尾的 code 不會吞掉後續正文',
     '{{Infobox file format\n| name = Demo\n| mime = </code><code>image/demo\n'
     '| extension = <code>.demo</code>\n}}\n'
     "'''Demo'''是一種檔案格式。\n\n== 歷史 ==\n正文完整。<ref>來源</ref>",
     ['## 歷史', '正文完整。'],
     ['<ref', '== 歷史 ==']),
    # 變體標記黏在圍欄標記上，所有「行首是不是 ```」的狀態機都認不出
    ('變體裡的程式碼區塊不會被撐開',
     '-{zh-hans:<syntaxhighlight lang="js">\nvar a = 1;  // 变量\nif (a) {\n    b();\n}\n</syntaxhighlight>'
     ';zh-hant:<syntaxhighlight lang="js">\nvar a = 1;  // 變數\nif (a) {\n    b();\n}\n</syntaxhighlight>;}-',
     ['```js\nvar a = 1;  // 變數\nif (a) {\n    b();\n}\n```'],
     ['var a = 1;  // 變數\n\nif']),
    # 以下來自 codex review 第二批
    # 地區用詞表被基礎字元表蓋掉：同一個詞在兩張表都有時，穩定排序讓先加入的
    # 基礎表永遠先命中（tw 26 個、cn 5 個衝突項）
    ('地區用詞表要蓋過基礎字元表',
     '他喜歡打台球，也去過索马里，並研究综合征的成因與治療方式。',
     ['撞球', '索馬利亞', '症候群'],
     ['檯球', '索馬里', '綜合徵']),
    # 數學式不是被壓平的清單：`a * b * c * d` 的「項目」是單一符號
    ('數學式不會被當成壓平的清單',
     '這個公式寫成 a * b * c * d 的形式，用來表示連乘的運算結果與意義。',
     ['a * b * c * d'],
     ['- b\n- c']),
    # 未閉合的 gallery 一路吃到頁尾，後面的章節整段消失
    ('未閉合的圖庫不會吞掉後文',
     '前言段落，長度足夠通過收錄門檻的限制條件設定文字說明。\n\n'
     '<gallery>\nFile:A.jpg|說明一\n\n== 後面的章節 ==\n這一整段正文不該消失，它在未閉合的圖庫之後。',
     ['## 後面的章節', '這一整段正文不該消失'],
     []),
    # 兩條公式間夾著會消失的標記（`''`、註解），清理後仍會黏成 $$
    ('中間夾著會消失的標記時公式仍要分開',
     "推導：<math>a+b</math>''<math>c+d</math>相等，這一段補足長度需求。",
     ['$a+b$ $c+d$'],
     ['$$']),
    ('換行標籤隔開的公式不會黏成行間分隔符',
     '常數為<math>a+b</math><br><math>c+d</math>，以下是完整說明。',
     ['$a+b$ $c+d$'],
     ['$$']),
    # codex review 第三批：把「因為短／因為不在白名單」的刪除全部拿掉
    ('資訊框保留白名單外的欄位與跨行值',
     '{{Infobox scientist\n| name = 愛因斯坦\n| image = E.jpg\n| image_size = 220px\n'
     '| known_for = 相對論<br>光電效應\n| workplaces = 蘇黎世理工、\n  普林斯頓高等研究院\n'
     '| office = 教授\n| office = 院長\n}}\n愛因斯坦是理論物理學家，提出相對論。',
     # 標籤來自 dump 自動抽出的對照表（infobox_labels.py）：模板頁自己寫的
     # `label = 知名於` / `label = 機構`。查不到的欄位才保留英文原鍵。
     ['知名於：相對論、光電效應', '機構：蘇黎世理工、普林斯頓高等研究院',
      '職位：教授', '職位：院長'],
     ['E.jpg', '220px']),
    # codex review 第二批
    # 導覽框跟事實資訊框共用命名慣例，但內容是導覽元素。混在一起的話
    # 「基本資料」會冒出 `group1：上古`、`list1：夏朝、商朝`
    ('導覽框不會被當成事實資訊框',
     '{{Navbox\n|name = 中國歷史\n|title = 中國歷史年表\n|group1 = 上古\n'
     '|list1 = [[夏朝]]、[[商朝]]\n}}\n中國歷史指中國的歷史發展過程，源遠流長。',
     ['中國歷史指中國的歷史發展過程'],
     ['基本資料', 'group1', 'list1', '中國歷史年表']),
    # 值裡「含有」px／.png 不代表它是版面設定。用子字串比對的話
    # `1080px顯示器的發明` 會讓整個欄位消失，而純檔名的欄位仍要跳過
    ('值裡含 px／檔名不代表它是版面設定',
     '{{Infobox scientist\n| image = Albert Einstein.jpg\n| image_size = 220px\n'
     '| known_for = 1080px顯示器的發明\n| awards = 使用example.png格式的標準\n'
     '| birth_date = 1879年3月14日\n}}\n這位科學家有多項發明，影響深遠。',
     ['1080px顯示器的發明', '使用example.png格式的標準'],
     ['Albert Einstein.jpg', '220px']),
    # 資訊框參數要依巢狀深度切。`{{plainlist|⏎* A⏎* B⏎}}` 的收尾 `}}` 自成一行，
    # 用正則找「行首的 |鍵 =」會斷在那裡，模板名直接漏進正文
    ('資訊框的巢狀清單模板要展開成並列的值',
     '{{Infobox scientist\n| name = 愛因斯坦\n| known_for = {{ubl|相對論|光電效應}}\n'
     '| awards = {{plainlist|\n* 諾貝爾物理學獎\n* 科普利獎章\n}}\n}}\n'
     '愛因斯坦是理論物理學家，提出相對論。',
     ['相對論、光電效應', '諾貝爾物理學獎、科普利獎章'],
     ['ubl', 'plainlist', '* 諾貝爾']),
    # 土耳其語的 `İ` 小寫後變成兩個碼位，拿 s.lower() 的副本找位置會整串錯開
    ('特殊大小寫字元不會讓圖片語法錯位',
     'İstanbul[[File:x.jpg|thumb|說明文字]]是土耳其最大的城市，人口眾多。',
     ['İstanbul是土耳其最大的城市'],
     ['File:x.jpg', '說明文字', 'thumb']),
    # 標籤查表：模板頁自己寫的 `label = …` 就是維基渲染出來的欄位名。
    # 同一個 `class` 在不同框是不同意思，靠各模板的專屬表分開
    ('資訊框標籤要查該模板自己寫的欄位名',
     '{{Infobox ship\n|class = 大和型戰艦\n|builder = 吳海軍工廠\n}}\n'
     '大和號是日本海軍的戰艦，於1941年服役。',
     ['大和型戰艦', '建造者：吳海軍工廠'],
     ['class：', 'builder：']),
    # 條目自己填的欄位名最準：`established_title=设立地级市` 就是
    # `established_date` 那一列在維基上真正渲染出來的標籤
    ('條目自己提供的欄位名要當標籤用',
     '{{Infobox settlement\n|subdivision_type1 = 省\n|subdivision_name1 = 广东省\n'
     '|established_title = 设立地级市\n|established_date = 1979年\n}}\n'
     '深圳市是中国广东省的副省级市，位于珠江口东岸。',
     ['省：廣東省', '設立地級市：1979年'],
     ['subdivision_name1', 'established_date', '成立：设立地级市']),
    # 通用 `{{Infobox}}` 直接寫在條目裡時，labelN／dataN 成對就是一條事實
    ('通用資訊框的 label／data 要配成一條事實',
     '{{Infobox\n|title = 某公司\n|label3 = 創辦人\n|data3 = 張三\n'
     '|label4 = 成立於\n|data4 = 1990年\n}}\n某公司是一家科技公司，總部設於台北。',
     ['創辦人：張三', '成立於：1990年'],
     ['label3', 'data3', 'label4']),
    # 分類階元的鍵在別的框裡是別的意思，但那只代表不能套分類學譯名，
    # 不代表這一欄沒有內容
    ('分類階元的鍵在別的框裡仍要保留內容',
     '{{Infobox automobile\n|class = 中大型車\n|manufacturer = 通用汽車\n}}\n'
     '別克君越是通用汽車推出的車款，於2006年上市。',
     ['中大型車', '通用汽車'],
     ['綱：中大型車']),
    # Python：縮排示例包在 `{{efn|…}}` 註腳模板裡，外殼不進正文，但那段程式碼
    # 沒有別的地方可去。而且 efn 寫在句子中間，行內展開會把圍欄壓成一行
    ('註腳模板裡的程式碼要保留且不被壓成一行',
     'Python 使用縮排來劃分程式碼區塊。{{efn|縮排示例：\n\n'
     '<syntaxhighlight lang="python">\ndef is_even(a: int) -> bool:\n'
     '    if a % 2 == 0:\n        return True\n</syntaxhighlight>\n}}\n\n'
     '這是縮排規則的說明段落內容。',
     ['```python\ndef is_even(a: int) -> bool:\n    if a % 2 == 0:\n        return True\n```'],
     ['def is_even(a: int) -> bool: if a']),
    # 捷爾諾波爾州：前言的變體標記裡有三個**自閉合**的 `<ref name="x" />`。
    # 撈註腳程式碼的規則若讓它跟遠處的 `</ref>` 配成一對，中間整段正文會被
    # 吃掉——實測簡體版的章節全部消失，而繁體版正常，parity 才抓到
    ('自閉合的 ref 不會吃掉後面的正文',
     "'''某州'''（-{zh-tw:又譯'''甲州'''<ref name=\"a\" />; zh-cn:又译'''甲州'''<ref name=\"a\" />}-）是一個州。\n\n"
     '== 歷史 ==\n原屬某帝國的一部份，一次大戰後歸屬改變<ref>來源一</ref>。\n\n'
     '== 行政區劃 ==\n下轄三個區，各有其行政中心。',
     ['## 歷史', '## 行政區劃', '原屬某帝國的一部份'],
     ['<ref', '來源一']),
    # Smalltalk：五段集合類別的實作全寫在 `<ref>` 註腳裡，filter_wiki 把整個
    # 註腳刪掉就整批消失。註腳外殼不進正文，但裡面的程式碼沒有別的地方可去
    ('註腳裡的程式碼要撈出來',
     '字串類別響應 select: 訊息，其定義寫在搜集類別中<ref>\nCollection Method Definitions:\n'
     '<syntaxhighlight lang="smalltalk">\nselect: aBlock\n  "Answer a new instance"\n'
     '</syntaxhighlight></ref>。以上是 Smalltalk 的說明文字。',
     ['```smalltalk\nselect: aBlock\n  "Answer a new instance"\n```'],
     ['Collection Method Definitions']),
    # 2channel文字人物：顏文字與 ASCII 藝術完全靠全形空格排版。逐字遮罩原本
    # 只保護半形空格與 tab，於是空白正規化收掉全形空格、`（　　　　）` 更被
    # 「空括號」規則整行刪掉，圖案全毀
    ('顏文字的全形空格要逐字保留',
     '以下是常見的顏文字範例，說明其構成方式。\n\n<pre>\n'
     '∧＿∧\n　（　・∀・）\n　（　　　　）\n　｜ ｜　|\n</pre>\n\n以上是顏文字的說明內容。',
     ['∧＿∧\n　（　・∀・）\n　（　　　　）\n　｜ ｜　|'],
     []),
    # 參宿四：半徑推導寫在 `<ref group="note">` 裡，十條公式中只有獨立成行的
    # 那條會變成 `$$…$$`，其餘同一行還有句號或粗體標記而成為行內 `$…$`
    ('註腳裡的行內公式也要撈出來',
     '参宿四的半徑計算方式如下所述。<ref group="note">\n開始計算的公式如下：\n\n'
     ':<math>{\\delta} = \\frac{d_B}{D_B}</math>\n\n代入後得到'
     '<math>d_B = 10.874</math>。\n</ref>以上是說明文字的內容。',
     ['\\frac{d_B}{D_B}', 'd_B = 10.874'],
     ['開始計算的公式如下']),
    ('一般引用照樣整段清掉',
     '這是一段正文內容說明。<ref>王明《中國史》，北京：中華書局，2001年。</ref>後續的正文仍然完整。',
     ['這是一段正文內容說明。後續的正文仍然完整。'],
     ['中華書局', '王明']),
    # HTML／Active Server Pages：範例程式本身就在示範註解寫法，
    # `<pre><!-- This is a comment --></pre>` 被註解清理整塊吃掉
    ('逐字區裡的註解是內容，不能當註解刪掉',
     'HTML 的註解寫法如下所示，瀏覽器不會顯示它。\n\n'
     '<pre><!-- This is a comment --></pre>\n\n以上是註解的範例說明文字。',
     ['```\n<!-- This is a comment -->\n```'],
     []),
    # 維基的 reflist 樣板註解本身就寫著 `<ref>` 與 `</ref>`。把文字依逐字區
    # 切段再逐段刪註解的話，跨越邊界的註解只被刪掉前半，尾巴留在正文裡
    ('跨越逐字區邊界的註解要整段刪掉',
     '這是條目的正文內容說明段落。\n\n== 參考資料 ==\n'
     '<!--See http://en.wikipedia.org/wiki/Wikipedia:Footnotes on how to generate '
     'footnotes using the <ref> and </ref> tags, and the template below-->\n\n'
     '這是最後一段的正文內容。',
     ['這是條目的正文內容說明段落。'],
     ['<ref>', 'template below', 'Footnotes']),
    # 但正文裡的註解仍要清掉，而且要在大括號配對之前——聖雄甘地的
    # `<!-- {Critique of political economy}} -->` 那個多出來的 }} 曾吃掉 33,000 字
    ('正文裡的註解照樣清掉',
     '正文開頭的段落內容如下。<!-- {Critique of political economy}} -->後續的正文仍然完整保留著。',
     ['正文開頭的段落內容如下。後續的正文仍然完整保留著。'],
     ['Critique', '}}']),
    # 一二·九運動：`{{cquote|…}}` 的引文只有 25 字，被「獨佔一行的模板要夠長
    # 才算成段文字」的門檻擋掉，整段引文消失。句末標點才是可靠的訊號
    ('短引文不會因為長度被丟掉',
     '他曾說過一段有名的話，流傳很廣。\n\n'
     '{{cquote|學問之道無他，求其放心而已矣，這是治學的根本方法。}}\n\n這句話影響深遠。',
     ['學問之道無他，求其放心而已矣，這是治學的根本方法。'],
     ['cquote']),
    # 博揚·波格丹諾維奇：`{{bd|1989年|4月18日}}` 只有生日，卻被加上破折號變成
    # 「1989年－4月18日」，看起來像已故。線上逐句比對才抓到的
    ('生卒模板只有生日時不加破折號',
     "'''某人'''（{{bd|1989年|4月18日}}）是一位籃球運動員，場上位置為小前鋒。",
     ['1989年4月18日'],
     ['1989年－4月18日']),
    ('生卒模板四個參數才是生－卒',
     "'''某人'''（{{bd|1923年|11月|2019年|10月21日}}）是一位物理學家，專長於量子力學。",
     ['1923年11月－2019年10月21日'],
     []),
    # 維護模板渲染成訊息方塊，不是條目內容；解析器函式算不出條件時取「成立」那一支
    ('解析器函式不留原文',
     '起初120個合數為：{{#ifexpr: 1 > 0 | 4、6、8、9 | 無}}⋯等等。\n\n'
     '每一個合數都可以寫成二個或多個質數的乘積形式。',
     ['4、6、8、9'],
     ['#ifexpr', '{{', '}}']),
    # 政府〈地圖〉整節只有兩張世界地圖：圖片位置標記在「決定保留哪些標題」時
    # 算成內容，純文字版稍後才拿掉標記，於是留下空標題（實測 670 個）。
    # 表格儲存格裡是圖片時同理，留下 `｜｜`（實測 9,887 筆）
    ('拿掉圖片後就空掉的章節與儲存格不留在純文字版',
     '這個條目說明政體的分類方式與世界分布情形。\n\n== 地圖 ==\n'
     '[[File:Forms of government.svg|thumb|依政體分類的世界地圖]]\n\n'
     '== 列表 ==\n\n{| class="wikitable"\n! 編號 !! 人物 !! 肖像 !! 生年\n|-\n'
     '| 1 || 朱德 || [[File:Zhu De.jpg|80px]] || 1886年\n|}\n\n'
     '以上是各國政體的整理結果。',
     ['1｜朱德｜1886年'],
     ['## 地圖', '｜｜']),
    # 黄金分割率：LaTeX 的巢狀括號會湊出假的 `}}`，資訊框的範圍配對從那裡
    # 斷掉，後面的欄位連同公式一起消失。逐字遮罩必須在抽取之前
    ('資訊框裡的公式不會被假的模板收尾切斷',
     '{{Infobox number\n| name=黃金比\n'
     '| 連分數=<math>1 + \\cfrac{1}{1 + \\cfrac{1}{1 + \\ddots}}</math>\n'
     '| algebraic=<math>\\frac{1 + \\sqrt{5}}{2}</math>\n}}\n'
     '黃金比例是一個無理數，約等於1.618。',
     ['連分數：$1 + \\cfrac{1}{1 + \\cfrac{1}{1 + \\ddots}}$',
      '$\\frac{1 + \\sqrt{5}}{2}$'],
     ['<math', '</math>']),
    # 亳州市：`|image = {{multiple image⏎ | total_width = 300⏎ …}}` 跨好幾行。
    # 深度若在換行時歸零，內部的版面參數會被當成頂層欄位，「基本資料」就冒出
    # `total_width：300`、`image_style：border:1px`
    ('巢狀跨行模板的版面參數不會變成事實',
     '{{Infobox China City\n|名稱 = 亳州市\n|image = {{multiple image\n'
     ' | total_width = 300\n | image_style = border:1px\n }}\n'
     '|leader_title2 = 市長\n|leader_name2 = 汪繼宏\n'
     '|carlicense = 皖S\n}}\n亳州市是安徽省下轄的地級市，位於安徽西北部。',
     ['皖S', '市長：汪繼宏'],
     ['total_width', 'image_style', 'multiple image', '導演：汪繼宏']),
    # 邢臺市：值裡有沒收尾的 `[[`，深度回不到 0 就會把整個資訊框吞成一個值
    ('資訊框沒收尾的括號只波及那一列',
     '{{Infobox China City\n|分類 = [[地級市\n|面積 = 114.505\n|省 = 河北省\n}}\n'
     '邢臺市是河北省下轄的地級市，位於河北南部。',
     ['面積：114.505', '省：河北省'],
     ['|面積 = ', '|省 = ']),
    # MediaWiki 的轉換器把 code/pre/syntaxhighlight/math 標成不轉換區。
    # 跟著轉的話會改掉程式語意：字串內容被改寫、`函数` 被台灣詞表換成 `函式`
    ('程式碼與公式不做繁簡轉換',
     '這段程式說明軟件的用法與網絡設定。\n\n'
     '<syntaxhighlight lang="csharp">\n// 构造函数\n'
     'Console.WriteLine("简体字符串");\n</syntaxhighlight>\n\n'
     '以上是範例，說明视频处理的流程。',
     ['// 构造函数', 'Console.WriteLine("简体字符串");', '軟體', '網路', '影片'],
     ['構造函式', '簡體字串']),
    # TeX：`{{markup|<syntaxhighlight>…</syntaxhighlight>|<math>…</math>}}` 並排
    # 展示原始碼與渲染結果，內容全在**位置參數**裡。保底規則只看具名參數時，
    # 整塊程式碼連同旁邊那條公式一起消失
    ('位置參數裡的程式碼與公式都要留下',
     '以二次方程為例，說明行內公式的寫法。\n\n{{markup|\n'
     '<syntaxhighlight lang="latex">\nThe quadratic formula is $x$\n\\bye\n'
     '</syntaxhighlight>\n|\n<math>x = 1</math>\n}}\n\n所有行内公式都須以美元符號包住。',
     ['```latex\nThe quadratic formula is $x$\n\\bye\n```', 'x = 1'],
     ['markup']),
    # 同一種欄位重複多列時維基在鍵尾加編號（`term_start2`）。去掉編號再查，
    # 但完整的鍵優先——`area_total_km2` 的 2 是單位不是編號
    ('編號變體的欄位要沿用同一個標籤',
     '{{Infobox officeholder\n|office2 = 中華人民共和國主席\n'
     '|term_start2 = 1988年4月8日\n|successor2 = 江澤民\n'
     '|area_total_km2 = 1997.47\n}}\n楊尚昆是中國政治人物，曾任國家主席。',
     ['職位：中華人民共和國主席', '任期開始：1988年4月8日', '繼任：江澤民',
      '總面積：1997.47'],
     ['term_start2', 'successor2', 'area_total_km']),
    # 杜鵑花屬：`{{臺灣植物|zh=杜鵑花屬}}是杜鵑花科的一個屬` —— 模板後面緊接著
    # 繫詞時，它渲染的就是主語，展開成空字串會讓句子沒有主詞
    ('繫詞前的模板要還原成主語',
     '{{臺灣植物| id =t0024701\n| binomial =Rhododendron\n| authority =L.| zh = 杜鵑花屬}}'
     '是杜鵑花科的一個屬，其下植物俗稱杜鵑花、映山紅。',
     ['杜鵑花屬是杜鵑花科的一個屬'],
     ['\n是杜鵑花科的一個屬']),
    ('頁首導航模板仍然不進正文',
     '{{about|其他用法|消歧義頁}}這是一篇正常的條目，開頭有頁首導航模板存在。',
     ['這是一篇正常的條目'],
     ['其他用法', '消歧義頁']),
    # 硅／軟體：`{{地区用词|cn=硅|tw=矽}}` 是中文維基處理兩岸用詞的常用模板，
    # 整個丟掉會讓條目第一句失去主語（「，是一種化學元素」）
    ('地區用詞模板要展開成該地區的用詞',
     '{{地区用词|cn=硅|tw=矽|start={{langx|en|Silicon}}|as=译}}，是一種化學元素，原子序數為14。',
     ['矽（Silicon），是一種化學元素'],
     ['，是一種化學元素，原子序數為14。\n']),
    # 矽／軟體／脫氧核糖核酸：`{{PAGENAME}}` 展開成空字串，條目第一句失去主語
    ('頁面名稱魔術字要換成條目標題',
     '{{PAGENAME}}是一種化學元素，化學符號為Si，原子序數為14，屬於類金屬。',
     ['測試是一種化學元素'],
     ['是一種化學元素，化學符號為Si，原子序數為14，屬於類金屬。\n']),
    # 編者也寫 `#redirect:目標`（冒號、沒方括號），只認 `[[` 會讓 282 篇漏進來
    ('沒有方括號的重定向也要丟掉',
     '#RAW##redirect:環鄱陽湖城市群',
     [],
     ['redirect', '環鄱陽湖城市群']),
    ('很短的條目也要收錄',
     '李某某是中國演員。',
     ['李某某是中國演員。'],
     []),
    # 湘西土家族苗族自治州：維基原始碼裡本來就有的落單圍欄
    ('落單圍欄會被清掉',
     '這一段是正文內容。\n```\n這一段也是正文內容。',
     ['這一段是正文內容。'],
     ['```']),
]


# omni 版專用檢查：(名稱, 輸入, 佔位符數, 圖片數, 圖說必須包含)
OMNI_CASES = [
    ('圖片佔位符在原本的位置',
     '[[File:Euclid.jpg|thumb|[[歐幾里得]]，古希臘數學家]]\n\n== 歷史 ==\n數學的歷史相當悠久。',
     1, 1, ['歐幾里得，古希臘數學家']),
    ('圖庫的每張圖都有佔位符',
     '<gallery>\nFile:G1.jpg|幾何圖形\nFile:G2.jpg|代數符號\n</gallery>',
     2, 2, ['幾何圖形', '代數符號']),
    ('沒有圖說的圖也算一張圖',
     '[[File:Deco.png|thumb|60px]]這一段是正文內容，長度足夠通過門檻檢查。',
     1, 1, ['']),
]


def run_omni():
    failed = 0
    for name, src, n_mark, n_img, captions in OMNI_CASES:
        rec = build_omni_record(src)
        problems = []
        if rec is None:
            problems.append('沒有產出記錄')
        else:
            marks = rec['text'].count(md.IMAGE_PLACEHOLDER)
            if marks != n_mark:
                problems.append(f'佔位符 {marks} 個，應為 {n_mark}')
            if len(rec['images']) != n_img:
                problems.append(f'圖片 {len(rec["images"])} 張，應為 {n_img}')
            if marks != len(rec['images']):
                problems.append('佔位符數與圖片數不一致')
            got = [im['caption'] for im in rec['images']]
            for want in captions:
                if want and want not in got:
                    problems.append(f'圖說缺少 {want!r}（實際 {got}）')
        if problems:
            failed += 1
            print(f'  ✗ [omni] {name}')
            for pr in problems:
                print(f'      {pr}')
        else:
            print(f'  ✓ [omni] {name}')
    return failed


def main():
    templates = template_store.load(PARSED)
    # CI 不會帶數 GB 的 parsed 產物；黃金案例只補它實際依賴的最小對照資料。
    # 本機有完整快取時仍使用 dump 抽出的真實表。
    if not templates:
        templates = {
            '〈': '-{zh-cn:《; zh-tw:〈; zh-hk:《;}-',
            '〉': '-{zh-cn:》; zh-tw:〉; zh-hk:》;}-',
        }
    wp.set_template_store(templates,
                          template_store.load_country_alias(PARSED),
                          template_store.load_maintenance(PARSED))
    import infobox_labels
    labels = infobox_labels.load(PARSED)
    if not any(labels):
        labels = (
            {
                'known for': '知名于',
                'workplaces': '机构',
                'builder': '建造者',
                'class': '分類',
            },
            {
                'subdivision name1': 'subdivision type1',
                'established date': 'established title',
                'leader name2': 'leader title2',
            },
            {}, {}, {},
        )
    wp.set_infobox_labels(*labels,
                          rendered=infobox_labels.load_rendered(PARSED))
    failed = 0
    for name, src, must, must_not in CASES:
        out = build(src)
        problems = [f'缺少 {w!r}' for w in must if w not in out]
        problems += [f'不該有 {w!r}' for w in must_not if w in out]
        if problems:
            failed += 1
            print(f'  ✗ {name}')
            for p in problems:
                print(f'      {p}')
            print(f'      實際輸出：{out!r}')
        else:
            print(f'  ✓ {name}')
    failed += run_omni()
    print(f'\n{"✓ 全部通過" if not failed else f"✗ {failed} 項失敗"}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
