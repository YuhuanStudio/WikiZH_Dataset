r"""正則效能回歸測試：每次改清理規則都要跑，避免再次引入指數級回溯

全量跑過三次因為這個問題卡死：
  1. _NAVBOX_LINE_RE 的 (?:\s*\|\s*[^\n|]{1,40}){2,}
  2. _IMAGE_CAPTION_RE 的 [\w\s.()\-]+
  3. _CELL_ATTR_RE 的 (?:key=value\s*)+\|
共同特徵：群組加號 + 內部可變長度，遇到不匹配的長輸入就爆炸。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infobox_labels
import template_store
from image_extractor import find_image_tags, iter_gallery_bodies
from md_to_dataset import _drop_orphan_fences
from wiki_parser import (convert_tables, expand_inline_templates, extract_infobox_facts,
                         remove_file_links, remove_template_blocks, _keep_math_all,
                         set_infobox_labels, set_template_store)
from wiki_text import clean_block, convert_script, strip_leftover_markup

# 模板對照表會影響展開行為，壓力測試要在同樣條件下跑
PARSED = os.environ.get('WIKIZH_PARSED_DIR', 'parsed/202608')
set_template_store(template_store.load(PARSED),
                   template_store.load_country_alias(PARSED),
                   template_store.load_maintenance(PARSED))
set_infobox_labels(*infobox_labels.load(PARSED),
                   rendered=infobox_labels.load_rendered(PARSED))

# 預算隨輸入大小調整：這個測試要抓的是「指數級回溯」，不是線性成本。
# 4 萬字的單行本來就要 100ms 級，那是正常的。
def budget_ms(text):
    return max(100, len(text) / 100)
CASES = [
    # O(n²) 回歸守門：舊版 remove_template_blocks 每遇到一個未配對的 `{{`
    # 就 ''.join(out) 重建整個輸出，實測讓全量解析卡在單一頁面 55 分鐘。
    # 子頁面模板查表若用 base 而非完整名稱，`{{X/1}}` 會查到含 9 個 `/n`
    # 引用的母模板，每輪膨脹 9 倍——《Pokémon GO》因此卡住 worker 9 分鐘。
    ('互相引用的模板', '截至今日共有{{NUMBEROFPOKEMONGO}}種\n' +
     '\n'.join('* 第%d世代開放全{{NUMBEROFPOKEMONGO/%d}}種' % (k, k) for k in range(1, 10))),

    ('未配對模板大量出現',
     '\n'.join('{{壞掉的模板%d 一些內容文字' % k for k in range(3000))),

    ('多屬性無結尾管線', '| ' + 'a=1 b=2 c=3 d=4 e=5 f=6 g=7 h=8 i=9 j=0 ' * 20),
    ('長表格',          '{|\n' + '| style="background: #ccc" |值 || 25\n|-\n' * 200 + '|}'),
    ('多管線',          '中國歷史年表 | ' * 400),
    ('巢狀模板',        '{{a|' * 60 + 'x' + '}}' * 60),
    ('長圖片語法',      '[[File:' + 'a b c ' * 100 + '.jpg|說明]]正文'),
    # 每個開頭都掃到頁尾會是 O(n²)；圖片語法有 3,000 字的明確上限。
    ('大量未閉合圖片鏈結', '[[File:x.jpg|thumb|說明' * 4000 + '正文'),
    ('大量等號',        'abc=def|' * 300),
    ('長無中文行',      'style=width head=yes ' * 200),
    ('大量括號',        '（）' * 2000),
    ('超長單行',        '文字' * 20000),
    ('大量 ref',        '<ref name="x">內容</ref>' * 500),
    # 行尾一長串「配不到結尾」的 ref。判斷引導語時要剝掉行尾的來源標註，
    # 那條正則若寫成 `(?:…|…)+$`（外層量詞包住含 `.*?` 的選擇組）就會在這裡
    # 指數級回溯——實測 13 個 worker 各燒掉 3 小時 50 分 CPU、整批解析卡死。
    ('行尾未閉合 ref 串',  '以下列出：' + '<ref name="a">' * 400),
    ('引導語後多重 ref',   '包括以下物種：' + '<ref>x</ref>' * 300 + '\n{{list|a|b}}'),
    # 依原始碼換行把參數分列（_group_arg_rows）：分類群清單可以有上千個參數，
    # 而且分列結果要跟過濾後的參數逐一比對，別讓那次比對變成 O(n²)。
    ('超多參數的清單模板', '本科包括以下屬：\n{{common taxon list|italic=yes\n'
     + ''.join('|物種%d |Species%d |\n' % (k, k) for k in range(1500)) + '}}'),
    ('全部擠在同一行的清單', '包括：\n{{list|' + '|'.join('項目%d' % k for k in range(2000)) + '}}'),
    # 公式改成逐條掃描重組（_keep_math_all），輸出用串列累積而不是反覆 join
    ('大量相鄰公式',      '推導：' + '<math>x_%d+1</math>' % 1 * 1 +
     ''.join('<math>a_{%d}=b</math>' % k for k in range(2000))),
    # 落單圍欄清理要逐行掃描；原始碼裡連續出現大量 ``` 不能讓它退化
    ('大量落單圍欄',      '\n\n'.join(['正文段落。', '```'] * 1000)),
    # 繁簡轉換要跳過程式碼與公式，落單的圍欄／美元符號不能讓它每次掃到文末
    ('大量落單圍欄與美元符號', '\n\n'.join(['正文段落。', '```', '$$'] * 1000)),
    ('超長圍欄內容',      '```py\n' + 'x = 1\n' * 8000 + '```\n正文。'),
    # 資訊框參數改成依巢狀深度切、每個值就地展開模板。參數多的框（行政區、
    # 分類群）動輒上百個欄位，別讓「每個值跑一次展開」變成瓶頸。
    ('超多欄位的資訊框', '{{Infobox settlement\n'
     + ''.join('|欄位%d = {{lang|en|Value %d}}\n' % (k, k) for k in range(400)) + '}}\n正文'),
    ('資訊框裡的巨大值', '{{Infobox\n|備註 = ' + '一段說明文字。' * 3000 + '\n}}\n正文'),
    ('沒收尾的資訊框',   '{{Infobox\n' + '|欄位 = 值\n' * 2000),
    # 變體解析原本是「找到一個就用切片重建整篇」，k 個標記重建 k 次
    ('大量變體標記',      ('正文內容' * 20 + '-{zh-tw:賽局理論;zh-cn:博弈论}-') * 2000),
]

# 圖片抽取有自己的入口，不走上面那條清理鏈，另外測。
# `find_image_tags` 原本對每個沒收尾的 `[[File:` 都掃到頁尾，是 O(n²)。
IMAGE_CASES = [
    ('大量未閉合圖片',    '[[File:x.jpg|thumb|說明' * 2000 + '正文' * 100),
    ('大量正常圖片',      '[[File:x.jpg|thumb|說明]]正文。' * 2000),
    ('大量 gallery 行',   '<gallery>\n' + 'File:x.jpg|說明\n' * 3000 + '</gallery>'),
]

fail = 0
for name, text in CASES:
    t = time.perf_counter()
    for fn in (convert_tables, expand_inline_templates, remove_file_links,
               remove_template_blocks, strip_leftover_markup, clean_block,
               _keep_math_all, _drop_orphan_fences, extract_infobox_facts):
        fn(text)
    convert_script(text, 'tw')
    ms = (time.perf_counter() - t) * 1000
    ok = ms < budget_ms(text)
    fail += not ok
    print(f'  {name:<16}{ms:8.1f} ms / 預算 {budget_ms(text):6.0f} ms  {"OK" if ok else "✗ 超過"}')

for name, text in IMAGE_CASES:
    t = time.perf_counter()
    find_image_tags(text)
    list(iter_gallery_bodies(text))
    ms = (time.perf_counter() - t) * 1000
    ok = ms < budget_ms(text)
    fail += not ok
    print(f'  {name:<16}{ms:8.1f} ms / 預算 {budget_ms(text):6.0f} ms  {"OK" if ok else "✗ 超過"}')

print(f'\n{"✓ 全部通過" if not fail else f"✗ {fail} 項超時"}')
sys.exit(1 if fail else 0)
