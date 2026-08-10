"""從維基百科自己的條目標題建一份中文詞表

用途是**斷詞的護欄**：台灣詞彙白名單是純子字串替換，`視頻→影片` 遇到
「電視頻道」就切成「電影片道」、`打印→列印` 遇到「攻打印度」就切成
「攻列印度」。要知道「這個位置是不是詞的邊界」，就需要一份詞表。

詞表不自己編，也不外掛第三方詞庫——維基百科的條目標題本身就是一份中文詞表，
而且和語料同源、同時代（`電視`、`印度`、`滑鼠`、`嚮導`、`標準`、`頻道` 全都是
條目）。這與專案其他離線對照表的來源一致：答案在維基自己身上。

只收純中文、2 到 6 個字的標題：更長的多半是句子式條目名（「中華人民共和國
行政區劃」），對邊界判斷沒有幫助又會拖慢查表；帶括號的消歧義標題
（「水星 (行星)」）也在過濾之列。

結果存成 `<page_dir>/title_words.json` 重複使用，與 `templates.json`、
`infobox_labels.json` 放在一起。
"""

import json
import os
import re

import zhconv

CACHE_NAME = 'title_words.json'
MIN_LEN = 2
MAX_LEN = 6

# 中間層每一列的開頭固定是 {"id": "...", "title": "..."，用正則取標題比
# 整列 json.loads 快一個量級——這一趟要讀 4 GB 以上，而我們只要標題。
_TITLE_RE = re.compile(r'^\{"id": "[^"]*", "title": "((?:[^"\\]|\\.)*)"')
_PURE_CJK_RE = re.compile(r'^[一-鿿]+$')


def _titles(page_dir):
    from page_store import shard_paths
    for path in shard_paths(page_dir):
        with open(path, encoding='utf-8') as f:
            for line in f:
                m = _TITLE_RE.match(line)
                if m:
                    yield json.loads(f'"{m.group(1)}"')


def build(page_dir, rebuild=False):
    """回傳條目標題構成的繁體詞表（結果會快取在 page_dir 下）"""
    cache = os.path.join(page_dir, CACHE_NAME)
    if not rebuild and os.path.exists(cache):
        with open(cache, encoding='utf-8') as f:
            return set(json.load(f))

    # 白名單作用在**已經做完字元轉換**的繁體文本上，所以詞表也要是繁體。
    to_tw = zhconv.get_converter('tw').convert
    words = set()
    for title in _titles(page_dir):
        if not (MIN_LEN <= len(title) <= MAX_LEN):
            continue
        if not _PURE_CJK_RE.match(title):
            continue
        words.add(to_tw(title))

    with open(cache, 'w', encoding='utf-8') as f:
        json.dump(sorted(words), f, ensure_ascii=False)
    return words


def load(page_dir=None):
    """取用詞表：指定中間層就用那一份，否則挑 `parsed/` 底下最新的快取

    圖片抽取是直接掃 dump，手上沒有中間層目錄，但詞表與那次解析同源即可。
    找不到就回空集合——白名單會退回沒有護欄的行為，並由呼叫端出聲。
    """
    import glob
    if page_dir:
        return build(page_dir)
    caches = sorted(glob.glob(os.path.join('parsed', '*', CACHE_NAME)))
    if not caches:
        return set()
    with open(caches[-1], encoding='utf-8') as f:
        return set(json.load(f))


if __name__ == '__main__':
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else 'parsed/202608'
    w = build(d, rebuild='--rebuild' in sys.argv)
    print(f'{len(w):,} 個詞 → {os.path.join(d, CACHE_NAME)}')
    for probe in ('電視', '影視', '印度', '滑鼠', '嚮導', '標準', '名稱', '頻道'):
        print(f'  {probe}: {"有" if probe in w else "沒有"}')
