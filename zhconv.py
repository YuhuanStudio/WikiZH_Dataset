"""
繁簡轉換：直接用維基百科自己的轉換表

為什麼不用 OpenCC：

我們的 ground truth 就是維基網站，而網站的繁體是 MediaWiki 的 LanguageConverter
用 `ZhConversion.php` 產生的。拿同一張表來轉，才可能跟網站一致。

實測（40 篇、17,998 個漢字，逐字對照網站 zh-tw 版）OpenCC 的偏差是 0.96%，
錯的清一色是台灣慣用詞與異體字偏好，而這些正是維基那張表涵蓋、OpenCC 沒有的：

    臺灣/台灣  發佈/發布  鏈接/連結  溼/濕  佔/占
    二噁英/戴奧辛  肖邦/蕭邦  莫臥兒/蒙兀兒  金酒/琴酒

跟 MediaWiki 一樣，把基礎表與地區用詞表**合併成一張表跑單一趟**最長匹配：
    zh-tw = zh2Hant（簡→繁）+ zh2TW（台灣慣用詞）
    zh-cn = zh2Hans（繁→簡）+ zh2CN（大陸慣用詞）

不能先做一次繁簡正規化再套地區用詞表。地區用詞表的鍵是哪一種字體並不一致
（zh2TW 的鍵多是簡體 `软件→軟體`，zh2CN 的鍵卻多是繁體 `軟體→软件`），先正規化
會讓 31% 的 zh2TW 條目與 57% 的 zh2CN 條目永遠查不到。以網站渲染結果為標準答案
實測（220 篇、繁體 145,828 字／簡體 283,190 字），與網站的逐字差異率（同一套量法，可作橫向比較）：

    繁體  先正規化 1.71%  →  合併單趟 1.44%
    簡體  先正規化 0.15%  →  合併單趟 0.09%

合併之後兩種字體的鍵同時在索引裡，最長匹配會先命中詞條再輪到單字規則，
所以繁體原文寫的 `軟體` 直接命中 zh2CN 而不會先被拆成 `软体`。
"""

import json
import os

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'zhconv.json')


class _Converter:
    """使用 trie 的最長匹配轉換器"""

    def __init__(self, tables):
        # 先合併成單一對照表，**後面的表覆蓋前面的**。
        #
        # 原本是把每張表的詞條各自append 進索引再依長度排序；同一個詞在兩張表
        # 裡都有時（`台球` 在 zh2Hant 是「檯球」、在 zh2TW 是「撞球」），長度相同，
        # 穩定排序讓先加入的基礎表永遠先命中，地區用詞表等於失效。實測衝突項
        # tw 26 個、cn 5 個：台球→檯球（該是撞球）、索马里→索馬里（該是索馬利亞）、
        # 综合征→綜合徵（該是症候群）。地區表是更精確的一層，必須蓋過基礎表。
        merged = {}
        for table in tables:
            merged.update(table)
        # 原本只依首字分組，每個字位還是要對同組內的每個詞條呼叫
        # startswith；「一」開頭的詞條就有上百個。trie 每個字只做一次
        # dict lookup，並記住途中最深的終點，語意仍是「最長鍵優先」。
        trie = {}
        for src, dst in merged.items():
            if not src:
                continue
            node = trie
            for ch in src:
                node = node.setdefault(ch, {})
            node[None] = dst
        self._trie = trie

    def convert(self, text):
        if not text:
            return text
        out = []
        i, n = 0, len(text)
        trie = self._trie
        missing = object()
        while i < n:
            node = trie.get(text[i])
            if node is None:
                out.append(text[i])
                i += 1
                continue

            j = i + 1
            best = node.get(None, missing)
            best_end = j
            while j < n:
                node = node.get(text[j])
                if node is None:
                    break
                j += 1
                value = node.get(None, missing)
                if value is not missing:
                    best = value
                    best_end = j
            if best is missing:
                out.append(text[i])
                i += 1
            else:
                out.append(best)
                i = best_end
        return ''.join(out)


_CACHE = {}


def _load():
    with open(_DATA, encoding='utf-8') as f:
        return json.load(f)


def get_converter(name):
    """取得轉換器（每個進程只建一次）"""
    conv = _CACHE.get(name)
    if conv is None:
        table_names = {
            'hans': ('zh2Hans',),
            'tw': ('zh2Hant', 'zh2TW'),
            'cn': ('zh2Hans', 'zh2CN'),
        }
        if name not in table_names:
            raise KeyError(name)
        t = _load()
        # 舊寫法的 dict literal 在只需要 tw 時仍會同時建三個
        # converter。這裡只建真正要放進 cache 的那一個。
        conv = _Converter([t[key] for key in table_names[name]])
        _CACHE[name] = conv
    return conv


def convert(text, lang):
    """轉換成指定變體（合併表、單一趟最長匹配，理由見模組說明）"""
    return get_converter(lang).convert(text)
