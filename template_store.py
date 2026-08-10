"""
從 dump 的 Template 命名空間建立「無參數模板」對照表

為什麼需要：`{{MLT}}`、`{{USA}}`、`{{JPN}}` 這類模板沒有參數，展開後才是
國名。解析時查不到內容就只能丟掉，於是《9月21日》的「馬爾他：獨立日」
變成「-：獨立日」——主語整個消失。實測 0.185% 的條目有這種斷句，而模板
出現在句子中間的還沒算進去（USA 在 20 萬頁裡出現 7,354 次）。

dump 本身就含 Template 命名空間，所以不必外部資料：先掃一趟把「沒有參數
且內容夠短」的模板收起來，解析時直接替換。

只收無參數模板（內容不含 `{{{`）。有參數的模板要真的實作 MediaWiki 的
參數展開與條件判斷，那是另一個量級的工程，且風險遠高於收益。
"""

import json
import os
import re

import bz2
from gensim.corpora.wikicorpus import extract_pages
from tqdm import tqdm

# 模板內容超過這個長度多半是排版框架（infobox、navbox），不是一段可讀文字
# 模板本體的長度上限。
#
# 這條看起來像「拿長度當判準」，但量過之後確定是對的：全量 104 萬個模板裡
# 超過 400 字的有 79,076 個（7.6%），逐一抽樣看，「無參數、非重定向」的長模板
# 清一色是版面元素——`Template:Wikipediasister`（2,818 字的姊妹計畫方塊）、
# `Template:PD-US`（授權標示 imbox）、`Template:GPL`。這些一旦收進對照表，
# 就會被原封不動塞進引用它們的條目正文。
#
# 換句話說：長度在這裡不是「內容價值」的代理，而是「這是不是版面方塊」的
# 代理，而且相關性很高。真正會渲染成正文的無參數模板都很短。
MAX_TEMPLATE_LEN = 400

_NOINCLUDE_RE = re.compile(r'(?is)<noinclude>.*?(?:</noinclude>|\Z)')
_INCLUDEONLY_RE = re.compile(r'(?is)</?includeonly>')
_ONLYINCLUDE_RE = re.compile(r'(?is)<onlyinclude>(.*?)</onlyinclude>')
_PARAM_RE = re.compile(r'\{\{\{')
# 參數／解析器函式區塊的開頭。它前面那一段是「無論有沒有給參數都會渲染出來」
# 的固定文字，對無參數呼叫（`{{X}}`）來說就是全部的輸出。
#
# 這條是為了救回被「含參數就不收」濾掉、但其實有固定前綴的模板。實例是
# 書名號：`Template:》` 的本體是純變體標記，收得下來；`Template:《` 多了一段
# `{{#switch:{{{2|}}}|…}}` 就整筆被丟掉，而 `Template:〈` 又是指向 `《` 的
# 重定向，於是一起陣亡。結果條目裡的書名號只剩右半邊（`桃花源` 出現 9 次）。
_PARAM_BLOCK_START_RE = re.compile(r'\{\{[#{]')


def _literal_prefix(body):
    """取出參數區塊之前那段固定文字；沒有可用的前綴就回傳空字串"""
    hit = _PARAM_BLOCK_START_RE.search(body)
    if not hit:
        return ''
    prefix = body[:hit.start()].strip()
    # 切點若落在 HTML 標籤中間，剩下的是一個沒有 `>` 的半截標籤——
    # `Template:Notelist` 的本體是 `<div class="notelist" style="list-style-type:
    # {{{1|decimal}}}">`，切完只剩 `<div class="notelist" style="list-style-type:`。
    # 下游清 HTML 的規則要求 `<…>` 成對，配不到就清不掉，於是那半截標籤原封不動
    # 流進正文（`交響詩`、`寒石散`、`衍慶宮淑妃` 的段落就這樣以它收尾）。
    # 全量有 495 個模板是這個形態。切到最後一個沒收尾的 `<` 之前。
    open_lt = prefix.rfind('<')
    if open_lt != -1 and prefix.find('>', open_lt) == -1:
        prefix = prefix[:open_lt].strip()
    # 前綴自己不能再含模板呼叫（收下來也展不開），也不該長到像是正文
    if not prefix or '{{' in prefix or len(prefix) > 200:
        return ''
    return prefix
_TEMPLATE_NS_RE = re.compile(r'^(?:Template|模板|样板|樣板)\s*:\s*(.+)$', re.I)
# 文件／沙盒／測試子頁不是會被引用的內容
_DOC_SUBPAGE_RE = re.compile(r'(?i)/(?:doc|sandbox|testcases|說明|说明|文档|文檔)$')
# 給編者看的錯誤提示，不是條目內容
# （`Template:=` 的內容就是「錯誤：已嵌入模板，請改成使用魔術字」）
_EDITOR_ERROR_RE = re.compile(r'\{\{\s*error\s*\||已嵌入模板|已停用|Expression error', re.I)

# 維護模板：`{{Or}}`、`{{Delete}}`、`{{Refimprove}}` 這些渲染成頁面上的訊息
# 方塊，是給編者看的，不是條目內容。原本靠一份手寫的 93 個名字擋，補不完也
# 沒有依據——實測 `{{Or|date=July 2011}}`（水星）、`{{Delete|多元}}`（新世界
# 發展）就漏在外面。
#
# 它們有結構訊號：本體一定引用訊息方塊家族（ambox 條目訊息框、imbox 圖片、
# cmbox 分類、tmbox 討論頁、ombox 其他、fmbox 頁首）。用這個從 dump 挖，
# 跟資訊框標籤表一樣是「維基自己怎麼渲染」而不是我們猜。
_MESSAGE_BOX_RE = re.compile(
    r'(?i)\{\{\s*(?:ambox|imbox|cmbox|tmbox|ombox|fmbox|mbox|asbox'
    r'|article[ _]issues|多個問題|多个问题|ambox/core)\b')
MAINT_NAME = 'maintenance.json'

STORE_NAME = 'templates.json'
ALIAS_NAME = 'country_alias.json'

# `Template:Country data 馬爾他` 這類頁面帶參數（`{{{1}}}`）所以不會進上面的
# 對照表，但國名就寫在它的 `alias`。旗幟模板 `{{flag|MLT}}` 靠它才能還原成
# 「馬爾他」，否則只會剩下代碼。
# 標題與重定向都可能用底線代替空白（`Country_data_United_States`）
# 模板重定向。dump 裡近六成的模板頁其實只是重定向，直接收下來的話
# `{{DPP}}` 會展開成 `#重定向 Template:DPP` 而出現在正文裡（實測 2.29%
# 的條目中招）。要跟著指到真正的內容，指不到就整筆丟掉。
_TMPL_REDIRECT_RE = re.compile(
    r'(?i)^\s*#\s*(?:REDIRECT|重定向|重新導向)\s*\[\[\s*:?\s*'
    r'(?:Template|模板|T)\s*:\s*([^\]|#]+)')
# 任何仍是重定向的內容都不能留——指不到目標就整筆丟掉，
# 否則 `#REDIRECT :Template:X`、`#redirect [[t:jpn]]` 會變成正文
_ANY_REDIRECT_RE = re.compile(r'(?i)^\s*#\s*(?:REDIRECT|重定向|重新導向)')

_COUNTRY_DATA_RE = re.compile(r'^(?:Template|模板)\s*:\s*Country[\s_]+data[\s_]+(.+)$', re.I)
_ALIAS_RE = re.compile(r'(?im)^\s*\|\s*alias\s*=\s*([^\n|}]{1,60})')
_REDIRECT_RE = re.compile(
    r'(?i)^\s*#\s*(?:REDIRECT|重定向|重新導向)\s*\[\[\s*(?:Template|模板)\s*:\s*'
    r'Country[\s_]+data[\s_]+([^\]|]+)')


def _norm(name):
    """模板名正規化：底線視同空白，統一小寫"""
    return name.replace('_', ' ').strip().lower()


def _clean_template_body(text):
    """取出模板實際會被展開的部分"""
    only = _ONLYINCLUDE_RE.search(text)
    if only:
        text = only.group(1)
    text = _NOINCLUDE_RE.sub('', text)
    text = _INCLUDEONLY_RE.sub('', text)
    return text.strip()


def _new_collector():
    """建立可供外部 dispatcher 共用的掃描狀態。"""
    return {'store': {}, 'alias': {}, 'redirects': {}, 'prefix_only': 0,
            'maint': set()}


def _collect_page(state, title, text):
    """收集單一 dump 頁；不做 I/O 與掃描後解析。"""
    store = state['store']
    alias = state['alias']
    redirects = state['redirects']

    cd = _COUNTRY_DATA_RE.match(title or '')
    if cd and text:
        key = _norm(cd.group(1))
        rd = _REDIRECT_RE.match(text)
        if rd:
            redirects[key] = _norm(rd.group(1))
        else:
            a = _ALIAS_RE.search(text)
            if a:
                alias[key] = a.group(1).strip()

    m = _TEMPLATE_NS_RE.match(title or '')
    if not m or not text:
        return
    name = _norm(m.group(1))
    if _DOC_SUBPAGE_RE.search(name):
        return
    # 維護模板要在長度／參數過濾**之前**認出來：它們幾乎都有參數（date=…），
    # 過濾之後就看不到了
    if _MESSAGE_BOX_RE.search(text):
        state['maint'].add(name)
    body = _clean_template_body(text)
    if not body or len(body) > MAX_TEMPLATE_LEN:
        return
    if _EDITOR_ERROR_RE.search(body):
        return
    if _PARAM_RE.search(body):          # 有參數的模板不收
        # 但參數區塊前面的固定文字是確定會渲染的，留下來總比整筆丟掉好
        body = _literal_prefix(body)
        if not body:
            return
        state['prefix_only'] += 1
    store[name] = body


def _finish_collector(state, out_dir):
    """解開重定向並寫出已掃完的狀態。"""
    store = state['store']
    alias = state['alias']
    redirects = state['redirects']
    prefix_only = state['prefix_only']

    # 解開模板重定向：`{{DPP}}` → `Template:民主進步黨` 的實際內容
    tmpl_redirects = {}
    for key, body in list(store.items()):
        if not _ANY_REDIRECT_RE.match(body):
            continue
        m = _TMPL_REDIRECT_RE.match(body)
        if m:
            tmpl_redirects[key] = _norm(m.group(1))
        del store[key]                      # 先移除，解得開才放回
    resolved = 0
    for src, dst in tmpl_redirects.items():
        cur, seen = dst, set()
        while cur in tmpl_redirects and cur not in seen:
            seen.add(cur)
            cur = tmpl_redirects[cur]
        if cur in store:
            store[src] = store[cur]
            resolved += 1
    print(f'  模板重定向 {len(tmpl_redirects):,} 筆，指到實際內容的 {resolved:,} 筆')

    # 解開國家資料的重定向（`country data MLT` → `country data Malta`）
    for src, dst in redirects.items():
        seen = set()
        cur = dst
        while cur in redirects and cur not in seen:
            seen.add(cur)
            cur = redirects[cur]
        if cur in alias:
            alias[src] = alias[cur]

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, STORE_NAME)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(store, f, ensure_ascii=False)
    mpath = os.path.join(out_dir, MAINT_NAME)
    with open(mpath, 'w', encoding='utf-8') as f:
        json.dump(sorted(state['maint']), f, ensure_ascii=False)
    print(f'✓ 收集到 {len(state["maint"]):,} 個維護模板（訊息方塊）→ {mpath}')
    apath = os.path.join(out_dir, ALIAS_NAME)
    with open(apath, 'w', encoding='utf-8') as f:
        json.dump(alias, f, ensure_ascii=False)
    print(f'✓ 收集到 {len(store):,} 個無參數模板 → {path}'
          f'（其中 {prefix_only:,} 筆是含參數模板的固定前綴）')
    print(f'✓ 收集到 {len(alias):,} 個國家名稱對照 → {apath}')
    return store


def build(xml_path, out_dir):
    """
    掃過 dump，把無參數模板寫成 JSON。

    Returns:
        dict: {模板名（小寫）: 展開內容}
    """
    opener = bz2.open if xml_path.endswith('.bz2') else open
    state = _new_collector()
    with opener(xml_path, 'rb') as f:
        for title, text, _pid in tqdm(extract_pages(f), desc='收集模板'):
            _collect_page(state, title, text)
    return _finish_collector(state, out_dir)


def load(out_dir):
    """讀取先前建立的對照表；沒有就回傳空 dict（解析仍可進行）"""
    return _read(os.path.join(out_dir, STORE_NAME))


def load_country_alias(out_dir):
    """讀取國家名稱對照表"""
    return _read(os.path.join(out_dir, ALIAS_NAME))


def load_maintenance(out_dir):
    """讀取維護模板名單（展開成空字串的那一批）"""
    data = _read(os.path.join(out_dir, MAINT_NAME))
    return set(data) if isinstance(data, list) else set(data or ())


def _read(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='建立無參數模板對照表')
    ap.add_argument('xml_path')
    ap.add_argument('out_dir')
    args = ap.parse_args()
    build(args.xml_path, args.out_dir)
