"""
補回 Wikidata 取值模板的實際數值

維基百科條目用 `{{wikidata|property|P1082}}` 在渲染時去 Wikidata 抓值，
值本身不在 `pages-articles` dump 裡。不補的話，法國市鎮條目會變成：

    原始碼   INSEE市镇编码为{{wikidata|properties|P374}}。
    網站顯示 INSEE市鎮編碼為59151。
    不補     INSEE市鎮編碼為。          ← 事實遺失

不下載 Wikidata 的完整 dump（`wikidata-YYYYMMDD-all.json.bz2` 實測 95.5 GB）——
全量掃過只有 41,096 篇條目、92 種屬性用得到。改用 API 按條目批次查
（一次 50 篇，約 35 分鐘），結果快取成 JSON，跟模板對照表一樣是可重複使用
的離線資料，代價比整包 dump 小三個數量級。

兩個階段：
    scan()  掃 dump，找出「哪些條目需要哪些屬性」
    fetch() 批次查 API，寫成 {條目標題: {屬性: 值}}
"""

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request

import bz2
from gensim.corpora.wikicorpus import extract_pages
from tqdm import tqdm

STORE_NAME = 'wikidata.json'
PROGRESS_NAME = 'wikidata.progress.json'
API = 'https://www.wikidata.org/w/api.php'
UA = 'WikiZH-Dataset/2.0 (https://github.com/YuhuanStudio/WikiZH_Dataset)'
BATCH = 50

# `{{wikidata|property|raw|P1082}}`、`{{wikidata|properties|P374}}`、
# `{{wikidata|qualifier|P1082|P585}}` 等寫法，取出裡面所有 P 編號
_WIKIDATA_RE = re.compile(r'\{\{\s*wikidata\s*\|([^{}]{0,120})\}\}', re.I)
_PROP_RE = re.compile(r'\bP\d+\b')
_NAMESPACE_RE = re.compile(
    r'^(?:Template|Category|Portal|Help|Draft|MediaWiki|Wikipedia|WikiProject'
    r'|File|Image|Topic|Special|Talk|Module|模块|模組|分類|分类|模板|幫助|帮助'
    r'|維基百科|维基百科)\s*:', re.I)


def scan(xml_path, limit=None):
    """掃 dump，回傳 {條目標題: [屬性…]}"""
    opener = bz2.open if xml_path.endswith('.bz2') else open
    need = {}
    n = 0
    with opener(xml_path, 'rb') as f:
        for title, text, _pid in tqdm(extract_pages(f), desc='找 Wikidata 取值'):
            n += 1
            if limit and n > limit:
                break
            if not text or _NAMESPACE_RE.match(title or ''):
                continue
            props = set()
            for m in _WIKIDATA_RE.finditer(text):
                props.update(_PROP_RE.findall(m.group(1)))
            if props:
                need[title] = sorted(props)
    return need


def _api(params):
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(f'{API}?{query}', headers={'User-Agent': UA})
    # API 是本模組固定的 Wikidata HTTPS 端點，scheme 不由輸入決定。
    with urllib.request.urlopen(req, timeout=60) as r:  # nosec B310
        return json.load(r)


def _plain(datavalue):
    """把 Wikidata 的值轉成人看得懂的字串（只處理會出現在正文裡的型別）"""
    if not isinstance(datavalue, dict):
        return None
    value = datavalue.get('value')
    kind = datavalue.get('type')
    if kind == 'string':
        return str(value)
    if kind == 'quantity' and isinstance(value, dict):
        amount = str(value.get('amount', '')).lstrip('+')
        if amount.endswith('.0'):
            amount = amount[:-2]
        return amount
    if kind == 'time' and isinstance(value, dict):
        m = re.match(r'[+-](\d{4})-(\d{2})-(\d{2})', str(value.get('time', '')))
        if not m:
            return None
        year, month, day = m.groups()
        if month == '00':
            return f'{int(year)}年'
        if day == '00':
            return f'{int(year)}年{int(month)}月'
        return f'{int(year)}年{int(month)}月{int(day)}日'
    if kind == 'monolingualtext' and isinstance(value, dict):
        return value.get('text')
    return None


def _need_digest(need):
    payload = json.dumps(
        need, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _write_progress(path, digest, completed, store):
    """原子保存 API 進度；中斷後不必重查已完成批次。"""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({
            'need_sha256': digest,
            'completed_batches': sorted(completed),
            'store': store,
        }, f, ensure_ascii=False)
    os.replace(tmp, path)


def _load_progress(path, digest, valid_starts):
    if not os.path.exists(path):
        return set(), {}
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if data.get('need_sha256') != digest:
        raise RuntimeError(
            f'Wikidata 續跑檔與目前需求表不符：{path}')
    completed = set(data.get('completed_batches') or [])
    store = data.get('store') or {}
    if (not all(isinstance(i, int) and i in valid_starts for i in completed)
            or not isinstance(store, dict)):
        raise RuntimeError(f'Wikidata 續跑檔格式不合法：{path}')
    return completed, store


def fetch(need, out_dir, sleep=0.2, checkpoint_every=20):
    """批次查 API，寫出 {條目標題: {屬性: 值}}"""
    if checkpoint_every <= 0:
        raise ValueError('checkpoint_every 必須是正整數')
    os.makedirs(out_dir, exist_ok=True)
    titles = sorted(need)
    starts = list(range(0, len(titles), BATCH))
    digest = _need_digest(need)
    progress_path = os.path.join(out_dir, PROGRESS_NAME)
    completed, store = _load_progress(
        progress_path, digest, set(starts))
    if completed:
        print(f'↻ 接續 Wikidata：已完成 {len(completed):,}/{len(starts):,} 批',
              flush=True)
    failed_batches = []
    newly_completed = 0
    try:
        for i in tqdm(starts, desc='查 Wikidata'):
            if i in completed:
                continue
            chunk = titles[i:i + BATCH]
            data = None
            for attempt in range(4):
                try:
                    data = _api({
                        'action': 'wbgetentities', 'sites': 'zhwiki',
                        'titles': '|'.join(chunk), 'props': 'claims',
                        'format': 'json', 'formatversion': 2,
                    })
                    break
                except Exception as e:
                    print(f'  ⚠ 批次 {i} 第 {attempt + 1}/4 次失敗: '
                          f'{type(e).__name__}: {e}')
                    if attempt < 3:
                        time.sleep(2 ** attempt)
            if data is None:
                failed_batches.append(chunk)
                continue
            entities = data.get('entities') or {}
            # formatversion=2 用 list；舊格式用 dict。兩種都接。
            items = (entities if isinstance(entities, list)
                     else list(entities.values()))
            if len(items) != len(chunk):
                print(f'  ⚠ 批次 {i} 回傳 {len(items)} 筆，預期 {len(chunk)} 筆')
                failed_batches.append(chunk)
                continue
            # API 依查詢順序回傳，逐一對應
            for title, ent in zip(chunk, items, strict=True):
                if not isinstance(ent, dict):
                    continue
                claims = ent.get('claims') or {}
                values = {}
                for prop in need[title]:
                    candidates = sorted(
                        claims.get(prop, []),
                        key=lambda claim: {'preferred': 0, 'normal': 1,
                                           'deprecated': 2}.get(
                                               claim.get('rank'), 1),
                    )
                    for claim in candidates:
                        if claim.get('rank') == 'deprecated':
                            continue
                        before = len(values)
                        text = _plain(
                            (claim.get('mainsnak') or {}).get('datavalue'))
                        if text:
                            values[prop] = text
                        # 限定詞也要收。`{{wikidata|qualifier|P1082|P585}}`
                        # 要的是 P1082 聲明上的 P585，不是 P1082 本身。
                        for qprop, qsnaks in (
                                claim.get('qualifiers') or {}).items():
                            for qsnak in qsnaks[:1]:
                                qtext = _plain(qsnak.get('datavalue'))
                                if qtext:
                                    values[f'{prop}/{qprop}'] = qtext
                        if len(values) > before:
                            break
                if values:
                    store[title] = values
            completed.add(i)
            newly_completed += 1
            if newly_completed % checkpoint_every == 0:
                _write_progress(
                    progress_path, digest, completed, store)
            time.sleep(sleep)
    except BaseException:
        _write_progress(progress_path, digest, completed, store)
        raise

    if failed_batches:
        _write_progress(progress_path, digest, completed, store)
        raise RuntimeError(
            f'{len(failed_batches)} 個 Wikidata 批次在重試後仍失敗；'
            f'進度保存在 {progress_path}，未覆蓋正式快取')

    path = os.path.join(out_dir, STORE_NAME)
    partial = path + '.tmp'
    with open(partial, 'w', encoding='utf-8') as f:
        json.dump(store, f, ensure_ascii=False)
    os.replace(partial, path)
    try:
        os.remove(progress_path)
    except FileNotFoundError:
        pass
    try:
        os.remove(path + '.partial')
    except FileNotFoundError:
        pass
    print(f'✓ {len(store):,} 篇條目取得數值 → {path}')
    return store


def load(out_dir):
    path = os.path.join(out_dir, STORE_NAME)
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='補回 Wikidata 取值')
    ap.add_argument('xml_path')
    ap.add_argument('out_dir')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    need = scan(args.xml_path, args.limit)
    print(f'需要查詢的條目: {len(need):,}，屬性種類: '
          f'{len({p for v in need.values() for p in v})}')
    with open(os.path.join(args.out_dir, 'wikidata_need.json'), 'w',
              encoding='utf-8') as f:
        json.dump(need, f, ensure_ascii=False)
    fetch(need, args.out_dir)
