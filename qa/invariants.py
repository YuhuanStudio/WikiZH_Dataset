"""
結構不變量檢查：找「照定義就不該發生」的情況

逐案抓缺陷會漏，因為每個修正都可能破壞別的地方。這支反過來——列出輸出**在
結構上必然成立**的性質，任何一條被違反就是 bug，不需要知道成因也能發現。

跟 validate_full 的差別：那支查「不該出現的字元」，這支查「不該出現的形狀」。
孤兒標題（`##` 之前沒有父層）就是這樣被抓到的——所有殘留字元檢查都是綠的，
但文檔的樹狀結構已經壞了。

用法：python qa/invariants.py output/tw [樣本數]
"""
import collections
import os
import random
import re
import sys

import pyarrow.parquet as pq
from validate_full import math_defects

OUT = sys.argv[1] if len(sys.argv) > 1 else 'output/tw'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 20260808
if N <= 0:
    raise SystemExit('樣本數必須是正整數')

_HEAD_RE = re.compile(r'^(#{2,6})\s*(.*)$')
_FENCE_RE = re.compile(r'(?s)```.*?```')


def sample(out_dir, n, seed):
    files = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir))
             if f.endswith('.parquet')]
    counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
    total = sum(counts)
    # Fixed seed is for reproducible QA sampling, not a security decision.
    rnd = random.Random(seed)  # nosec B311
    wanted = sorted(rnd.sample(range(total), min(n, total)))
    rows, base, w = [], 0, 0
    for path, cnt in zip(files, counts, strict=True):
        take = []
        while w < len(wanted) and wanted[w] < base + cnt:
            take.append(wanted[w] - base)
            w += 1
        if take:
            tbl = pq.read_table(path, columns=['title', 'text'])
            rows.extend(tbl.slice(i, 1).to_pylist()[0] for i in take)
        base += cnt
    return rows, total


# 每條回傳 None（成立）或違反／警示的說明字串。
# blocking=False 的檢查與 validate_full 一樣只是來源品質警示，
# 不可影響出貨門檻的退出碼。
CHECKS = []


def check(name, *, blocking=True):
    def deco(fn):
        CHECKS.append((name, fn, blocking))
        return fn
    return deco


@check('標題層級必須連續')
def _heading_tree(text, lines):
    """`####` 之前必須先出現過 `###`——跳階代表中間的父章節被丟了"""
    prev = 1
    for line in lines:
        m = _HEAD_RE.match(line)
        if not m:
            continue
        level = len(m.group(1))
        if level > prev + 1:
            return f'{prev}→{level} 「{m.group(2)[:30]}」'
        prev = level
    return None


@check('首個標題必須是二級')
def _first_heading(text, lines):
    for line in lines:
        m = _HEAD_RE.match(line)
        if m:
            return None if len(m.group(1)) == 2 else f'#{len(m.group(1))} 「{m.group(2)[:30]}」'
    return None


@check('第一行必須是條目標題')
def _title_line(text, lines):
    if not lines:
        return '空文檔'
    return None if lines[0].strip() and not lines[0].startswith('#') else repr(lines[0][:40])


@check('程式碼圍欄必須成對')
def _fence_pairs(text, lines):
    n = text.count('```')
    return None if n % 2 == 0 else f'``` 出現 {n} 次'


@check('公式括號／分隔符', blocking=False)
def _valid_math(text, lines):
    _has_tex, unbalanced, missing_delimiter = math_defects(text)
    if unbalanced:
        return 'LaTeX 大括號不平衡'
    if missing_delimiter:
        return 'LaTeX 指令在公式分隔符外'
    return None


# 原始內容可以合法含單一 `$`（貨幣、shell 提示符、BASIC 型別後綴）或單一反引號
#（字元表、轉寫符號）。它們不是必然成對的結構，不能列為不變量；公式改由上面的
# 正面 LaTeX 檢查驗證，程式區塊則由圍欄檢查驗證。


@check('不得有空章節')
def _empty_section(text, lines):
    """標題底下既沒有內文、也沒有更深層的子標題"""
    for i, line in enumerate(lines):
        m = _HEAD_RE.match(line)
        if not m:
            continue
        level = len(m.group(1))
        for nxt in lines[i + 1:]:
            nm = _HEAD_RE.match(nxt)
            if nm:
                if len(nm.group(1)) <= level:
                    return f'「{m.group(2)[:30]}」'
                break
            if nxt.strip():
                break
        else:
            return f'「{m.group(2)[:30]}」（文末）'
    return None


def main():
    rows, total = sample(OUT, N, SEED)
    if not rows:
        raise SystemExit(f'✗ {OUT} 沒有可檢查的 Parquet 記錄')
    print(f'=== 結構不變量 {OUT}：抽 {len(rows):,} / {total:,} 筆 ===\n')

    counts = collections.Counter()
    samples = collections.defaultdict(list)
    for r in rows:
        lines = r['text'].split('\n')
        for name, fn, _blocking in CHECKS:
            bad = fn(r['text'], lines)
            if bad:
                counts[name] += 1
                if len(samples[name]) < 5:
                    samples[name].append((r['title'], bad))

    n = len(rows)
    worst = 0
    print('【阻斷性結構不變量（必須為 0）】')
    for name, _fn, blocking in CHECKS:
        if not blocking:
            continue
        c = counts[name]
        worst = max(worst, c)
        flag = '' if c == 0 else '   ← 違反'
        print(f'  {name:<18}{c:>7,} ({c/n:7.4%}){flag}')

    print('\n【來源品質警示（不阻擋出貨）】')
    for name, _fn, blocking in CHECKS:
        if blocking:
            continue
        c = counts[name]
        flag = '' if c == 0 else '   ← 需核對來源'
        print(f'  {name:<18}{c:>7,} ({c/n:7.4%}){flag}')

    print()
    for name in sorted(counts, key=lambda k: -counts[k]):
        for title, bad in samples[name][:4]:
            print(f'  [{name}] {title}: {bad}')

    print('\n✓ 全部阻斷性不變量成立' if worst == 0 else
          '\n✗ 有違反的阻斷性不變量')
    sys.exit(1 if worst else 0)


if __name__ == '__main__':
    main()
