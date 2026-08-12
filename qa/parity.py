"""繁簡結構對等檢查：同一篇條目，兩個版本只該差在字，不該差在結構

tw 與 cn 是同一份中間層產生的，ID 序列、標題層級、清單項數、表格列數、公式數、
圍欄數**必須完全一樣**。段落行數只列警示：MediaWiki 的地區變體可以提供整段
不同譯文，兩個分支本來就可能有不同的換行（例如動畫各話的兩地官方譯名表）。

這個檢查的價值在於它抓得到「兩邊都看起來正常」的缺陷：`{{legend|#ff0000|…}}`
把顏色碼洩漏進圖說時，殘留字元檢查全綠，是繁簡兩版差了一筆圖片記錄才露餡。

用法：python qa/parity.py [樣本數] [亂數種子] [--output-root 目錄]
"""
import argparse
import collections
import hashlib
import os
import random
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa.validate_full import _math_spans

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_ROOT = os.environ.get(
    'OUTPUT_ROOT', os.path.join(REPO_ROOT, 'output'))

_HEAD_RE = re.compile(r'(?m)^(#{2,6})[ \t]')
_ITEM_RE = re.compile(r'(?m)^- ')
_FENCE_RE = re.compile(r'(?s)```.*?```')
_INLINE_CODE_RE = re.compile(r'`[^`\n]*`')
_ADVISORY_KEYS = frozenset({'行數'})


def shape(text):
    """抽出與語言無關的結構特徵"""
    outside_fences = _FENCE_RE.sub('', text)
    return {
        '行數': text.count('\n'),
        '標題數': len(_HEAD_RE.findall(text)),
        '標題層級': ''.join(str(len(h)) for h in _HEAD_RE.findall(text)),
        '清單項數': len(_ITEM_RE.findall(text)),
        '表格欄分隔': text.count('｜'),
        '圍欄數': text.count('```'),
        # 只數真正配成對的語法；裸 `$` 可以是貨幣，裸反引號可以是字元表內容。
        '公式數': len(_math_spans(outside_fences)),
        '行內程式碼數': len(_INLINE_CODE_RE.findall(outside_fences)),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='檢查繁簡資料集的結構對等性')
    parser.add_argument('sample_size', nargs='?', type=int, default=20000,
                        help='結構抽樣筆數（預設 20000）')
    parser.add_argument('seed', nargs='?', type=int, default=20260808,
                        help='亂數種子')
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT,
                        help='同時包含 tw/ 與 cn/ 的輸出根目錄')
    args = parser.parse_args(argv)
    if args.sample_size <= 0:
        parser.error('樣本數必須是正整數')
    return args


def files(output_root, lang):
    root = os.path.join(output_root, lang)
    if not os.path.isdir(root):
        raise SystemExit(f'✗ 找不到 {lang} 輸出目錄：{root}')
    return [os.path.join(root, name) for name in sorted(os.listdir(root))
            if name.endswith('.parquet')]


def signature(output_root, lang):
    """以順序雜湊驗證完整 ID 序列，不把 148 萬篇正文全塞進記憶體。"""
    digest = hashlib.sha256()
    total = 0
    for path in files(output_root, lang):
        pf = pq.ParquetFile(path)
        total += pf.metadata.num_rows
        for batch in pf.iter_batches(columns=['id'], batch_size=50000):
            for page_id in batch.column(0).to_pylist():
                digest.update(str(page_id).encode('utf-8'))
                digest.update(b'\n')
    return total, digest.hexdigest()


def sample(output_root, lang, wanted):
    """依全域列號取樣；記憶體只跟樣本數成長，不跟整份資料集成長。"""
    rows, base, cursor = [], 0, 0
    for path in files(output_root, lang):
        pf = pq.ParquetFile(path)
        for group in range(pf.num_row_groups):
            count = pf.metadata.row_group(group).num_rows
            take = []
            while cursor < len(wanted) and wanted[cursor] < base + count:
                take.append(wanted[cursor] - base)
                cursor += 1
            if take:
                # 只載入命中樣本的 row group；舊版 `pq.read_table(path)` 會為了
                # 幾千筆樣本把整個 270 MB shard 展開，200k 樣本時 RSS 超過 4 GB。
                table = pf.read_row_group(
                    group, columns=['id', 'title', 'text'])
                rows.extend(table.take(pa.array(take)).to_pylist())
            base += count
    return rows


def main(argv=None):
    args = parse_args(argv)
    output_root = os.path.abspath(args.output_root)
    tw_total, tw_sig = signature(output_root, 'tw')
    cn_total, cn_sig = signature(output_root, 'cn')
    print(f'tw {tw_total:,} 篇，cn {cn_total:,} 篇')
    if not tw_total or not cn_total:
        raise SystemExit('✗ tw 或 cn 沒有可比對的 Parquet 記錄')
    sequence_bad = tw_total != cn_total or tw_sig != cn_sig
    if sequence_bad:
        print('  ✗ 條目 ID 的數量或順序不一致')
    else:
        print('  ✓ 條目 ID 序列完全一致')

    total = min(tw_total, cn_total)
    # Fixed seed is for reproducible QA sampling, not a security decision.
    rnd = random.Random(args.seed)  # nosec B311
    wanted = sorted(rnd.sample(range(total), min(args.sample_size, total)))
    tw = sample(output_root, 'tw', wanted)
    cn = sample(output_root, 'cn', wanted)
    print(f'\n比對 {len(wanted):,} 篇的結構\n')

    bad = collections.Counter()
    advisory = collections.Counter()
    samples = collections.defaultdict(list)
    for tw_row, cn_row in zip(tw, cn, strict=True):
        if tw_row['id'] != cn_row['id']:
            bad['樣本 ID'] += 1
            continue
        a, b = shape(tw_row['text']), shape(cn_row['text'])
        for key in a:
            if a[key] != b[key]:
                target = advisory if key in _ADVISORY_KEYS else bad
                target[key] += 1
                if len(samples[key]) < 5:
                    samples[key].append((tw_row['title'], a[key], b[key]))

    n = len(wanted)
    for key in shape(''):
        c = advisory[key] if key in _ADVISORY_KEYS else bad[key]
        flag = '' if c == 0 else (
            '   ← 地區譯文警示' if key in _ADVISORY_KEYS else '   ← 不對等')
        print(f'  {key:<12}{c:>7,} ({c / n:8.5%}){flag}')

    print()
    findings = bad | advisory
    for key in sorted(findings, key=lambda k: -findings[k]):
        for title, x, y in samples[key][:4]:
            print(f'  [{key}] {title}: tw={str(x)[:60]!r} cn={str(y)[:60]!r}')

    print('\n✓ 核心結構完全對等' if not bad else '\n✗ 有不對等的核心結構')
    sys.exit(1 if (bad or sequence_bad) else 0)


if __name__ == '__main__':
    main()
