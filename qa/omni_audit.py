"""omni（圖文交錯）版校驗

這個版本多了一層必然成立的性質：**`<image>` 的個數必須等於 `images` 的長度**。
對不上就代表佔位符與圖片脫鉤，模型會把圖對到錯的位置——而純文字的殘留檢查
完全看不到這種缺陷。

順帶檢查 omni 版與純文字版的條目集合一致（同一份中間層產出，不該有落差）。

用法：python qa/omni_audit.py [tw|cn] [樣本數] [--output-root 目錄]
"""
import argparse
import collections
import os
import sys

import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_ROOT = os.environ.get(
    'OUTPUT_ROOT', os.path.join(REPO_ROOT, 'output'))

PLACEHOLDER = '<image>'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='校驗 omni 圖文交錯資料集')
    parser.add_argument('lang', nargs='?', choices=('tw', 'cn'), default='tw')
    parser.add_argument('sample_size', nargs='?', type=int, default=0,
                        help='檢查筆數；0 代表全部')
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT,
                        help='同時包含 tw/ 與 cn/ 的輸出根目錄')
    args = parser.parse_args(argv)
    if args.sample_size < 0:
        parser.error('樣本數不可為負數')
    return args


def parquet_files(path):
    if not os.path.isdir(path):
        raise SystemExit(f'✗ 找不到資料集目錄：{path}')
    files = [os.path.join(path, name) for name in sorted(os.listdir(path))
             if name.endswith('.parquet')]
    if not files:
        raise SystemExit(f'✗ {path} 沒有 parquet')
    return files


def iter_rows(path, columns):
    for parquet_path in parquet_files(path):
        tbl = pq.read_table(parquet_path, columns=columns)
        for row in tbl.to_pylist():
            yield row


def main(argv=None):
    args = parse_args(argv)
    output_root = os.path.abspath(args.output_root)
    omni_dir = os.path.join(output_root, args.lang, 'omni')
    text_dir = os.path.join(output_root, args.lang)
    counts = collections.Counter()
    samples = collections.defaultdict(list)
    total = 0
    omni_ids = set()
    with_images = 0
    image_total = 0

    for row in iter_rows(omni_dir, ['id', 'title', 'text', 'images']):
        total += 1
        if row['id'] in omni_ids:
            counts['omni 重複 ID'] += 1
        omni_ids.add(row['id'])
        marks = row['text'].count(PLACEHOLDER)
        images = row['images'] or []
        image_total += len(images)
        if images:
            with_images += 1
        if marks != len(images):
            counts['佔位符與圖片數不符'] += 1
            if len(samples['佔位符與圖片數不符']) < 5:
                samples['佔位符與圖片數不符'].append(
                    (row['title'], f'{marks} 個佔位符 vs {len(images)} 張圖'))
        for img in images:
            if not img.get('url', '').startswith('https://'):
                counts['圖片網址不合法'] += 1
                break
            if not img.get('file_name'):
                counts['缺少檔名'] += 1
                break
        if PLACEHOLDER in row['title']:
            counts['標題含佔位符'] += 1
        if args.sample_size and total >= args.sample_size:
            break

    if not total:
        raise SystemExit(f'✗ {omni_dir} 沒有可檢查的 Parquet 記錄')

    text_ids = set()
    text_duplicates = 0
    for row in iter_rows(text_dir, ['id']):
        if row['id'] in text_ids:
            text_duplicates += 1
        text_ids.add(row['id'])
    counts['純文字重複 ID'] = text_duplicates
    if not text_ids:
        raise SystemExit(f'✗ {text_dir} 沒有可比對的 Parquet 記錄')

    print(f'=== omni 版 {omni_dir}：{total:,} 筆 ===\n')
    print(f'  含圖片的條目      {with_images:>9,} ({with_images / max(total, 1):.1%})')
    print(f'  圖片總數          {image_total:>9,}')
    print(f'  平均每篇          {image_total / max(with_images, 1):>9.1f} 張（有圖的條目）\n')
    for name in ('佔位符與圖片數不符', '圖片網址不合法', '缺少檔名',
                 '標題含佔位符', 'omni 重複 ID', '純文字重複 ID'):
        c = counts[name]
        flag = '' if c == 0 else '   ← 需修正'
        print(f'  {name:<16}{c:>8,} ({c / max(total, 1):8.5%}){flag}')

    only_omni = omni_ids - text_ids
    only_text = set() if args.sample_size else text_ids - omni_ids
    print(f'\n  與純文字版的條目集合：只有 omni {len(only_omni)}，只有純文字 {len(only_text)}')

    print()
    for name, rows in samples.items():
        for title, detail in rows[:4]:
            print(f'  [{name}] {title}: {detail}')

    bad = sum(counts.values()) + len(only_omni) + len(only_text)
    print('\n✓ omni 版一致' if not bad else '\n✗ omni 版有問題')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
