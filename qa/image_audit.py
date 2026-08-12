"""圖片資料集校驗

正文有 validate_full 把關，圖片 JSONL 一直沒有——結果 653 筆圖說帶著私有區
字元、繁簡兩版內容完全相同（變體標記沒有依語言挑選就直接轉換）就這樣出貨了。

用法：python qa/image_audit.py output/tw/wiki_images_dataset.jsonl [tw|cn]
"""
import collections
import hashlib
import json
import re
import sys

PATH = sys.argv[1]
LANG = sys.argv[2] if len(sys.argv) > 2 else 'tw'

# 本專案的保留區（變體標記 + 逐字遮罩）與一般私有區，輸出裡都不該有
PUA_RE = re.compile(r'[-\U000f0000-\U0010fffd]')
MARKUP_RE = re.compile(r'\{\{|\}\}|\[\[|\]\]|<ref|</ref|&[a-z]{2,6};|__[A-Z]+__')
CTRL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
FIELDS = ('url', 'file_name', 'caption', 'alt', 'page', 'page_id', 'page_url', 'section')
IMAGE_URL_PREFIX = 'https://zh.wikipedia.org/wiki/Special:FilePath/'
PAGE_URL_PREFIX = 'https://zh.wikipedia.org/wiki/'
DEFECTS = ('私有區字元', '控制字元', '殘留標記', '欄位缺漏',
           '沒有任何文字', '網址不合法', '重複記錄', 'JSON 壞掉')


def main():
    counts = collections.Counter()
    samples = collections.defaultdict(list)
    seen = set()
    total = 0
    with open(PATH, encoding='utf-8') as fh:
        for line in fh:
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts['JSON 壞掉'] += 1
                continue
            canonical = json.dumps(
                row, ensure_ascii=False, sort_keys=True,
                separators=(',', ':')).encode('utf-8')
            digest = hashlib.blake2b(canonical, digest_size=16).digest()
            if digest in seen:
                counts['重複記錄'] += 1
            else:
                seen.add(digest)
            missing = [k for k in FIELDS if k not in row]
            if missing:
                counts['欄位缺漏'] += 1
            text = f"{row.get('caption', '')}\n{row.get('alt', '')}"
            for name, pattern in (('私有區字元', PUA_RE), ('殘留標記', MARKUP_RE),
                                  ('控制字元', CTRL_RE)):
                if pattern.search(text):
                    counts[name] += 1
                    if len(samples[name]) < 4:
                        samples[name].append((row.get('page', ''), text[:80]))
            if not row.get('caption') and not row.get('alt'):
                counts['沒有任何文字'] += 1
            if (not str(row.get('url', '')).startswith(IMAGE_URL_PREFIX)
                    or not str(row.get('page_url', '')).startswith(
                        PAGE_URL_PREFIX)):
                counts['網址不合法'] += 1

    print(f'=== 圖片資料集 {PATH}（lang={LANG}）：{total:,} 筆 ===\n')
    if total == 0:
        print('  ✗ 圖片資料集為空')
        sys.exit(1)
    for name in DEFECTS:
        c = counts[name]
        flag = '' if c == 0 else '   ← 需修正'
        print(f'  {name:<12}{c:>8,} ({c / max(total, 1):8.5%}){flag}')
    print()
    for name, rows in samples.items():
        for page, text in rows[:3]:
            print(f'  [{name}] {page}: {text!r}')
    sys.exit(1 if any(counts[name] for name in DEFECTS) else 0)


if __name__ == '__main__':
    main()
