"""
解析後頁面的中間層儲存

v1 把每個條目寫成一個 .md 檔，一次 dump 產生 155 萬個小檔案。這在 NTFS 上
是純粹的損耗：建檔佔掉轉換時間的大半、列目錄要好幾秒、刪除要好幾分鐘、
每個檔案還有 cluster 浪費。中間層的用途只是「讓 tw／cn 共用同一次 XML 解析」，
不需要一個條目一個檔案。

改成分片 JSONL（每片 5,000 篇，約 310 片）後：
- 檔案數從 1,557,509 降到約 310
- 順序讀寫，不再有百萬次 open/close
- 可以存結構化欄位，不必再用私有區字元夾帶語言變體

格式（每行一筆）：
    {"id": "100", "title": "农业", "text": "# 农业\\n\\n...", "images": ["File:…|thumb|圖說"]}
"""

import json
import os
import re

SHARD_SIZE = 5000
SHARD_RE = re.compile(r'^pages-\d{5}\.jsonl$')
DONE_MARKER = '.conversion_complete'
FAILURE_REPORT = 'parse_failures.txt'


class PageWriter:
    """把解析結果寫成分片 JSONL"""

    def __init__(self, out_dir, shard_size=SHARD_SIZE):
        self.out_dir = out_dir
        self.shard_size = shard_size
        os.makedirs(out_dir, exist_ok=True)
        # 開工前先撤掉完成標記，並清掉上一次留下的分片。
        #
        # 分片是從 0 號開始覆寫的，但編號較大的舊分片不會被蓋掉：這次產出的
        # 頁數只要比上次少（規則改了、dump 換了），多出來的舊分片就會被下一
        # 階段一起讀進去，混入上一版的內容。中斷重跑更糟——完成標記還在，
        # 半成品會被當成完整結果。
        for name in os.listdir(out_dir):
            if SHARD_RE.match(name) or name == DONE_MARKER:
                os.remove(os.path.join(out_dir, name))
        self._fh = None
        self._in_shard = 0
        self._shard_index = 0
        self.total = 0

    def _rotate(self):
        if self._fh:
            self._fh.close()
        path = os.path.join(self.out_dir, f'pages-{self._shard_index:05d}.jsonl')
        self._fh = open(path, 'w', encoding='utf-8', newline='\n')
        self._shard_index += 1
        self._in_shard = 0

    def write(self, page_id, title, text, images=None):
        if self._fh is None or self._in_shard >= self.shard_size:
            self._rotate()
        record = {'id': str(page_id), 'title': title, 'text': text}
        # 圖片語法（原始的 `File:…|thumb|圖說`），順序對應正文裡的位置標記。
        # omni 版本要用它把圖片插回原位；純文字版本忽略即可。
        if images:
            record['images'] = images
        self._fh.write(json.dumps(record, ensure_ascii=False) + '\n')
        self._in_shard += 1
        self.total += 1

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def shard_paths(page_dir):
    """依序回傳所有分片路徑"""
    if not os.path.isdir(page_dir):
        return []
    return [os.path.join(page_dir, n) for n in sorted(os.listdir(page_dir)) if SHARD_RE.match(n)]


def iter_pages(page_dir, shards=None):
    """走訪所有頁面記錄"""
    for path in (shards if shards is not None else shard_paths(page_dir)):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def count_pages(page_dir):
    return sum(1 for _ in iter_pages(page_dir))


def mark_complete(page_dir, total):
    """轉換成功後才寫入的標記，用來區分「已完成」與「跑到一半被中斷」"""
    with open(os.path.join(page_dir, DONE_MARKER), 'w', encoding='utf-8') as f:
        f.write(f'{total}\n')


def is_complete(page_dir):
    marker = os.path.join(page_dir, DONE_MARKER)
    if not os.path.isfile(marker) or not shard_paths(page_dir):
        return False
    # 舊版即使有單頁解析失敗仍會寫完成標記。保留失敗報告的同時，也要讓
    # 這類既有目錄不再被後續流程誤認成完整資料。
    failure_report = os.path.join(page_dir, FAILURE_REPORT)
    try:
        if os.path.isfile(failure_report) and os.path.getsize(failure_report) > 0:
            return False
    except OSError:
        # 無法確認失敗報告內容時採保守判定，避免沿用可能不完整的中間層。
        return False
    try:
        with open(marker, encoding='utf-8') as f:
            total = int(f.read().strip())
    except (OSError, ValueError):
        return False
    return total > 0
