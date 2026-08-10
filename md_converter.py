"""
維基百科 XML → 解析後頁面（中間層）

把 dump 解析成「語言中立」的頁面記錄，讓繁體／簡體兩個資料集共用同一次
XML 解析。輸出是分片 JSONL 而不是一條目一檔——理由見 page_store.py。

**多進程解析**。實測 8,000 頁：解壓與走訪 XML 佔 33%、解析佔 67%，而解析
是純 CPU 運算、每篇互不相干。單執行緒等於讓 20 核的機器只用一核去跑那 67%。
現在主進程只負責解壓與分派，解析交給 worker pool，解壓也因此與解析重疊。

worker 用 fork 啟動，90,730 筆的模板對照表靠 copy-on-write 共用，不必每個
進程各複製一份。
"""

import multiprocessing
import os

from tqdm import tqdm

import infobox_labels
import metadata_store
import template_store
import wikidata_store
from page_store import PageWriter, mark_complete
from wiki_parser import (WIKIParse, set_infobox_labels, set_template_store,
                         set_wikidata_values)

# 一次丟給 worker 的頁數。太小則 IPC 往返蓋過解析，太大則尾端負載不均。
CHUNK_SIZE = 24

_PARSER = None


_WIKIDATA = {}


def _init_worker(markdown):
    """每個 worker 建一份 parser（模板對照表已由 fork 繼承）"""
    global _PARSER
    _PARSER = WIKIParse.__new__(WIKIParse)
    _PARSER.markdown = markdown
    _PARSER.nl = '\n\n' if markdown else '\n'


# 診斷用：設 WIKIZH_TRACE=<目錄> 時，每個 worker 會把當前處理的條目寫下來。
# 單一頁面把 worker 卡住幾十分鐘時，這是唯一能知道是哪一篇的辦法。
_TRACE_DIR = os.environ.get('WIKIZH_TRACE')
_TRACE_PATH = None


def _parse_one(page):
    """
    解析單一頁面，回傳 (id, 文本, 圖片語法清單, 錯誤)。

    單一條目解析失敗不該中斷整批——維基上什麼奇怪內容都有，155 萬篇曾經跑到
    57% 才因為某篇的 `{{x|1¹¹=…}}` 整個中斷。失敗改成回傳錯誤資訊。
    """
    if _TRACE_DIR:
        global _TRACE_PATH
        if _TRACE_PATH is None:
            _TRACE_PATH = os.path.join(_TRACE_DIR, f'trace-{os.getpid()}.txt')
        with open(_TRACE_PATH, 'w', encoding='utf-8') as f:
            f.write((page[0] if page else '?') + '\n')
    # 這一篇條目可用的 Wikidata 數值
    set_wikidata_values(_WIKIDATA.get(page[0] if page else ''))
    try:
        page_id, text, images = _PARSER.parse(page)
    except Exception as e:
        return None, None, None, (page[0] if page else '?', f'{type(e).__name__}: {e}')
    if page_id is None:
        return None, None, None, None
    return page_id, text, images, None


class WIKIParse2Doc(WIKIParse):
    """解析 XML 並寫出分片 JSONL"""

    def __init__(self, input_file, output_dir, markdown=True):
        super(WIKIParse2Doc, self).__init__(input_file, markdown)
        self.output_dir = output_dir
        # 無參數模板對照表（`{{MLT}}` → 馬爾他）。
        #
        # 沒有就**現場建**，不能靜默跳過：只有 QA 腳本呼叫過 template_store.build，
        # 正式流程從來沒建過表。換一個月份第一次跑會得到「0 個模板、0 個國名」的
        # 降級輸出——`{{MLT}}` 不再變成「馬爾他」、國旗模板全部空掉——而且不會
        # 報錯，只在日誌留一行 0。
        global _WIKIDATA
        store = template_store.load(output_dir)
        alias = template_store.load_country_alias(output_dir)
        labels = infobox_labels.load(output_dir)
        if not store and not labels[0]:
            print('沒有模板與資訊框標籤表，現在合併建立（掃一次 dump）…',
                  flush=True)
            metadata_store.build(input_file, output_dir)
            store = template_store.load(output_dir)
            alias = template_store.load_country_alias(output_dir)
            labels = infobox_labels.load(output_dir)
        elif not store:
            print('沒有模板對照表，現在建立（掃一次 dump，約 10 分鐘）…', flush=True)
            template_store.build(input_file, output_dir)
            store = template_store.load(output_dir)
            alias = template_store.load_country_alias(output_dir)
        _WIKIDATA = wikidata_store.load(output_dir)
        if not _WIKIDATA:
            # Wikidata 值要打 API，不適合自動觸發；但一定要講清楚缺了什麼，
            # 否則正文會出現「INSEE市鎮編碼為。」這種沒有值的句子。
            print('⚠ 沒有 Wikidata 數值對照表（wikidata.json）。'
                  '需要的話先跑 wikidata_store.scan + fetch，否則帶 {{wikidata}} '
                  '的欄位會是空的。', flush=True)
        print(f'已載入 {len(store):,} 個無參數模板、{len(alias):,} 個國家名稱、'
              f'{len(_WIKIDATA):,} 篇 Wikidata 數值', flush=True)
        # 資訊框標籤表：模板頁自己寫的欄位名。沒有的話 75% 的事實行會掛著
        # 英文原鍵（`carlicense：冀X`），所以跟模板表一樣，缺了就現場建。
        if not labels[0]:
            print('沒有資訊框標籤表，現在建立（掃一次 dump，約 10 分鐘）…', flush=True)
            infobox_labels.build(input_file, output_dir)
            labels = infobox_labels.load(output_dir)
        print(f'已載入 {len(labels[0]):,} 個靜態標籤、{len(labels[1]):,} 個動態標籤、'
              f'{len(labels[2]):,} 個別名推導標籤、{len(labels[3]):,} 個模板的專屬標籤',
              flush=True)
        # 要在 fork 之前設好，worker 才能靠 copy-on-write 共用
        set_template_store(store, alias, template_store.load_maintenance(output_dir))
        set_infobox_labels(*labels, rendered=infobox_labels.load_rendered(output_dir))

    def run(self, num=None, workers=None):
        """
        Args:
            num: 只處理前 N 篇（測試用）
            workers: 解析進程數，預設 CPU 數減一（留一核給主進程解壓）

        Returns:
            int: 寫出的條目數
        """
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        print(f'解析進程數: {workers}', flush=True)
        iterator = tqdm(self.wiki_content, desc='Articles parsed: 0')
        failed = []

        ctx = multiprocessing.get_context('fork')
        pool = ctx.Pool(workers, initializer=_init_worker, initargs=(self.markdown,))
        try:
            with PageWriter(self.output_dir) as writer:
                for page_id, text, images, err in pool.imap_unordered(
                        _parse_one, iterator, chunksize=CHUNK_SIZE):
                    if err:
                        failed.append(err)
                        continue
                    if page_id is None:
                        continue

                    title = text.split('\n', 1)[0].lstrip('# ').strip()
                    writer.write(page_id, title, text, images)

                    if writer.total % 1000 == 0:
                        iterator.set_description(f'Articles parsed: {writer.total}')

                    if num is not None and writer.total >= num:
                        break

                total = writer.total
        finally:
            pool.terminate()
            pool.join()

        iterator.set_description(f'Articles parsed: {total}')

        # 失敗的條目要明確報出來，不能靜默吞掉
        if failed:
            print(f'\n⚠ {len(failed)} 篇解析失敗（已跳過）:')
            for title, err in failed[:10]:
                print(f'    {title[:40]}: {err}')
            with open(os.path.join(self.output_dir, 'parse_failures.txt'), 'w',
                      encoding='utf-8') as f:
                for title, err in failed:
                    f.write(f'{title}\t{err}\n')

        mark_complete(self.output_dir, total)
        return total
