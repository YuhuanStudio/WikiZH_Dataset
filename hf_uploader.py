"""
Hugging Face Hub 上傳模組

負責把本地生成好的資料集推送到 Hugging Face，並自動維護月份歸檔。

沿用現有 repo 的慣例：
- root 永遠放「最新版本」的資料檔（load_dataset 預設讀到的就是最新版）
- 上傳新版本前，先把 root 上的舊檔案歸檔到 `YYMM/` 月份資料夾
- 歸檔使用 HF 的 server-side copy（LFS 檔案不需重新上傳，秒完成）

上傳前會先跑一輪基本檢查（檔案完整性、格式、繁簡一致性、與上一版的大小落差），
任何一項不通過就中止上傳，避免把壞掉的資料推上公開 repo。
"""

import json
import os
import re
import sys
from datetime import datetime, timezone


# ============================================================
# repo 設定
# ============================================================

# 文字 Pretrain 資料集：每個語言一個 repo
PRETRAIN_REPOS = {
    'tw': 'yuhuanstudio/wikipedia-zh-tw',
    'cn': 'yuhuanstudio/wikipedia-zh',
}

# 圖文交錯（omni）資料集：正文帶 `<image>` 佔位符 + images 陣列。
# 跟純文字版分成不同 repo，因為 schema 不同（多一個 list-of-struct 欄位），
# 混在同一個 repo 會讓 `load_dataset` 的自動 schema 推斷失敗。
OMNI_REPOS = {
    'tw': 'yuhuanstudio/wikipedia-omni-zh-tw',
    'cn': 'yuhuanstudio/wikipedia-omni-zh',
}

# 圖片資料集：兩個語言共用同一個 repo，以檔名區分
IMAGE_REPO = 'yuhuanstudio/wikipedia-image-zh-tw'
IMAGE_REPO_FILENAME = {
    'tw': 'wiki_images_dataset.jsonl',
    'cn': 'wiki_images_dataset_CN.jsonl',
}

# repo root 上會被視為「資料檔」的檔名（歸檔與清理的判斷依據）
# 同時涵蓋舊版的 JSON 分段，這樣切換到 Parquet 時舊檔才會被正確歸檔並從 root 移除
PRETRAIN_FILE_RE = re.compile(r'^(?:wiki_pretrain_part\d+\.json|train-\d{5}-of-\d{5}\.parquet)$')
IMAGE_FILE_RE = re.compile(r'^wiki_images_dataset(_CN)?(_\d+)?\.jsonl$')

# ============================================================
# Dataset card（HF 上的 README）
#
# 卡片由程式產生，不手寫。手寫的話 GitHub 改了、HF 沒改，兩邊說法就會分岔——
# 這個專案的欄位、不變量、授權說明都是會演進的，靠人記得同步不可靠。
# ============================================================

_CARD_CHANGES = """
## `2607` 以前與 `2608` 起的差異

**整條處理流程重寫過**，這不是單純的月度刷新——資料的形狀變了。
若你正在用 `2607` 或更早的版本，升級前請先看這張表：

| | `2607` 以前 | `2608` 起 |
|---|---|---|
| 筆數／單位 | 371 萬列（**段落級**）| **148 萬篇**（文檔級，一條目一筆）|
| 檔案格式 | JSON 分段（`wiki_pretrain_part*.json`）| **Parquet 分片** |
| 章節結構 | 切碎後不存在 | 保留 `##` 階層 |
| 表格 | 整塊丟棄 | **轉成文字**（表格佔可見內容約 15%）|
| 程式碼 | 散文規則套上去，縮排與換行被破壞 | **逐字保留**，``` 圍欄 |
| 公式 | 丟棄 | **保留 LaTeX**，以 `$` 包住 |
| 資訊框 | 整塊丟棄 | 事實抽成 `## 基本資料`，欄位名取自維基的模板定義 |
| 繁簡轉換 | OpenCC `s2twp` | **維基官方轉換表** + 台灣慣用詞白名單（OpenCC 在百科語境誤轉率高：大力支持→大力支援、重整程序→重整程式）|
| 排版空格 | 有（pangu）| **移除**——不可逆的有損轉換，語料應保真 |
| 長度過濾 | 有 | **沒有**——一句話的條目也是完整知識 |
| 圖片 | 54.8 萬列（按檔名全域去重）、圖說有 10.87% 夾著未清的 wiki 標記、繁體檔有 43.8% 的條目名沒轉成繁體 | **91.1 萬列**（保留每張圖的每個使用語境）、圖說殘留標記 0.00%、**新增 `page_id`／`page_url`／`alt`／`section`** |
| 圖文交錯 | 無 | **新增** omni 版（`<image>` 佔位符 + `images` 陣列）|

### 內容缺失率（全量，非抽樣）

**資料集的核心指標是「原文的東西丟了多少」**——殘留標記可以靠把整段刪掉變成 0，
那不是乾淨。掃一次 dump，從原始碼取出可驗證的內容項目，回頭去兩版正文裡找。
只比兩版**都收錄**的 1,367,033 篇：

| 項目 | 樣本數 | `2607` 留住 | `2608` 留住 |
|---|---|---|---|
| 公式 | 91,552 | 0.0% | **99.2%** |
| 程式碼區塊 | 6,261 | 2.9% | **99.0%** |
| 表格儲存格 | 2,563,309 | 2.5% | **51.4%** |
| 清單項 | 5,522,474 | 30.9% | **70.2%** |
| 資訊框欄位值 | 6,721,977 | 5.1% | **30.7%** |
| 段落句子 | 5,564,852 | 59.0% | **74.8%** |
| **合計** | **20,470,425** | **26.4%** | **56.3%** |

連純散文，`2607` 也只留住 59%——四成的正文句子不見了。

`2608` 的 56.3% 不是滿分，但剩下的有明確歸屬：資訊框欄位值的探針含大量版面設定
（圖片檔名、尺寸、色碼），表格探針含樣式屬性（以人工挑過的探針量是 92.3% OK），
段落則是參考資料／外部連結等章節依設計整節不收。公式與程式碼的探針最乾淨，
99% 可視為保留率的上界參考。

條目收錄：1,419,539 篇 → **1,482,182 篇**（+62,643）。

### 實測對照（全量，非抽樣）

把公開的 `2607` 資料整份抓下來（2.9 GB），與 `2608` 的全部 148 萬篇套完全相同的
判準逐筆掃描：

| | `2607` 以前 | `2608` 起 | |
|---|---|---|---|
| 筆數 | 3,711,316 列 | 1,482,182 篇 | 單位從段落改成整篇 |
| **總字元** | **9.45 億** | **17.56 億** | **×1.86** |
| 每筆平均字元 | 255 | 1,185 | |

**同一份維基百科，新版多出 8.1 億字元**——多的不是重複，是原本被丟掉的表格、
資訊框事實、清單、程式碼與公式。

內容特徵的出現率（`2607` 幾乎全 0）：

| 特徵 | `2607` 以前 | `2608` 起 |
|---|---|---|
| 章節標題 | 0.00% | **84.21%** |
| 清單項 | 0.00% | **72.92%** |
| 資訊框事實 | 0.01% | **53.26%** |
| 表格列 | 0.02% | **16.23%** |
| LaTeX 公式 | 0.00% | **1.12%** |
| 程式碼圍欄 | 0.00% | **0.11%** |

排版與殘留（每百萬字元，已挖掉逐字區塊）：

| | `2607` 以前 | `2608` 起 | |
|---|---|---|---|
| 排版空格（pangu）| 36,851.26 | 1,276.03 | **−96.5%** |
| 私有區字元 | 0.38 | 0.00 | **−99.8%** |
| 表格空儲存格 | 0.03 | 0.00 | **清零** |
| 殘留標記合計 | 0.21 | ≤0.36 | 見下 |

殘留那一項要說清楚：`2607` 之所以低，是因為它把含有這些標記的內容整批刪掉了
——沒有表格、沒有資訊框、沒有程式碼，自然沒有殘留。`2608` 的殘留已逐筆歸因
（389 筆全部判定）：`<nowiki>` 字面值、真實內容長得像標記（`{{1,2,3}}` 是集合論、
`A<B> a;` 是 C++ 模板語法）、以及原文自己沒關標籤。

筆數從 371 萬「列」變成 148 萬「篇」不是資料變少——切分單位從段落改成整篇。
要段落的人可自行以 `\\n\\n` 切，反過來救不回來。

`2607` 以前的版本仍存放在本 repo 的 `YYMM/` 月份資料夾裡，需要的話可以直接讀舊路徑。
"""

_CARD_COMMON = """
## 這份資料怎麼來的

從維基百科官方 dump 解析，每月更新。原則是**只移除確定是標記的東西，
絕不刪除自然語言內容**：模板展開而非刪除、表格轉文字而非丟棄、程式碼與公式
逐字保留、圖庫圖說當內容。完整說明與品質數據見
[GitHub 專案](https://github.com/YuhuanStudio/WikiZH_Dataset)。

## 品質

每一輪都跑全量稽核，數字公開在專案 README：

| 檢查 | 結果 |
|---|---|
| 程式碼區塊與原始碼逐字相符 | 94.3% |
| 公式與原始碼逐字相符 | 99.3% |
| 繁簡兩版結構對等 | 20 萬篇中 2 篇不對等 |
| 空白／私有區字元／表格空儲存格等八類硬性缺陷 | 0 |

## 授權

內容來自維基百科，採 CC BY-SA 4.0。使用時請保留 `url`／`page_url` 欄位以符合署名要求。
"""


def _card_header(zh, tasks, glob):
    tasks_yaml = '\n'.join(f'- {t}' for t in tasks)
    return f"""---
license: cc-by-sa-4.0
language:
- {zh}
task_categories:
{tasks_yaml}
size_categories:
- 1M<n<10M
configs:
- config_name: default
  data_files:
  - split: train
    path: "{glob}"
---
"""


def _pretrain_card(lang, version, dump_date):
    name = '台灣正體中文' if lang == 'tw' else '简体中文'
    zh = 'zh-Hant' if lang == 'tw' else 'zh-Hans'
    return _card_header(zh, ['text-generation'], PRETRAIN_DATA_GLOB) + f"""
# 中文維基百科純文字資料集（{name}）

一個條目一筆記錄，保留章節結構，可直接用於語言模型預訓練。

> 📅 **目前版本**：`{version}`（維基百科 dump 日期：{format_dump_date(dump_date)}）

## 欄位

| 欄位 | 說明 |
|---|---|
| `id` | 維基百科條目 ID |
| `title` | 條目標題 |
| `url` | 條目原文網址（用未經繁簡轉換的原始標題組成，直接命中不轉址）|
| `text` | 條目全文。第一行為標題，章節以 `##` 標示，列表為 `- `，表格列以 `｜` 分隔，程式碼用 ``` 圍欄，公式保留 LaTeX 並以 `$` 包住 |

**不含圖說**——圖說說的是「這張圖是什麼」，不是條目本身的敘述，它在
[圖文配對版](https://huggingface.co/datasets/{IMAGE_REPO}) 與
[圖文交錯版](https://huggingface.co/datasets/{OMNI_REPOS[lang]}) 裡。

側邊資訊框的事實會抽出來接在文末的 `## 基本資料` 章節——生卒年、面積、成立年份
這些往往只寫在資訊框裡，正文不會重複。欄位名取自維基自己的模板定義，不是猜的。

**沒有長度過濾**：一句話的條目也是完整的知識內容，要篩隨時可以自己做，
被丟掉的救不回來。
""" + _CARD_CHANGES + _CARD_COMMON


def _omni_readme(lang, version, dump_date):
    name = '台灣正體中文' if lang == 'tw' else '简体中文'
    zh = 'zh-Hant' if lang == 'tw' else 'zh-Hans'
    return _card_header(zh, ['image-to-text', 'text-generation'],
                        PRETRAIN_DATA_GLOB) + f"""
# 中文維基百科圖文交錯資料集（{name}）

正文與圖片**交錯**排列：`text` 裡的每個 `<image>` 佔位符對應 `images` 陣列的
同一個位置，順序就是圖片在條目中原本出現的位置。格式對齊 MMC4／OBELICS 的慣例。

> 📅 **目前版本**：`{version}`（維基百科 dump 日期：{format_dump_date(dump_date)}）

## 欄位

| 欄位 | 說明 |
|---|---|
| `id` / `title` / `url` | 同純文字版，可用 `id` 對應 |
| `text` | 條目全文，圖片位置以 `<image>` 標示 |
| `images` | 圖片陣列：`url` / `file_name` / `caption` / `alt` |

**不變量**：`text.count("<image>") == len(images)`，全量實測 0 誤差。章節被丟棄時
它的圖片一併丟棄，兩者永遠對得上。

與[純文字版](https://huggingface.co/datasets/{PRETRAIN_REPOS[lang]})的差別只有
圖片處理：純文字版把標記整行移除、並清掉「拿掉圖片後就空掉」的章節；這一版
保留它們——只有地圖的一節對純文字模型是空標題，對多模態模型是有價值的圖文對。

> ⚠️ 圖片授權各不相同（CC BY-SA、公有領域，也有合理使用的非自由圖片）。
> 授權資訊不在 dump 裡，本資料集**不含授權欄位**，使用前請自行向 Commons 查證。
""" + _CARD_CHANGES + _CARD_COMMON


def _image_card(version, dump_date):
    """圖文配對資料集的卡片（tw/cn 共用一個 repo，以檔名區分）"""
    return _card_header('zh-Hant', ['image-to-text', 'text-to-image'],
                        'wiki_images_dataset*.jsonl') + f"""
# 中文維基百科圖文配對資料集

一圖一列，附圖說、替代文字、所在條目與章節。繁體（`wiki_images_dataset.jsonl`）
與簡體（`wiki_images_dataset_CN.jsonl`）各一份。

> 📅 **目前版本**：`{version}`（維基百科 dump 日期：{format_dump_date(dump_date)}）

## 欄位

| 欄位 | 說明 |
|---|---|
| `url` | 圖片檔案網址（`Special:FilePath`，會轉址到實際檔案）|
| `file_name` | 原始檔名 |
| `caption` | 圖說。與正文走**同一套**模板展開、殘留標記清理與繁簡轉換 |
| `alt` | 無障礙替代文字，約 2% 的圖片有，與圖說是不同的描述 |
| `page` / `page_id` | 圖片出現的條目與其 ID，可用 `page_id` 對應正文資料集的 `id` |
| `page_url` | 條目網址，CC BY-SA 署名依據 |
| `section` | 圖片出現在哪一節（前言為空字串），與正文的 `##` 標題可直接對應 |

**一次使用一列**：同一張圖出現在不同條目、配著不同圖說，正是圖文配對最有價值
的部分，因此不做全域去重。要一圖一列的人可自行以 `url` 去重。沒有圖說也沒有
`alt` 的純裝飾圖片會略過，不拿檔名充當描述。

唯一會去掉的是**同一條目裡整列完全相同**的重複（圖、圖說、`alt`、章節都一樣）：
路牌圖示之類的小圖會在同一節裡逐條重出，那些列一個字都沒多帶，只會讓裝飾性
圖示被過度加權。圖說或章節有任何不同就保留——那是同一張圖的另一個語境。

> ⚠️ 圖片授權各不相同（CC BY-SA、公有領域，也有合理使用的非自由圖片如商標、
> 專輯封面）。授權資訊不在 `pages-articles` dump 裡，本資料集**不含授權欄位**，
> 使用前請自行向 Commons／維基百科查證個別檔案。`caption` 文字採 CC BY-SA。
""" + _CARD_CHANGES + _CARD_COMMON


def dataset_card(repo_id, version, dump_date):
    """依 repo 決定要用哪張卡片；不是我們管理的 repo 回傳 None"""
    for lang, rid in PRETRAIN_REPOS.items():
        if rid == repo_id:
            return _pretrain_card(lang, version, dump_date)
    for lang, rid in OMNI_REPOS.items():
        if rid == repo_id:
            return _omni_readme(lang, version, dump_date)
    if repo_id == IMAGE_REPO:
        return _image_card(version, dump_date)
    return None


# README frontmatter 中資料檔的 glob（切成 Parquet 後要一併更新）
PRETRAIN_DATA_GLOB = 'train-*.parquet'

# 由本工具維護的版本記錄檔，讓下次上傳能準確知道 root 上是哪個月份的資料
VERSION_FILE = 'dataset_version.json'

# 月份資料夾格式：YYMM（例如 2608）
MONTH_DIR_RE = re.compile(r'^\d{4}$')

# README 中由本工具維護的版本區塊
README_MARK_START = '<!-- WIKIZH_VERSION:START -->'
README_MARK_END = '<!-- WIKIZH_VERSION:END -->'

# 與上一版本的大小落差超過這個比例就視為異常（可能是解析出錯導致資料缺漏）
SIZE_TOLERANCE = 0.30


# ============================================================
# 版本 / 日期工具
# ============================================================

def dump_date_to_version(dump_date):
    """dump 日期轉月份版本號：'20260804' -> '2608'"""
    dump_date = str(dump_date)
    if len(dump_date) != 8 or not dump_date.isdigit():
        raise ValueError(f"dump 日期格式錯誤（需為 YYYYMMDD）: {dump_date}")
    return dump_date[2:6]


def shift_month(version, delta):
    """月份版本號加減月份：('2608', -1) -> '2607'，('2612', 1) -> '2701'"""
    year = int(version[:2])
    month = int(version[2:])
    total = year * 12 + (month - 1) + delta
    return f"{total // 12 % 100:02d}{total % 12 + 1:02d}"


def format_dump_date(dump_date, style='slash'):
    """格式化 dump 日期供 README 使用"""
    dt = datetime.strptime(str(dump_date), '%Y%m%d')
    if style == 'slash':
        return f"{dt.year}/{dt.month}/{dt.day}"
    if style == 'zh':
        return f"{dt.year} 年 {dt.month} 月 {dt.day} 日"
    return dt.strftime('%Y-%m-%d')


def _human_size(num_bytes):
    """位元組轉人類可讀字串"""
    value = float(num_bytes)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if value < 1024 or unit == 'GB':
            return f"{value:.2f} {unit}"
        value /= 1024


def _resolve_token(token=None):
    """取得 HF token：參數 > 環境變數 > huggingface_hub 的本地登入快取"""
    return (
        token
        or os.environ.get('HF_TOKEN')
        or os.environ.get('HUGGINGFACE_HUB_TOKEN')
        or None  # None 時交由 huggingface_hub 讀取 `hf auth login` 的快取
    )


# ============================================================
# 上傳前檢查（CI 判定）
# ============================================================

class CheckResult:
    """檢查結果：收集所有錯誤與警告，最後一次回報"""

    def __init__(self, name):
        self.name = name
        self.infos = []
        self.errors = []
        self.warnings = []

    def info(self, msg):
        self.infos.append(msg)

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    @property
    def ok(self):
        return not self.errors

    def report(self):
        status = '✓ 通過' if self.ok else '✗ 未通過'
        print(f"  [{status}] {self.name}")
        for msg in self.infos:
            print(f"      {msg}")
        for msg in self.warnings:
            print(f"      ⚠ {msg}")
        for msg in self.errors:
            print(f"      ✗ {msg}")
        return self.ok


def _read_head(path, size=262144):
    with open(path, 'rb') as f:
        return f.read(size)


def _read_tail(path, size=4096):
    with open(path, 'rb') as f:
        file_size = os.path.getsize(path)
        f.seek(max(0, file_size - size))
        return f.read()


def _first_json_object(head_bytes):
    """從 JSON 陣列開頭的位元組中解析出第一筆物件"""
    text = head_bytes.decode('utf-8', errors='ignore')
    start = text.find('{')
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj
    except ValueError:
        return None


def _sample_records(path, limit=200):
    """從 pretrain JSON 陣列檔取樣前幾筆記錄（不需載入整個 500MB 檔案）"""
    records = []
    decoder = json.JSONDecoder()
    buffer = ''
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        buffer = f.read(4 * 1024 * 1024)
    idx = buffer.find('{')
    while idx != -1 and len(records) < limit:
        try:
            obj, end = decoder.raw_decode(buffer[idx:])
        except ValueError:
            break  # 讀到被截斷的記錄，取樣結束
        records.append(obj)
        idx = buffer.find('{', idx + end)
    return records


def _conversion_drift(texts, mode):
    """
    檢查文本的繁簡一致性，用專案自己的轉換表（不再依賴 opencc）。

    tw 資料再轉一次繁體應該幾乎不變（本來就是繁體），
    cn 資料再轉一次簡體也應該幾乎不變。若差異過大代表 lang 參數傳錯或轉換沒生效。

    這裡只用基礎表（zh2Hant／zh2Hans），不套地區用詞表——地區用詞本來就會
    造成差異，混進來會讓門檻失去意義。

    回傳字元層級的差異比例（0.0 ~ 1.0）。
    """
    import zhconv

    tables = zhconv._load()
    cc = zhconv._Converter([tables['zh2Hant' if mode == 's2t' else 'zh2Hans']])
    total = 0
    diff = 0
    for text in texts:
        if not text:
            continue
        converted = cc.convert(text)
        if len(converted) != len(text):
            # 長度不同時無法逐字比對，以整段視為差異
            total += len(text)
            diff += abs(len(converted) - len(text))
            continue
        total += len(text)
        diff += sum(1 for a, b in zip(text, converted) if a != b)
    if total == 0:
        return None
    return diff / total


PRETRAIN_COLUMNS = ('id', 'title', 'url', 'text')


def _size_drift_note(ratio, what):
    """體積相對上一版的變化：如實報告，不當作擋人的理由

    只憑「跟上一版差幾 %」分不出「資料變多」與「資料壞了」——這一版圖片多了
    71%（撈到更多圖、而且開始帶圖說），純文字則因為改用 Parquet 壓縮而小了
    42%，兩個都是對的。拿一個判不出對錯的訊號去擋上傳，結果只會是每次都加
    `--force-upload`，把真正判得出來的檢查（分片連續、欄位非空、schema、截斷、
    繁簡方向）一起關掉。所以體積只出警告，讓人看一眼；擋，留給判得準的項目。
    """
    if ratio > 0:
        cause = '撈到更多資料，或是同一批資料被重複寫出'
    else:
        cause = '壓縮方式改變，或是資料真的少了'
    return (f'{what}與上一版差異 {ratio:+.1%}，超過 ±{SIZE_TOLERANCE:.0%}'
            f'（可能是{cause}，請確認筆數與抽樣內容）')


def _check_lang(lang):
    """語言碼只有兩個值。傳錯就當場停，不要讓它默默走進另一邊的分支"""
    if lang not in ('tw', 'cn'):
        raise ValueError(f"lang 只能是 'tw' 或 'cn'，收到 {lang!r}")


def check_pretrain_files(files, lang, prev_total_size=None, columns=None, label=None):
    """檢查文字資料集的 Parquet 輸出

    `columns` 讓 omni 版沿用同一套檢查：它多一個 `images` 欄，schema 不同但
    其餘（分片連續、欄位非空、url 合法、平均長度、繁簡一致）判準完全一樣。

    `lang` 只能是 `tw`／`cn`，因為它決定繁簡一致性往哪個方向轉；顯示用的名稱
    走 `label`。曾經把 `'tw omni'` 當 lang 傳進來，`lang == 'tw'` 不成立就靜默
    落到簡體分支，於是繁體資料被判成「23.6% 不符合簡體」。
    """
    _check_lang(lang)
    columns = columns or PRETRAIN_COLUMNS
    import pyarrow.parquet as pq

    result = CheckResult(f"pretrain-{label or lang}（{len(files)} 個分片）")

    if not files:
        result.error('找不到任何 train-*.parquet 檔案')
        return result

    # 分片編號必須連續且與檔名宣告的總數一致，避免漏傳其中一片
    names = [os.path.basename(p) for p in sorted(files)]
    parsed = [re.match(r'train-(\d{5})-of-(\d{5})\.parquet$', n) for n in names]
    if not all(parsed):
        result.error(f'檔名不符合 train-NNNNN-of-NNNNN.parquet: {names}')
        return result
    declared = {int(m.group(2)) for m in parsed}
    indexes = sorted(int(m.group(1)) for m in parsed)
    if len(declared) != 1 or declared.pop() != len(files):
        result.error(f'分片總數與檔名宣告不一致: {names}')
    if indexes != list(range(len(files))):
        result.error(f'分片編號不連續: {indexes}')

    total_size = 0
    total_rows = 0
    total_chars = 0
    samples = []

    for path in sorted(files):
        name = os.path.basename(path)
        size = os.path.getsize(path)
        total_size += size

        try:
            pf = pq.ParquetFile(path)
        except Exception as e:
            result.error(f'{name} 不是合法的 Parquet 檔: {e}')
            continue

        cols = tuple(pf.schema_arrow.names)
        if cols != columns:
            result.error(f'{name} 欄位不符，預期 {columns}，實際 {cols}')
            continue

        rows = pf.metadata.num_rows
        total_rows += rows
        if rows == 0:
            result.error(f'{name} 沒有任何記錄')
            continue

        # 讀第一個 row group 就夠做內容抽查，不必載入整個分片
        batch = pf.read_row_group(0).to_pylist()
        for r in batch[:200]:
            if not r.get('text') or not r.get('title'):
                result.error(f'{name} 有記錄的 title/text 為空')
                break
            if not str(r.get('url', '')).startswith('https://zh.wikipedia.org/'):
                result.error(f"{name} 的 url 欄位不正確: {r.get('url')!r}")
                break
        samples.extend(batch[:200])
        total_chars += sum(len(r['text']) for r in batch)

    result.info(f"總大小: {_human_size(total_size)}｜{total_rows:,} 篇條目")

    # 文檔級資料不該再出現舊版那種百來字的碎片
    if samples:
        avg = sum(len(r['text']) for r in samples) / len(samples)
        result.info(f"抽樣平均長度: {avg:,.0f} 字")
        if avg < 300:
            result.error(f'抽樣平均只有 {avg:.0f} 字，可能又退回段落級切分')

        # pangu 排版空格應該已經移除
        blob = '\n'.join(r['text'] for r in samples[:500])
        spaced = len(re.findall(r'\d\s[一-鿿]', blob))
        per_k = spaced / max(len(blob), 1) * 1000
        result.info(f"數字後贅空格: 每千字 {per_k:.2f} 個")
        if per_k > 5:
            result.error(f'每千字有 {per_k:.1f} 個數字後空格，pangu 可能沒有關掉')

        drift = _conversion_drift([r['text'] for r in samples[:200]], 's2t' if lang == 'tw' else 't2s')
        if drift is None:
            result.info('繁簡一致性: 樣本無可判別字元，略過')
        elif drift > 0.05:
            expect = '繁體' if lang == 'tw' else '簡體'
            result.error(f'繁簡一致性檢查失敗：{drift:.1%} 的字元不符合{expect}，lang 參數可能傳錯')
        else:
            result.info(f"繁簡一致性: {drift:.2%} 差異（正常）")

    if prev_total_size:
        ratio = total_size / prev_total_size - 1
        result.info(f"與上一版比較: {ratio:+.1%}（上一版 {_human_size(prev_total_size)}）")
        if abs(ratio) > SIZE_TOLERANCE:
            result.warn(_size_drift_note(ratio, '總大小'))

    return result


def check_image_file(path, lang, prev_size=None):
    """檢查圖片資料集的輸出檔案"""
    _check_lang(lang)
    result = CheckResult(f"image-{lang}（{os.path.basename(path) if path else '無檔案'}）")

    if not path or not os.path.exists(path):
        result.error('找不到圖片 JSONL 檔案')
        return result

    size = os.path.getsize(path)
    result.info(f"檔案大小: {_human_size(size)}")
    if size < 1024 * 1024:
        result.error(f'檔案只有 {_human_size(size)}，明顯過小')
        return result

    required = ('url', 'file_name', 'caption', 'alt', 'page', 'page_id', 'page_url')
    samples = []
    line_count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line_count += 1
            if i < 200:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError as e:
                    result.error(f'第 {i + 1} 行不是合法 JSON: {e}')
                    break
                missing = [k for k in required if k not in obj]
                if missing:
                    result.error(f'第 {i + 1} 行缺少欄位: {missing}')
                    break
                samples.append(obj)

    result.info(f"資料筆數: {line_count:,}")

    # 最後一行必須是完整 JSON，確認檔案沒被寫到一半就中斷
    tail = _read_tail(path).decode('utf-8', errors='ignore').strip().split('\n')[-1]
    try:
        json.loads(tail)
    except ValueError:
        result.error('最後一行不是完整的 JSON，檔案可能被截斷')

    if samples:
        # 拿 caption／page 來判，不是 title——圖片記錄沒有 title 這個欄位，
        # 取到的永遠是空字串，於是這個檢查一路「無可判別字元」空轉。
        drift = _conversion_drift([f"{r.get('caption', '')}\n{r.get('page', '')}"
                                   for r in samples],
                                  's2t' if lang == 'tw' else 't2s')
        # 樣本裡沒有可判別的字元時回傳 None（例如標題全是英文或數字），
        # 那是「無從判斷」不是「檢查通過」，直接拿去比大小會 TypeError。
        if drift is None:
            result.info('繁簡一致性: 樣本無可判別字元，略過')
        elif drift > 0.05:
            expect = '繁體' if lang == 'tw' else '簡體'
            result.error(f'繁簡一致性檢查失敗：{drift:.1%} 的字元不符合{expect}')
        else:
            result.info(f"繁簡一致性: {drift:.2%} 差異（正常）")

    if prev_size:
        ratio = size / prev_size - 1
        result.info(f"與上一版比較: {ratio:+.1%}（上一版 {_human_size(prev_size)}）")
        if abs(ratio) > SIZE_TOLERANCE:
            result.warn(_size_drift_note(ratio, '檔案大小'))

    return result


# ============================================================
# 上傳器
# ============================================================

class HFUploader:
    """把資料集上傳到 Hugging Face，並維護月份歸檔"""

    def __init__(self, token=None, dry_run=False, refresh_cards=False):
        from huggingface_hub import HfApi

        self.dry_run = dry_run
        # True 時把 dataset card 整張換成程式產生的版本（`--refresh-cards`）
        self.refresh_cards = refresh_cards
        self.token = _resolve_token(token)
        self.api = HfApi(token=self.token)
        self._tree_cache = {}

        who = self.api.whoami()
        print(f"HF 帳號: {who.get('name')}")
        if dry_run:
            print("※ dry-run 模式：只做檢查與規劃，不會實際寫入 HF")

    # ---------- repo 內容查詢 ----------

    def ensure_repo(self, repo_id):
        """repo 不存在就建一個，回傳「這是新建的嗎」

        新增一種資料集形態時（這次的 omni）repo 還不存在，`list_repo_tree`
        會拋 404 並中止整個上傳——但「還沒有這個 repo」不是錯誤，是第一次
        上傳的正常狀態。新 repo 沒有舊版本可歸檔，樹當成空的。
        """
        from huggingface_hub.errors import RepositoryNotFoundError
        try:
            if self.api.repo_exists(repo_id, repo_type='dataset'):
                return False
        except RepositoryNotFoundError:
            pass
        if self.dry_run:
            print(f"  （dry-run）repo 不存在，實際執行時會建立 {repo_id}")
        else:
            self.api.create_repo(repo_id, repo_type='dataset', exist_ok=True)
            print(f"  建立新 repo: {repo_id}")
        self._tree_cache[repo_id] = []
        return True

    def _tree(self, repo_id, refresh=False):
        """列出 repo root 的內容（含檔案大小），結果會快取

        repo 不存在時回空清單而不是拋例外：呼叫端問的是「上面有什麼」，
        「什麼都沒有」是這個問題的合法答案。
        """
        from huggingface_hub.errors import RepositoryNotFoundError
        if refresh or repo_id not in self._tree_cache:
            try:
                self._tree_cache[repo_id] = list(
                    self.api.list_repo_tree(repo_id, repo_type='dataset',
                                            recursive=False)
                )
            except RepositoryNotFoundError:
                self._tree_cache[repo_id] = []
        return self._tree_cache[repo_id]

    def _root_data_files(self, repo_id, pattern):
        """root 上符合資料檔命名規則的檔案 {檔名: 大小}"""
        from huggingface_hub.hf_api import RepoFile

        return {
            item.path: item.size
            for item in self._tree(repo_id)
            if isinstance(item, RepoFile) and pattern.match(item.path)
        }

    def _month_dirs(self, repo_id):
        """repo 上已有的月份資料夾"""
        from huggingface_hub.hf_api import RepoFolder

        return sorted(
            item.path for item in self._tree(repo_id)
            if isinstance(item, RepoFolder) and MONTH_DIR_RE.match(item.path)
        )

    def _read_repo_file(self, repo_id, path):
        """讀取 repo 上的文字檔，不存在時回傳 None"""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (EntryNotFoundError,
                                            RepositoryNotFoundError)

        # repo 本身還不存在（第一次上傳這種形態）就別去下載了：
        # `force_download=True` 會把底下的 404 包成「Force download failed」，
        # 型別對不上，於是正常狀態被印成錯誤。
        try:
            if not self.api.repo_exists(repo_id, repo_type='dataset'):
                return None
        except RepositoryNotFoundError:
            return None

        try:
            local = hf_hub_download(
                repo_id, path, repo_type='dataset', token=self.token,
                force_download=True,  # 避免讀到過期的本地快取
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            return None
        except Exception as e:
            print(f"  ⚠ 讀取 {path} 失敗: {e}")
            return None
        with open(local, 'r', encoding='utf-8') as f:
            return f.read()

    # ---------- 月份自動判斷 ----------

    def detect_archive_version(self, repo_id, new_version, pattern):
        """
        判斷 root 上現有資料屬於哪個月份，也就是這次的歸檔目標。

        依序嘗試（越前面越可靠）：
        1. 由本工具維護的 dataset_version.json
        2. 已存在的月份資料夾最大值 + 1 個月
        3. root 資料檔最後一次 commit 的日期
        4. 新版本的前一個月
        """
        # 1. 版本記錄檔
        raw = self._read_repo_file(repo_id, VERSION_FILE)
        if raw:
            try:
                version = json.loads(raw).get('version')
                if version and MONTH_DIR_RE.match(str(version)):
                    print(f"  月份判斷: {version}（來源: {VERSION_FILE}）")
                    return str(version)
            except ValueError:
                pass

        # 2. 已有的月份資料夾往後推一個月
        months = self._month_dirs(repo_id)
        if months:
            version = shift_month(months[-1], 1)
            print(f"  月份判斷: {version}（來源: 最新歸檔資料夾 {months[-1]} + 1 個月）")
            return version

        # 3. root 資料檔的最後 commit 日期
        try:
            commits = self.api.list_repo_commits(repo_id, repo_type='dataset')
            if commits:
                created = commits[0].created_at
                version = f"{created.year % 100:02d}{created.month:02d}"
                print(f"  月份判斷: {version}（來源: 最後 commit 日期 {created:%Y-%m-%d}）")
                return version
        except Exception as e:
            print(f"  ⚠ 讀取 commit 記錄失敗: {e}")

        # 4. 保底：新版本的前一個月
        version = shift_month(new_version, -1)
        print(f"  月份判斷: {version}（來源: 新版本前一個月，保底推算）")
        return version

    # ---------- 歸檔與上傳 ----------

    def archive_root(self, repo_id, archive_version, pattern):
        """把 root 上的資料檔 server-side copy 到 YYMM/ 月份資料夾"""
        from huggingface_hub import CommitOperationCopy

        files = self._root_data_files(repo_id, pattern)
        if not files:
            print("  root 沒有可歸檔的資料檔，略過歸檔")
            return True

        if archive_version in self._month_dirs(repo_id):
            print(f"  ⚠ {archive_version}/ 已存在，略過歸檔（不覆蓋既有歷史版本）")
            return True

        total = sum(files.values())
        print(f"  歸檔 {len(files)} 個檔案（{_human_size(total)}）到 {archive_version}/")
        for name in sorted(files):
            print(f"    {name} → {archive_version}/{name}")

        if self.dry_run:
            return True

        ops = [
            CommitOperationCopy(src_path_in_repo=name, path_in_repo=f'{archive_version}/{name}')
            for name in sorted(files)
        ]
        self.api.create_commit(
            repo_id, repo_type='dataset', operations=ops,
            commit_message=f'chore: archive {archive_version} dataset files',
        )
        self._tree_cache.pop(repo_id, None)
        print(f"  ✓ 已歸檔到 {archive_version}/")
        return True

    def _readme_block(self, repo_id, version, dump_date):
        """組出 README 中的版本說明區塊"""
        months = [m for m in self._month_dirs(repo_id) if m != version]
        history = '、'.join(f'`{m}/`' for m in reversed(months[-3:])) if months else ''
        lines = [
            README_MARK_START,
            f"> 📅 **目前版本**：`{version}`（維基百科 dump 日期：{format_dump_date(dump_date)}）",
        ]
        if history:
            lines.append(f"> 歷史版本存放於 {history} 等月份資料夾。")
        lines.append(README_MARK_END)
        return '\n'.join(lines)

    def _retarget_readme_data_files(self, content):
        """
        把 README frontmatter 裡的資料檔清單改指向 Parquet。

        舊版列了 wiki_pretrain_part1~6.json 六個檔名，換成 Parquet 之後
        必須改成 glob，否則 load_dataset() 會找不到檔案。
        """
        if PRETRAIN_DATA_GLOB in content:
            return content  # 已經指向 Parquet

        # configs > data_files > path 底下的檔名清單（可能是 - "x.json" 或 - x.json）
        pattern = re.compile(r'(\n\s*path:[ \t]*\n)(?:[ \t]*-[ \t]*["\']?[\w./*-]+["\']?[ \t]*\n)+')
        new_content, count = pattern.subn(f'\\1    - "{PRETRAIN_DATA_GLOB}"\n', content, count=1)
        if count:
            return new_content

        # 單行寫法 path: "xxx.json"
        pattern2 = re.compile(r'(\n\s*path:[ \t]*)["\']?[\w./*-]+["\']?[ \t]*(?=\n)')
        new_content, count = pattern2.subn(f'\\1"{PRETRAIN_DATA_GLOB}"', content, count=1)
        if count:
            return new_content

        print(f"    ⚠ README frontmatter 的 data_files 無法自動改寫，請手動改成 {PRETRAIN_DATA_GLOB}")
        return content

    # 文檔級 Parquet 的欄位說明，用來取代 README 中舊版（段落級）的敘述
    PRETRAIN_SCHEMA_BLOCK = '''```json
{
  "id": "100",
  "title": "農業",
  "url": "https://zh.wikipedia.org/wiki/農業",
  "text": "農業\\n\\n農業屬於第一級產業，包括作物種植、畜牧、漁業養殖、林業等活動…\\n\\n## 定義\\n\\n根據東漢時期《說文解字》…"
}
```

'''

    PRETRAIN_FIELDS_BLOCK = '''- `id`: (string) 維基百科條目 ID
- `title`: (string) 條目標題
- `url`: (string) 條目原文網址（CC BY-SA 署名用）
- `text`: (string) 條目全文，第一行為標題，章節以 Markdown `##` 標示

'''

    def _replace_section(self, content, keyword, block):
        """把 README 中某個 `## 標題` 到下一個 `## ` 之間的內容整段換掉"""
        pattern = re.compile(
            r'(^##[^\n]*' + re.escape(keyword) + r'[^\n]*\n+)(?:.*?)(?=^## |\Z)',
            re.MULTILINE | re.DOTALL,
        )
        new_content, count = pattern.subn(lambda m: m.group(1) + block, content, count=1)
        if not count:
            print(f"    ⚠ README 找不到「{keyword}」章節，請手動更新欄位說明")
        return new_content

    def _update_readme_schema(self, content):
        """更新 README 的資料集結構與欄位說明，讓它符合文檔級 Parquet"""
        content = self._replace_section(content, '資料集結構', self.PRETRAIN_SCHEMA_BLOCK)
        content = self._replace_section(content, '欄位說明', self.PRETRAIN_FIELDS_BLOCK)
        return content

    def _render_readme(self, content, repo_id, version, dump_date, is_image):
        """更新 README 的 dump 日期與版本區塊，回傳新內容（沒有變化時回傳 None）"""
        original = content
        block = self._readme_block(repo_id, version, dump_date)

        if README_MARK_START in content and README_MARK_END in content:
            content = re.sub(
                re.escape(README_MARK_START) + r'.*?' + re.escape(README_MARK_END),
                lambda _: block, content, flags=re.DOTALL,
            )
        else:
            # 第一次上傳，把區塊插到第一個標題之後
            match = re.search(r'^#\s+.*$', content, flags=re.MULTILINE)
            if match:
                rest = content[match.end():].lstrip('\n')
                content = content[:match.end()] + '\n\n' + block + '\n\n' + rest
            else:
                content = block + '\n\n' + content.lstrip('\n')

        if not is_image:
            content = self._retarget_readme_data_files(content)
            content = self._update_readme_schema(content)

        if is_image:
            # 圖片資料集的 README 有寫死的 dump 取得日期，一併更新
            content = re.sub(
                r'於\s*\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日取自維基百科\s*dump',
                f'於 {format_dump_date(dump_date, "zh")}取自維基百科dump',
                content,
            )
            content = re.sub(
                r'(Retrieved from the Wikipedia dump on\s*)\d{4}/\d{1,2}/\d{1,2}',
                lambda m: m.group(1) + format_dump_date(dump_date),
                content,
            )

        return content if content != original else None

    def upload(self, repo_id, file_map, version, dump_date, pattern, is_image=False):
        """
        上傳資料檔到 repo root，同時更新 README 與版本記錄檔。

        Args:
            file_map: {本地路徑: repo 內路徑}
        """
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        # root 上多餘的舊檔案要一併刪掉（例如這次只有 5 個 part，上一版有 6 個）
        existing = set(self._root_data_files(repo_id, pattern))
        stale = sorted(existing - set(file_map.values()))

        total = sum(os.path.getsize(p) for p in file_map)
        print(f"  上傳 {len(file_map)} 個檔案（{_human_size(total)}）到 {repo_id}")
        for local, remote in sorted(file_map.items(), key=lambda kv: kv[1]):
            print(f"    {os.path.basename(local)} → {remote} ({_human_size(os.path.getsize(local))})")
        for name in stale:
            print(f"    刪除 root 上多餘的舊檔案: {name}")

        ops = [
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local)
            for local, remote in sorted(file_map.items(), key=lambda kv: kv[1])
        ]
        ops += [CommitOperationDelete(path_in_repo=name) for name in stale]

        # README：更新 dump 日期與版本區塊
        readme = self._read_repo_file(repo_id, 'README.md')
        card = dataset_card(repo_id, version, dump_date)
        if self.refresh_cards and card:
            # 整張換掉。卡片由程式產生，GitHub 與 HF 的說法才不會分岔——
            # 欄位、不變量、授權說明都會演進，靠人記得同步不可靠。
            print("    重寫 README.md（dataset card 由程式產生）")
            ops.append(CommitOperationAdd(
                path_in_repo='README.md', path_or_fileobj=card.encode('utf-8')))
        elif readme is not None:
            updated = self._render_readme(readme, repo_id, version, dump_date, is_image)
            if updated:
                print("    更新 README.md（dump 日期 / 版本區塊）")
                ops.append(CommitOperationAdd(
                    path_in_repo='README.md',
                    path_or_fileobj=updated.encode('utf-8'),
                ))
        elif card:
            print("    建立 README.md（repo 還沒有 dataset card）")
            ops.append(CommitOperationAdd(
                path_in_repo='README.md', path_or_fileobj=card.encode('utf-8')))
        elif repo_id in OMNI_REPOS.values():
            # omni 是新開的 repo，root 還沒有 README。沒有 dataset card 的話
            # Hugging Face 認不出欄位結構，`load_dataset` 也讀不到設定。
            print("    建立 README.md（omni dataset card）")
            lang = next(k for k, v in OMNI_REPOS.items() if v == repo_id)
            ops.append(CommitOperationAdd(
                path_in_repo='README.md',
                path_or_fileobj=_omni_readme(lang, version, dump_date).encode('utf-8'),
            ))
        else:
            print("    ⚠ 找不到 README.md，略過更新")

        # 版本記錄檔：讓下次上傳能準確判斷 root 是哪個月份
        version_info = {
            'version': version,
            'dump_date': str(dump_date),
            'updated_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'source': 'https://github.com/YuhuanStudio/WikiZH_Dataset',
        }
        ops.append(CommitOperationAdd(
            path_in_repo=VERSION_FILE,
            path_or_fileobj=json.dumps(version_info, ensure_ascii=False, indent=2).encode('utf-8'),
        ))

        if self.dry_run:
            print("  （dry-run，未實際上傳）")
            return True

        self.api.create_commit(
            repo_id, repo_type='dataset', operations=ops,
            commit_message=f'feat: update dataset to {version} (wiki dump {dump_date})',
        )
        self._tree_cache.pop(repo_id, None)
        print(f"  ✓ 上傳完成: https://huggingface.co/datasets/{repo_id}")
        return True

    # ---------- 對外主要流程 ----------

    def upload_pretrain(self, lang, files, version, dump_date, archive=True, archive_as=None):
        """上傳文字 Pretrain 資料集"""
        repo_id = PRETRAIN_REPOS[lang]
        print(f"\n--- 上傳 Pretrain ({lang}) → {repo_id} ---")

        is_new = self.ensure_repo(repo_id)

        if archive and not is_new:
            archive_version = archive_as or self.detect_archive_version(repo_id, version, PRETRAIN_FILE_RE)
            if archive_version >= version:
                print(f"  ⚠ 推算出的歸檔月份 {archive_version} 不早於新版本 {version}，略過歸檔")
            else:
                self.archive_root(repo_id, archive_version, PRETRAIN_FILE_RE)

        file_map = {p: os.path.basename(p) for p in sorted(files)}
        return self.upload(repo_id, file_map, version, dump_date, PRETRAIN_FILE_RE)

    def upload_omni(self, lang, files, version, dump_date, archive=True, archive_as=None):
        """上傳圖文交錯（omni）資料集

        檔名規則與分片格式跟純文字版相同，所以沿用 PRETRAIN_FILE_RE 做歸檔
        與清理的判斷；差別只在 repo 與 README。
        """
        repo_id = OMNI_REPOS[lang]
        print(f"\n--- 上傳 omni ({lang}) → {repo_id} ---")

        is_new = self.ensure_repo(repo_id)

        if archive and not is_new:
            archive_version = archive_as or self.detect_archive_version(
                repo_id, version, PRETRAIN_FILE_RE)
            if archive_version >= version:
                print(f"  ⚠ 推算出的歸檔月份 {archive_version} 不早於新版本 {version}，略過歸檔")
            else:
                self.archive_root(repo_id, archive_version, PRETRAIN_FILE_RE)

        file_map = {p: os.path.basename(p) for p in sorted(files)}
        return self.upload(repo_id, file_map, version, dump_date, PRETRAIN_FILE_RE)

    def upload_images(self, files_by_lang, version, dump_date, archive=True, archive_as=None):
        """上傳圖片資料集（tw / cn 共用同一個 repo）"""
        repo_id = IMAGE_REPO
        print(f"\n--- 上傳圖片資料集 → {repo_id} ---")

        is_new = self.ensure_repo(repo_id)

        if archive and not is_new:
            archive_version = archive_as or self.detect_archive_version(repo_id, version, IMAGE_FILE_RE)
            if archive_version >= version:
                print(f"  ⚠ 推算出的歸檔月份 {archive_version} 不早於新版本 {version}，略過歸檔")
            else:
                self.archive_root(repo_id, archive_version, IMAGE_FILE_RE)

        file_map = {}
        for lang, paths in sorted(files_by_lang.items()):
            base = IMAGE_REPO_FILENAME[lang]
            for i, path in enumerate(sorted(paths)):
                if i == 0:
                    file_map[path] = base
                else:
                    # 資料被切分成多個檔案時，README 的 data_files 需要手動補上
                    file_map[path] = base.replace('.jsonl', f'_{i + 1}.jsonl')
                    print(f"  ⚠ {lang} 有多個分割檔，README 的 data_files 需手動加入 {file_map[path]}")
        return self.upload(repo_id, file_map, version, dump_date, IMAGE_FILE_RE, is_image=True)

    # ---------- 取得上一版大小（供檢查比對） ----------

    def previous_pretrain_size(self, lang):
        """上一版 pretrain 資料的總大小"""
        try:
            return sum(self._root_data_files(PRETRAIN_REPOS[lang], PRETRAIN_FILE_RE).values())
        except Exception as e:
            print(f"  ⚠ 無法取得 {lang} 上一版大小: {e}")
            return None

    def previous_image_size(self, lang):
        """上一版圖片資料的檔案大小"""
        try:
            files = self._root_data_files(IMAGE_REPO, IMAGE_FILE_RE)
            return files.get(IMAGE_REPO_FILENAME[lang])
        except Exception as e:
            print(f"  ⚠ 無法取得 {lang} 上一版圖片大小: {e}")
            return None


# ============================================================
# 本地輸出檔案探索
# ============================================================

def find_pretrain_files(output_dir):
    """找出目錄下的 wiki_pretrain_part*.json"""
    if not os.path.isdir(output_dir):
        return []
    return sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if PRETRAIN_FILE_RE.match(f)
    )


def find_image_files(output_dir):
    """找出目錄下的圖片 JSONL（含被分割的 _1、_2 檔案）"""
    if not os.path.isdir(output_dir):
        return []
    return sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if IMAGE_FILE_RE.match(f)
    )


# ============================================================
# 主流程
# ============================================================

def preflight_check(langs=('tw', 'cn'), token=None, upload_pretrain=True,
                    upload_images=True, upload_omni=True):
    """
    生成前先確認 Hugging Face 認證與 repo 都沒問題。

    生成一次要跑好幾個小時，不該等到最後才發現 token 過期或沒有權限。

    Returns:
        bool: 是否可以順利上傳
    """
    print("=" * 60)
    print("檢查 Hugging Face 連線與權限")
    print("=" * 60)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("✗ 未安裝 huggingface_hub，請執行: pip install huggingface_hub")
        return False

    api = HfApi(token=_resolve_token(token))

    try:
        who = api.whoami()
        print(f"  ✓ 已登入: {who.get('name')}")
    except Exception as e:
        print(f"  ✗ HF 認證失敗: {e}")
        print("    請執行 `hf auth login`，或設定 HF_TOKEN 環境變數 / 加上 --hf-token")
        return False

    repos = []
    if upload_pretrain:
        repos += [PRETRAIN_REPOS[lang] for lang in langs]
    if upload_omni:
        repos += [OMNI_REPOS[lang] for lang in langs]
    if upload_images:
        repos.append(IMAGE_REPO)

    ok = True
    for repo_id in repos:
        try:
            api.repo_info(repo_id, repo_type='dataset')
            print(f"  ✓ {repo_id}")
        except Exception as e:
            print(f"  ✗ {repo_id}: {e}")
            ok = False

    if ok:
        print("\n✓ Hugging Face 檢查通過\n")
    else:
        print("\n✗ 有 repo 無法存取，請確認 token 權限\n")
    return ok


def run_upload(base_dir, dump_date, langs=('tw', 'cn'), token=None, dry_run=False,
               archive=True, archive_as=None, skip_checks=False, force=False,
               upload_pretrain=True, upload_images=True, upload_omni=True,
               refresh_cards=False):
    """
    檢查並上傳所有資料集。

    Args:
        base_dir: 專案根目錄（輸出檔在 base_dir/output/{lang}）
        dump_date: 維基百科 dump 日期（YYYYMMDD）
        force: 檢查未通過時仍強制上傳

    Returns:
        bool: 是否全部成功
    """
    version = dump_date_to_version(dump_date)

    print("\n" + "=" * 60)
    print("上傳到 Hugging Face")
    print("=" * 60)
    print(f"dump 日期: {dump_date}")
    print(f"版本號: {version}")
    print(f"語言: {', '.join(langs)}")

    uploader = HFUploader(token=token, dry_run=dry_run, refresh_cards=refresh_cards)

    # 收集本地輸出檔案
    pretrain_files = {}
    omni_files = {}
    image_files = {}
    for lang in langs:
        output_dir = os.path.join(base_dir, 'output', lang)
        if upload_pretrain:
            pretrain_files[lang] = find_pretrain_files(output_dir)
        if upload_omni:
            # omni 版的分片檔名與純文字版相同，靠 omni/ 子目錄區分
            omni_files[lang] = find_pretrain_files(os.path.join(output_dir, 'omni'))
        if upload_images:
            image_files[lang] = find_image_files(output_dir)

    # ---- 上傳前檢查 ----
    if skip_checks:
        print("\n⚠ 已略過上傳前檢查")
    else:
        print("\n" + "-" * 60)
        print("上傳前檢查")
        print("-" * 60)
        results = []
        for lang in langs:
            if upload_pretrain:
                results.append(check_pretrain_files(
                    pretrain_files[lang], lang, uploader.previous_pretrain_size(lang)))
                results[-1].report()
            if upload_omni and omni_files.get(lang):
                results.append(check_pretrain_files(
                    omni_files[lang], lang, None,
                    columns=PRETRAIN_COLUMNS + ('images',),
                    label=f'{lang} omni'))
                results[-1].report()
            if upload_images:
                paths = image_files[lang]
                results.append(check_image_file(
                    paths[0] if paths else None, lang, uploader.previous_image_size(lang)))
                results[-1].report()

        if not all(r.ok for r in results):
            print("\n✗ 上傳前檢查未通過")
            if not force:
                print("  已中止上傳。確認資料無誤後可加上 --force-upload 略過檢查。")
                return False
            print("  ⚠ --force-upload 已指定，仍繼續上傳")

    # ---- 實際上傳 ----
    ok = True
    for lang in langs:
        if upload_pretrain and pretrain_files.get(lang):
            try:
                uploader.upload_pretrain(lang, pretrain_files[lang], version, dump_date,
                                         archive=archive, archive_as=archive_as)
            except Exception as e:
                print(f"✗ Pretrain ({lang}) 上傳失敗: {e}")
                import traceback
                traceback.print_exc()
                ok = False

    for lang in langs:
        if upload_omni and omni_files.get(lang):
            try:
                uploader.upload_omni(lang, omni_files[lang], version, dump_date,
                                     archive=archive, archive_as=archive_as)
            except Exception as e:
                print(f"✗ omni ({lang}) 上傳失敗: {e}")
                import traceback
                traceback.print_exc()
                ok = False

    if upload_images:
        available = {lang: paths for lang, paths in image_files.items() if paths}
        if available:
            try:
                uploader.upload_images(available, version, dump_date,
                                       archive=archive, archive_as=archive_as)
            except Exception as e:
                print(f"✗ 圖片資料集上傳失敗: {e}")
                import traceback
                traceback.print_exc()
                ok = False

    print("\n" + "=" * 60)
    print("✓ 上傳流程完成" if ok else "✗ 上傳流程有錯誤")
    print("=" * 60)
    return ok


def main():
    import argparse

    parser = argparse.ArgumentParser(description='將維基百科資料集上傳到 Hugging Face')
    parser.add_argument('--dump-date', type=str, required=True, help='維基百科 dump 日期（YYYYMMDD）')
    parser.add_argument('--base-dir', type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help='專案根目錄（預設為本檔案所在目錄）')
    parser.add_argument('--lang', type=str, nargs='+', choices=['tw', 'cn'], default=['tw', 'cn'],
                        help='要上傳的語言版本')
    parser.add_argument('--hf-token', type=str, help='HF token（預設讀 HF_TOKEN 環境變數或本地登入快取）')
    parser.add_argument('--dry-run', action='store_true', help='只做檢查與規劃，不實際上傳')
    parser.add_argument('--no-archive', action='store_true', help='不要把 root 舊檔案歸檔到月份資料夾')
    parser.add_argument('--archive-as', type=str, help='手動指定歸檔月份（YYMM），預設自動判斷')
    parser.add_argument('--skip-checks', action='store_true', help='略過上傳前檢查')
    parser.add_argument('--force-upload', action='store_true', help='檢查未通過時仍強制上傳')
    parser.add_argument('--only-pretrain', action='store_true', help='只上傳文字 Pretrain 資料集')
    parser.add_argument('--only-images', action='store_true', help='只上傳圖片資料集')

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            pass

    args = parser.parse_args()

    ok = run_upload(
        base_dir=args.base_dir,
        dump_date=args.dump_date,
        langs=tuple(args.lang),
        token=args.hf_token,
        dry_run=args.dry_run,
        archive=not args.no_archive,
        archive_as=args.archive_as,
        skip_checks=args.skip_checks,
        force=args.force_upload,
        upload_pretrain=not args.only_images,
        upload_images=not args.only_pretrain,
    )
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
