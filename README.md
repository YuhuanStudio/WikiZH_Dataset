# WikiZH_Dataset

[![License](https://img.shields.io/github/license/YuhuanStudio/WikiZH_Dataset?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![GitHub Stars](https://img.shields.io/github/stars/YuhuanStudio/WikiZH_Dataset?style=flat-square&logo=github)](https://github.com/YuhuanStudio/WikiZH_Dataset/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/YuhuanStudio/WikiZH_Dataset?style=flat-square&logo=github)](https://github.com/YuhuanStudio/WikiZH_Dataset/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/YuhuanStudio/WikiZH_Dataset?style=flat-square&logo=github)](https://github.com/YuhuanStudio/WikiZH_Dataset/issues)
[![Hugging Face](https://img.shields.io/badge/Datasets-Hugging%20Face-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/yuhuanstudio)

維基百科中文語料處理工具集合，用於生成預訓練與多模態資料集。

本專案用於生成以下 Hugging Face 資料集 🤗：
- [![Hugging Face](https://img.shields.io/badge/wikipedia--pretrain--zh--tw-Download-blue?style=flat-square&logo=huggingface)](https://huggingface.co/datasets/yuhuanstudio/wikipedia-pretrain-zh-tw) 台灣正體中文維基百科（3.6M 條目）🇹🇼
- [![Hugging Face](https://img.shields.io/badge/wikipedia--pretrain--zh-Download-blue?style=flat-square&logo=huggingface)](https://huggingface.co/datasets/yuhuanstudio/wikipedia-pretrain-zh) 簡體中文維基百科（3.6M 條目）🇨🇳
- [![Hugging Face](https://img.shields.io/badge/wikipedia--image--zh--tw-Download-blue?style=flat-square&logo=huggingface)](https://huggingface.co/datasets/yuhuanstudio/wikipedia-image-zh-tw) 台灣正體中文維基百科圖像資料（528K 條目）🖼️

---

## ✨ 專案特色

- 🎯 **統一 CLI 入口**：單一命令介面支援下載、轉換、提取與下載
- 🌐 **語言切換支援**：透過 `--lang` 選項輸出繁體（tw）或簡體（cn）
- 🤖 **多模態支援**：文字 Pretrain 與圖片 Dataset
- 📝 **共享 Markdown**：Markdown 檔案語言無關，減少重複存儲
- 🔍 **高質量過濾**：僅保留有意義的知識性段落，去除圖片、表格、infobox、外部連結、註釋等非知識內容

---

## 🚀 快速開始

### 📦 安裝相依套件

```bash
python -m pip install -r requirements.txt
```

### 💻 CLI 使用範例

#### ⭐ 主要任務（推薦使用）

- **生成文字 Pretrain Dataset（預設）**：
  ```bash
  python wiki_cli.py --pretrain-dataset
  ```

- **生成圖片 Dataset**：
  ```bash
  python wiki_cli.py --image-dataset
  ```

- **下載圖片**：
  ```bash
  python wiki_cli.py --download-images
  ```

-- **一次生成繁體、簡體與圖片資訊（不含下載）**：
  ```bash
  python wiki_cli.py --generate-all
  ```

#### 🔧 單步操作（進階）

- **下載維基百科數據**：
  ```bash
  python wiki_cli.py --download
  ```

- **將 XML 轉換為 Markdown**：
  ```bash
  python wiki_cli.py --to-md
  ```

- **將 Markdown 轉換為 Pretrain JSON**：
  ```bash
  python wiki_cli.py --to-pretrain
  ```

- **提取圖片資訊**：
  ```bash
  python wiki_cli.py --extract-images
  ```

### 📖 顯示完整說明

```bash
python wiki_cli.py --help
```

---

## 🤗 使用 Hugging Face 資料集

你也可以直接從 Hugging Face 載入已生成的資料集：

```python
from datasets import load_dataset

# 台灣正體中文維基百科（3.6M 條目）
dataset_tw = load_dataset("yuhuanstudio/wikipedia-pretrain-zh-tw", split="train")

# 簡體中文維基百科（3.6M 條目）
dataset_cn = load_dataset("yuhuanstudio/wikipedia-pretrain-zh", split="train")

# 台灣正體中文維基百科圖像資料（528K 條目）
dataset_images = load_dataset("yuhuanstudio/wikipedia-image-zh-tw", split="train")
```

- **圖片資訊**：`output/{lang}/wiki_images_dataset.jsonl`
- **圖片檔案**：`images/{lang}/`

> 預設語言：`tw`（繁體中文）。指定 `--lang cn` 可輸出簡體中文。

---

## 📊 資料集結構

### 📚 文字 Pretrain Dataset

```json
{
  "title": "農業 - 定義",
  "text": "根據東漢時期《說文解字》和清康熙時期《康熙字典》的解釋，農字都是耕種的意思，這表示在中國古代就只有種植業才會被稱作農業。但現代對農業的定義更加廣泛，包括利用自然資源生產維持生命所需的物品，如食物、纖維、林業產品、園藝作物，以及與之相關的服務。"
}
```

**欄位說明**：

- `title` (string)：段落標題（如有，通常為「主題 - 章節」格式）🏷️
- `text` (string)：維基百科文本內容 📝

**特點**：

- 每筆資料為一個段落，非整篇文章 ✅
- 內容經過長度與質量篩選，部分過短或雜訊段落已被排除 🎯
- 僅保留有意義的知識性段落，去除圖片、表格、infobox、外部連結、註釋等非知識內容 🧹

### 🖼️ 圖片 Dataset

```json
{
  "url": "https://zh.wikipedia.org/wiki/Special:FilePath/Euclid.jpg",
  "title": "歐幾里得，西元前三世紀的古希臘數學家，而現在被認為是幾何之父，此畫為拉斐爾的作品《雅典學院》",
  "file_name": "Euclid.jpg",
  "page": "数学",
  "tag": "[[File:Euclid.jpg|right|thumb|200px|[[歐幾里得]]，西元前三世紀的[[古希臘]][[數學家]]，而現在被認為是[[幾何]]之父，此畫為[[拉斐爾·聖齊奧|拉斐爾]]的作品《[[雅典學院 (畫作)|雅典學院]]》]]"
}
```

**欄位說明**：

- `url` (string)：圖片的原始網址，來自 Wikipedia 🔗
- `title` (string)：圖片說明 📝
- `file_name` (string)：圖片檔案名稱，便於本地儲存或引用 📁
- `page` (string)：圖片所屬的維基百科條目主題或頁面名稱 📄
- `tag` (string)：原始維基語法標註，包含圖片顯示方式、尺寸、說明等設定 ⚙️

**特點**：

- 經 OpenCC 轉換，確保繁簡體不混雜 ✨
- `title` 部分有中文解釋者，經 OpenCC 轉換成繁體中文後沒有繁簡體混雜的問題 🇹🇼
- 對於沒有中文說明的條目，`title` 直接使用檔案名稱 📄

---

## 📈 資料統計

| 資料集 | 語言 | 條目數量 | 大小 | 更新日期 |
|--------|------|----------|------|----------|
| **wikipedia-pretrain-zh-tw** | 台灣正體中文 | 3,607,037 | 1.64 GB | 2026-01-02 |
| **wikipedia-pretrain-zh** | 簡體中文 | 3,606,872 | 1.64 GB | 2026-01-02 |
| **wikipedia-image-zh-tw** | 台灣正體中文 | 527,856 | 192 MB (原始) / 91.2 MB (Parquet) | 2025-12-01 |

---

## ⚠️ 資料說明與開源注意事項

- 本專案**不包含**大型已生成資料（`downloads/`, `output/`, `images/`）。
- 若要重現 Dataset，請使用 `wiki_cli.py` 中的步驟下載與處理來源。
- 請確認你的環境有足夠磁碟空間與網路帶寬以處理維基百科資料。
- 資料經過 OpenCC 轉換，確保繁簡體不混雜。
- **重要**：此資料經過 OpenCC 轉換，不適用於嚴謹的科研項目。如需原始簡體資料，請使用 `wikipedia-pretrain-zh` 資料集。
- 文字資料僅包含知識性段落，圖片、詳細資訊可能不被納入其中，並且可能不包含所有文章。

---

## 🧩 模組說明

| 檔案 | 說明 |
|------|------|
| **wiki_cli.py** | 統一 CLI 入口，協調各模組 🎮 |
| **wiki_downloader.py** | 下載維基百科 XML.bz2 ⬇️ |
| **wiki_parser.py** | 解析維基百科 XML 🔍 |
| **md_converter.py** | 將 XML 轉換為 Markdown 🔄 |
| **md_to_json.py** | 將 Markdown 轉換為 Pretrain JSON（段落） 📄 |
| **image_extractor.py** | 從 XML 提取圖片資訊 🖼️ |
| **image_downloader.py** | 根據 JSONL 下載圖片 📥 |

---

## 📜 授權

請參閱 [LICENSE](LICENSE) 檔案。

---

## 🤝 貢獻

歡迎提 Issue 與 Pull Request。請參閱 [CONTRIBUTING.md](CONTRIBUTING.md) 了解詳情。

---

## 🔒 安全

如果你發現安全漏洞，請參閱 [SECURITY.md](SECURITY.md) 了解如何回報。

---

## 🔗 相關連結與資源

- [OpenCC 繁簡轉換工具](https://github.com/BYVoid/OpenCC)
- [維基百科 dumps 文件](https://dumps.wikimedia.org/zhwiki/)


---

## 📬 聯絡資訊

如有任何問題或建議，請聯絡我們：

- **Email**: huhu11256@gmail.com
- **GitHub**: [YuhuanStudio](https://github.com/YuhuanStudio)


