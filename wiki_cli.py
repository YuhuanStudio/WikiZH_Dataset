"""
維基百科中文數據處理 CLI
統一的命令行入口，支援以下功能：

獨立任務（主要功能）：
1. 文字 Dataset：下載 → 解析成中間層 → 轉 Parquet（預設）
2. 圖片 Dataset：提取圖片資訊 JSONL
3. 下載圖片：根據圖片資訊下載
4. 上傳 Dataset：檢查輸出並推送到 Hugging Face

單獨步驟（進階功能）：
- 下載維基百科數據
- 將 XML 解析成中間層（分片 JSONL）
- 由中間層生成文檔級 Parquet 資料集
- 從 XML 提取圖片資訊
- 下載圖片
- 上傳到 Hugging Face
"""

import os
import sys
import argparse
import shutil
import importlib.util


# 必要的第三方套件：{import 名稱: pip 安裝名稱}
# 用於在執行任何耗時任務前，先確認所有依賴都已安裝且可正常匯入
REQUIRED_PACKAGES = {
    'gensim': 'gensim',
    'tqdm': 'tqdm',
    'pyarrow': 'pyarrow',
    'huggingface_hub': 'huggingface_hub',
}

# 只有特定任務才需要的套件。列進上面那張表的話，不做該任務的人也會被擋下來——
# `pangu` 早就移除了（排版空格是不可逆的有損轉換），`requests` 只有「下載圖片」
# 用得到，卻讓「上傳資料集」也無法執行。
OPTIONAL_PACKAGES = {
    'download_images': {'requests': 'requests'},
}


# 本地資料保留幾個月份（當月 + 上個月）
# 舊版本在 Hugging Face 上有月份資料夾存檔，本地不需要一直留著佔空間
KEEP_MONTHS = 2


def check_optional(task):
    """檢查某個任務專屬的套件；缺了才報，不影響其他任務"""
    need = OPTIONAL_PACKAGES.get(task, {})
    missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"✗ 「{task}」需要以下套件：{', '.join(missing)}")
        print(f"    pip install {' '.join(missing)}")
        sys.exit(1)


def check_dependencies():
    """執行前檢查所有必要的 pip 套件是否已安裝且可正常匯入。

    避免在下載／轉換等耗時步驟跑到一半才因缺少套件而失敗。
    若有缺漏或無法匯入的套件，會印出明確的安裝指令並中止程式。
    """
    print("=" * 60)
    print("檢查必要套件")
    print("=" * 60)

    missing = []  # 完全沒安裝
    broken = []   # 有安裝但匯入失敗（例如版本不相容或缺少系統依賴）

    for import_name, pip_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(import_name) is None:
            print(f"  ✗ {import_name:<10} (未安裝)")
            missing.append(pip_name)
            continue

        # 實際匯入一次，確保套件不僅存在、且能正常載入
        try:
            __import__(import_name)
            print(f"  ✓ {import_name}")
        except Exception as e:
            print(f"  ✗ {import_name:<10} (匯入失敗: {e})")
            broken.append(pip_name)

    if missing or broken:
        print("\n" + "=" * 60)
        print("✗ 缺少必要套件，無法繼續執行")
        print("=" * 60)
        to_install = missing + broken
        if missing:
            print(f"未安裝: {', '.join(missing)}")
        if broken:
            print(f"匯入失敗: {', '.join(broken)}")
        print("\n請先安裝所有依賴後再執行：")
        print(f"    pip install {' '.join(to_install)}")
        print("或直接安裝 requirements.txt：")
        print("    pip install -r requirements.txt")
        sys.exit(1)

    print("\n✓ 所有必要套件皆已安裝且可正常使用\n")


class WikiCLI:
    """維基百科數據處理統一入口"""

    def __init__(self, lang='tw'):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.lang = lang  # 'tw' for Traditional Chinese, 'cn' for Simplified Chinese
        self.download_dir = os.path.join(self.base_dir, 'downloads')
        # 中間層（語言中立的解析結果，tw/cn 共用）。存成分片 JSONL 而不是
        # 一條目一檔——155 萬個小檔案在 NTFS 上的建檔／列目錄／刪除成本極高。
        self.md_dir = os.path.join(self.base_dir, 'parsed')
        self.json_dir = os.path.join(self.base_dir, f'output/{lang}')
        self.image_dir = os.path.join(self.base_dir, f'images/{lang}')

    def _ensure_dir(self, dir_path):
        """確保目錄存在"""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _image_output_file(self):
        """圖片 JSONL 的預設輸出路徑（檔名與 Hugging Face repo 上的一致）"""
        from hf_uploader import IMAGE_REPO_FILENAME

        return os.path.join(self.json_dir, IMAGE_REPO_FILENAME[self.lang])

    def _clean_stale_outputs(self, pattern):
        """
        清掉輸出目錄中上一次生成留下的舊檔案。

        避免新版本檔案數量變少時（例如從 6 個 part 變成 5 個），
        殘留的舊檔案被誤認為本次結果一起上傳。
        """
        if not os.path.isdir(self.json_dir):
            return
        removed = [f for f in sorted(os.listdir(self.json_dir)) if pattern.match(f)]
        for name in removed:
            os.remove(os.path.join(self.json_dir, name))
        if removed:
            print(f"已清除 {len(removed)} 個上一次生成的舊檔案: {', '.join(removed)}")

    def _prune_old_versions(self, root, keep=KEEP_MONTHS):
        """
        只保留最新 keep 個月份的資料，更舊的整個刪掉。

        例如 8 月執行時保留 7、8 月，9 月執行時 7 月就會被清掉。
        downloads/ 用 YYYYMMDD、parsed/ 用 YYYYMM 命名，兩者都取前 6 碼判斷月份，
        所以同一個月有多份 dump 時會被視為同一個月一起保留。
        """
        if not os.path.isdir(root):
            return []

        # 依月份分組（同月份的多個目錄視為一組）
        by_month = {}
        for name in sorted(os.listdir(root)):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            if len(name) in (6, 8) and name.isdigit():
                by_month.setdefault(name[:6], []).append(name)

        keep_months = set(sorted(by_month)[-keep:])
        removed = []
        for month in sorted(by_month):
            if month in keep_months:
                continue
            for name in by_month[month]:
                print(f"正在刪除舊資料夾: {os.path.join(os.path.basename(root), name)}")
                shutil.rmtree(os.path.join(root, name))
                removed.append(name)

        if removed:
            print(f"✓ 已刪除 {len(removed)} 個舊資料夾（只保留最新 {keep} 個月份）")
        else:
            print(f"沒有需要刪除的舊資料夾（目前保留: {', '.join(sorted(keep_months)) or '無'}）")
        return removed

    def _get_latest_dump_date(self):
        """從 downloads 目錄推算最新的 dump 日期（YYYYMMDD）"""
        xml_path = self._get_latest_xml_file()
        if xml_path is None:
            return None
        return os.path.basename(os.path.dirname(xml_path))

    def _get_latest_md_dir(self):
        """獲取最新的中間層目錄"""
        if not os.path.exists(self.md_dir):
            return None

        # 查找所有符合 YYYYMM 格式的目錄
        month_dirs = []
        for item in os.listdir(self.md_dir):
            if os.path.isdir(os.path.join(self.md_dir, item)):
                if len(item) == 6 and item.isdigit():
                    month_dirs.append(item)

        if not month_dirs:
            return None

        # 返回最新的月份目錄
        latest_month = max(month_dirs)
        return os.path.join(self.md_dir, latest_month)

    def _get_latest_xml_file(self):
        """獲取最新的 XML 文件"""
        if not os.path.exists(self.download_dir):
            return None

        # 查找所有日期目錄
        date_dirs = []
        for item in os.listdir(self.download_dir):
            if os.path.isdir(os.path.join(self.download_dir, item)):
                if len(item) == 8 and item.isdigit():
                    date_dirs.append(item)

        if not date_dirs:
            return None

        # 返回最新日期的 XML 文件
        latest_date = max(date_dirs)
        date_dir = os.path.join(self.download_dir, latest_date)

        # 查找 .xml.bz2 文件
        for file in os.listdir(date_dir):
            if file.endswith('.xml.bz2'):
                return os.path.join(date_dir, file)

        return None

    def download_wiki(self):
        """下載維基百科數據"""
        print("=" * 60)
        print("步驟 1/2: 下載維基百科數據")
        print("=" * 60)

        try:
            from wiki_downloader import WIKIDownload

            downloader = WIKIDownload(self.download_dir)
            xml_path, latest_date = downloader.run(verbose=True)

            print(f"\n✓ 下載完成: {xml_path}")
            file_size_gb = os.path.getsize(xml_path) / (1024*1024*1024)
            print(f"  文件大小: {file_size_gb:.2f} GB")

            # 只保留當月與上個月的 dump，更舊的清掉
            self._prune_old_versions(self.download_dir)

            return xml_path, latest_date
        except Exception as e:
            print(f"✗ 下載失敗: {e}")
            return None, None

    def _md_dir_is_ready(self, path):
        """
        檢查中間層是否已經完整轉換過。

        只看「有沒有檔案」是不夠的——中斷的轉換也會留下一堆分片，
        沿用那份半成品會產出殘缺的資料集，所以用完成標記判斷。
        """
        from page_store import is_complete

        return is_complete(path)

    def convert_to_md(self, xml_path=None, latest_date=None, force=False):
        """將 XML 解析成中間層（分片 JSONL）"""
        if xml_path is None:
            xml_path = self._get_latest_xml_file()
            if xml_path is None:
                print("✗ 找不到 XML 文件，請先執行下載")
                return None

        if latest_date is None:
            # 從路徑提取日期
            xml_dir = os.path.dirname(xml_path)
            latest_date = os.path.basename(xml_dir)

        print("\n" + "=" * 60)
        print("步驟 2/2: 解析 XML（產生語言中立的中間層）")
        print("=" * 60)

        # 創建帶月份標記的輸出目錄（格式：YYYYMM）
        simplified_date = latest_date[:4] + latest_date[4:6]
        output_dir_with_month = os.path.join(self.md_dir, simplified_date)
        print(f"輸出目錄: {output_dir_with_month}")

        # 這一步要跑一個多小時，已經轉換過就直接沿用，讓後續步驟可以單獨重跑
        if not force and self._md_dir_is_ready(output_dir_with_month):
            print("✓ 該月份的中間層已存在，跳過解析（要重新解析請加 --force-md）")
            return output_dir_with_month

        try:
            # 創建新的輸出目錄
            self._ensure_dir(output_dir_with_month)

            # 只保留當月與上個月的中間層，更舊的清掉
            self._prune_old_versions(self.md_dir)

            # 使用 WIKIParse2Doc 類解析並生成文件
            from md_converter import WIKIParse2Doc

            parser = WIKIParse2Doc(xml_path, output_dir_with_month, markdown=True)
            md_count = parser.run(num=None)

            from page_store import shard_paths

            print("\n" + "=" * 60)
            print("✓ 完成！")
            print(f"  輸出目錄: {output_dir_with_month}")
            print(f"  解析條目數: {md_count:,}")
            print(f"  分片數: {len(shard_paths(output_dir_with_month))}")
            print("=" * 60)

            return output_dir_with_month
        except Exception as e:
            print(f"✗ 轉換失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

    def convert_md_to_datasets(self, input_dir=None, dump_date=None):
        """一次產出 tw/cn × 純文字/omni 四組 Parquet（共用同一次中間層走訪）"""
        if input_dir is None:
            input_dir = self._get_latest_md_dir()
            if input_dir is None:
                print("✗ 找不到中間層目錄，請先執行解析")
                return None

        print("=" * 60)
        print("由中間層生成四組資料集（繁體／簡體 × 純文字／omni）")
        print("=" * 60)

        try:
            from hf_uploader import PRETRAIN_FILE_RE
            from md_to_dataset import process_directory_variants

            output_dirs = {}
            for lang in ('tw', 'cn'):
                sub = WikiCLI(lang=lang)
                sub._ensure_dir(sub.json_dir)
                sub._clean_stale_outputs(PRETRAIN_FILE_RE)
                omni_dir = os.path.join(sub.json_dir, 'omni')
                sub._ensure_dir(omni_dir)
                for name in os.listdir(omni_dir):
                    if name.endswith('.parquet'):
                        os.remove(os.path.join(omni_dir, name))
                output_dirs[lang] = sub.json_dir

            result = process_directory_variants(input_dir=input_dir,
                                                output_dirs=output_dirs)
            if not result:
                return None

            print("\n" + "=" * 60)
            print("✓ 完成！")
            for (lang, mode), (files, count) in sorted(result.items()):
                print(f"  {lang} {mode:<5}: {count:,} 筆／{len(files)} 個分片")
            print("=" * 60)
            return result
        except Exception as e:
            print(f"✗ 轉換失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

    def convert_md_to_dataset(self, input_dir=None, dump_date=None):
        """由中間層生成文檔級 Parquet 資料集（純文字版 + omni 交錯版）

        dump_date 只用於顯示，不寫進資料本身（版本資訊記在 dataset_version.json 與 README）。
        """
        if input_dir is None:
            input_dir = self._get_latest_md_dir()
            if input_dir is None:
                print("✗ 找不到中間層目錄，請先執行解析")
                return None

        self._ensure_dir(self.json_dir)

        print("=" * 60)
        print("由中間層生成文檔級 Parquet 資料集")
        print("=" * 60)

        try:
            from hf_uploader import PRETRAIN_FILE_RE
            from md_to_dataset import process_directory_doc

            # 先清掉上一次生成的檔案，避免新舊版本混在一起
            self._clean_stale_outputs(PRETRAIN_FILE_RE)

            files, count = process_directory_doc(
                input_dir=input_dir,
                output_dir=self.json_dir,
                lang=self.lang,
            )
            if not files:
                return None

            # omni（文圖交錯）版寫在 omni/ 子目錄。少了這一步，正式流程只產得出
            # 純文字版，交錯版永遠只存在於 QA 腳本裡。
            omni_dir = os.path.join(self.json_dir, 'omni')
            self._ensure_dir(omni_dir)
            for name in os.listdir(omni_dir):
                if name.endswith('.parquet'):
                    os.remove(os.path.join(omni_dir, name))
            omni_files, omni_count = process_directory_doc(
                input_dir=input_dir,
                output_dir=omni_dir,
                lang=self.lang,
                omni=True,
            )

            print("\n" + "=" * 60)
            print("✓ 完成！")
            print(f"  純文字: {count:,} 筆／{len(files)} 個分片")
            print(f"  omni  : {omni_count:,} 筆／{len(omni_files)} 個分片")
            print("=" * 60)

            return files
        except Exception as e:
            print(f"✗ 轉換失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

    def extract_images(self, xml_path=None, output_file=None, max_images=None):
        """從 XML 提取圖片資訊"""
        if xml_path is None:
            xml_path = self._get_latest_xml_file()
            if xml_path is None:
                print("✗ 找不到 XML 文件，請先執行下載")
                return None

        if output_file is None:
            self._ensure_dir(self.json_dir)
            output_file = self._image_output_file()

        print("=" * 60)
        print("從 XML 提取圖片資訊")
        print("=" * 60)
        print(f"輸入文件: {xml_path}")
        print(f"輸出文件: {output_file}")
        print(f"語言版本: {'繁體中文' if self.lang == 'tw' else '簡體中文'}")

        try:
            from hf_uploader import IMAGE_FILE_RE
            from image_extractor import extract_wiki_images

            # 先清掉上一次生成的圖片 JSONL，避免新舊版本混在一起
            self._clean_stale_outputs(IMAGE_FILE_RE)

            extract_wiki_images(xml_path, output_file, max_images=max_images, lang=self.lang)

            # image_extractor 一律以 <base>_<序號>.jsonl 命名，
            # 沒有被分割時改回不帶序號的正式檔名（與 Hugging Face 上的檔名一致）
            output_file = self._normalize_image_output(output_file)

            print("\n" + "=" * 60)
            print("✓ 完成！")
            print("=" * 60)

            return output_file
        except Exception as e:
            print(f"✗ 提取失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _normalize_image_output(self, output_file):
        """把 image_extractor 產出的 <base>_1.jsonl 改名成 <base>.jsonl（未被分割時）"""
        base, ext = os.path.splitext(output_file)
        first_part = f"{base}_1{ext}"
        if not os.path.exists(first_part):
            return output_file

        # 有第二個分割檔就代表資料被切開了，維持原本帶序號的命名
        if os.path.exists(f"{base}_2{ext}"):
            print(f"⚠ 圖片資料被分割成多個檔案，維持 {os.path.basename(base)}_N{ext} 命名")
            return first_part

        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename(first_part, output_file)
        print(f"已重新命名為正式檔名: {os.path.basename(output_file)}")
        return output_file

    def upload_to_hf(self, dump_date=None, langs=('tw', 'cn'), **kwargs):
        """檢查輸出檔案並上傳到 Hugging Face"""
        if dump_date is None:
            dump_date = self._get_latest_dump_date()
            if dump_date is None:
                print("✗ 無法判斷 dump 日期，請用 --dump-date 指定")
                return False

        try:
            from hf_uploader import run_upload

            return run_upload(self.base_dir, dump_date, langs=langs, **kwargs)
        except Exception as e:
            print(f"✗ 上傳失敗: {e}")
            import traceback
            traceback.print_exc()
            return False

    def download_images(self, jsonl_path=None, output_dir=None):
        """下載圖片"""
        if jsonl_path is None:
            # 在 json_dir 中查找圖片 JSONL 文件
            self._ensure_dir(self.json_dir)
            image_files = [f for f in os.listdir(self.json_dir) if 'image' in f.lower() and f.endswith('.jsonl')]
            if image_files:
                jsonl_path = os.path.join(self.json_dir, image_files[0])
            else:
                print("✗ 找不到圖片 JSONL 文件，請先執行提取")
                return None

        if output_dir is None:
            output_dir = self.image_dir

        print("=" * 60)
        print("下載圖片")
        print("=" * 60)
        print(f"輸入文件: {jsonl_path}")
        print(f"輸出目錄: {output_dir}")

        try:
            check_optional('download_images')
            from image_downloader import download_images_from_jsonl

            download_images_from_jsonl(jsonl_path, output_dir)

            print("\n" + "=" * 60)
            print("✓ 完成！")
            print("=" * 60)

            return output_dir
        except Exception as e:
            print(f"✗ 下載失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_task(self, task='text'):
        """執行指定的獨立任務

        Args:
            task: 任務類型
                - 'text': 文字 Dataset（下載 → 解析成中間層 → 轉 Parquet）
                - 'image-dataset': 圖片 Dataset（提取圖片資訊）
                - 'download-images': 下載圖片
        """
        print("\n" + "=" * 60)
        print("維基百科中文數據處理流程")
        print("=" * 60)

        if task == 'text':
            # 文字 Dataset 任務
            print("任務：生成文字 Pretrain Dataset")
            print("步驟：下載 → 解析成中間層 → 轉 Parquet 資料集")
            print()

            # 步驟 1: 下載
            xml_path, latest_date = self.download_wiki()
            if xml_path is None:
                return

            # 步驟 2: 解析成中間層
            md_dir = self.convert_to_md(xml_path, latest_date)
            if md_dir is None:
                return

            # 步驟 3: 轉換為文檔級 Parquet 資料集
            self.convert_md_to_dataset(md_dir, dump_date=latest_date)

        elif task == 'image-dataset':
            # 圖片 Dataset 任務
            print("任務：生成圖片 Dataset")
            print("步驟：提取圖片資訊 JSONL")
            print()

            # 步驟 1: 下載
            xml_path, latest_date = self.download_wiki()
            if xml_path is None:
                return

            # 步驟 2: 提取圖片資訊
            self.extract_images(xml_path)

        elif task == 'download-images':
            # 下載圖片任務
            print("任務：下載圖片")
            print("步驟：根據圖片資訊 JSONL 下載圖片")
            print()

            # 下載圖片
            self.download_images()

        else:
            print(f"✗ 未知的任務類型: {task}")
            return

        print("\n" + "=" * 60)
        print("✓ 任務完成！")
        print("=" * 60)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='維基百科中文數據處理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
使用範例:

主要任務：
    生成文字 Pretrain Dataset（預設）:
        python wiki_cli.py --pretrain-dataset

    生成圖片 Dataset:
        python wiki_cli.py --image-dataset

    下載圖片:
        python wiki_cli.py --download-images

單步操作（進階）：
    下載維基百科數據:
        python wiki_cli.py --download

    將 XML 解析成中間層:
        python wiki_cli.py --to-md

    由中間層生成 Parquet 資料集:
        python wiki_cli.py --to-pretrain

    提取圖片資訊:
        python wiki_cli.py --extract-images

Hugging Face 上傳：
    一次生成全部並自動上傳（每月例行作業）:
        python wiki_cli.py --generate-all

    只上傳已生成好的輸出:
        python wiki_cli.py --upload

    先試跑看看會做什麼（不會真的寫入 HF）:
        python wiki_cli.py --upload --dry-run
                """
    )

    # 獨立任務參數
    parser.add_argument('--pretrain-dataset', action='store_true', help='生成文字 Pretrain Dataset（下載 → 轉 MD → 轉 Parquet 資料集，預設）')
    parser.add_argument('--image-dataset', action='store_true', help='生成圖片 Dataset（提取圖片資訊 JSONL）')

    # 單獨步驟參數（進階功能）
    parser.add_argument('--download', action='store_true', help='下載維基百科數據')
    parser.add_argument('--to-md', action='store_true', help='解析 XML 產生中間層')
    parser.add_argument('--to-pretrain', action='store_true', help='由中間層生成文檔級 Parquet 資料集')
    parser.add_argument('--extract-images', action='store_true', help='從 XML 提取圖片資訊')
    parser.add_argument('--download-images', action='store_true', help='下載圖片')
    parser.add_argument('--generate-all', action='store_true', help='一次生成繁體、簡體與圖片資訊（不含下載）')

    #（已移除舊版相容參數及 skip-* 選項，介面已簡化）

    # 語言參數
    parser.add_argument('--lang', type=str, choices=['tw', 'cn'], default='tw',
                    help='輸出語言版本 (tw=繁體中文, cn=簡體中文, 預設: tw)')

    # 路徑參數
    parser.add_argument('--xml-path', type=str, help='指定 XML 文件路徑')
    parser.add_argument('--md-dir', type=str, help='指定中間層目錄路徑')
    parser.add_argument('--json-file', type=str, help='指定 JSONL 輸出文件路徑')
    parser.add_argument('--image-json', type=str, help='指定圖片 JSONL 文件路徑')
    parser.add_argument('--image-dir', type=str, help='指定圖片輸出目錄')
    parser.add_argument('--max-images', type=int, help='最大圖片數量（用於 --extract-images）')

    # Hugging Face 上傳參數
    parser.add_argument('--upload', action='store_true', help='將現有輸出上傳到 Hugging Face')
    parser.add_argument('--no-upload', action='store_true', help='--generate-all 完成後不要自動上傳')
    parser.add_argument('--dump-date', type=str, help='指定 dump 日期（YYYYMMDD），預設由 downloads 目錄推算')
    parser.add_argument('--hf-token', type=str, help='HF token（預設讀 HF_TOKEN 環境變數或本地登入快取）')
    parser.add_argument('--dry-run', action='store_true', help='上傳只做檢查與規劃，不實際寫入 HF')
    parser.add_argument('--no-archive', action='store_true', help='上傳時不要把舊版本歸檔到月份資料夾')
    parser.add_argument('--refresh-cards', action='store_true',
                        help='把 HF 的 dataset card 整張換成程式產生的版本')
    parser.add_argument('--archive-as', type=str, help='手動指定歸檔月份（YYMM），預設自動判斷')
    parser.add_argument('--skip-checks', action='store_true', help='略過上傳前的資料檢查')
    parser.add_argument('--force-upload', action='store_true', help='上傳前檢查未通過時仍強制上傳')

    parser.add_argument('--force-md', action='store_true', help='即使該月份的中間層已存在也重新解析')
    parser.add_argument('--skip-check', action='store_true', help='跳過執行前的套件檢查')

    # 確保輸出使用 UTF-8，避免在 Windows GBK 主控台輸出 ✓／✗ 等字元時崩潰
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            pass

    args = parser.parse_args()

    # 執行前先確認所有必要套件皆已安裝且可用，避免跑到一半才失敗
    if not args.skip_check:
        check_dependencies()

    cli = WikiCLI(lang=args.lang)

    # 上傳相關的共用參數
    upload_kwargs = dict(
        token=args.hf_token,
        dry_run=args.dry_run,
        refresh_cards=args.refresh_cards,
        archive=not args.no_archive,
        archive_as=args.archive_as,
        skip_checks=args.skip_checks,
        force=args.force_upload,
    )

    # 如果沒有指定任何操作，預設執行文字 Pretrain Dataset 任務
    if not any([args.pretrain_dataset, args.image_dataset, args.download, args.to_md, args.to_pretrain,
                args.extract_images, args.download_images, args.generate_all, args.upload]):
        print("未指定任務，執行預設任務：生成文字 Pretrain Dataset\n")
        args.pretrain_dataset = True

    try:
        # 只上傳，不重新生成
        if args.upload and not args.generate_all:
            ok = cli.upload_to_hf(dump_date=args.dump_date, **upload_kwargs)
            sys.exit(0 if ok else 1)

        # 執行獨立任務
        if args.generate_all:
            # 一次性生成：下載 → 轉 MD → 為 tw/cn 各自生成 Parquet → 提取圖片資訊 → 上傳 HF
            print("任務：一次生成繁體、簡體與圖片資訊（不含下載）")

            # 生成要跑好幾個小時，先確認 HF 認證與權限沒問題，避免最後一步才失敗
            if not args.no_upload:
                from hf_uploader import preflight_check

                if not preflight_check(langs=('tw', 'cn'), token=args.hf_token):
                    print("✗ Hugging Face 檢查未通過，已中止")
                    print("  若只想生成不上傳，可加上 --no-upload")
                    sys.exit(1)

            # 下載一次 XML
            xml_path, latest_date = cli.download_wiki()
            if xml_path is None:
                return

            # 解析成中間層（僅需執行一次，tw/cn 共用）
            md_dir = cli.convert_to_md(xml_path, latest_date, force=args.force_md)
            if md_dir is None:
                return

            # 四組資料集（tw/cn × 純文字/omni）一次走完中間層。逐語言各跑一次的話
            # 同一頁要重複組裝四次，而繁簡兩版的差別只在字體。
            if cli.convert_md_to_datasets(input_dir=md_dir, dump_date=latest_date) is None:
                print("✗ Parquet 資料集生成失敗，中止流程")
                sys.exit(1)

            # 圖片資訊仍逐語言抽（圖說要各自轉換）
            for lang in ["tw", "cn"]:
                print(f"\n-- 提取圖片資訊: {lang} --")
                if WikiCLI(lang=lang).extract_images(xml_path=xml_path) is None:
                    print(f"✗ {lang} 的圖片資訊提取失敗，中止流程")
                    sys.exit(1)

            print("\n" + "=" * 60)
            print("✓ generate-all 生成完成！")
            print("=" * 60)

            # 生成完直接上傳（--no-upload 可停在這一步）
            if args.no_upload:
                print("\n已指定 --no-upload，略過 Hugging Face 上傳")
                print("之後可執行 `python wiki_cli.py --upload` 補上傳")
                return

            ok = cli.upload_to_hf(dump_date=args.dump_date or latest_date, **upload_kwargs)
            sys.exit(0 if ok else 1)

        if args.pretrain_dataset:
            cli.run_task('text')
        elif args.image_dataset:
            cli.run_task('image-dataset')
        elif args.download_images:
            cli.download_images()
        else:
            # 單獨步驟（進階功能）
            if args.download:
                cli.download_wiki()

            if args.to_md:
                cli.convert_to_md(xml_path=args.xml_path, force=args.force_md)

            if args.to_pretrain:
                cli.convert_md_to_dataset(input_dir=args.md_dir, dump_date=args.dump_date)

            if args.extract_images:
                cli.extract_images(xml_path=args.xml_path, output_file=args.image_json, max_images=args.max_images)

    except KeyboardInterrupt:
        print("\n\n⚠ 用戶中斷操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
