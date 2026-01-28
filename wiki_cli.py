"""
維基百科中文數據處理 CLI
統一的命令行入口，支援以下功能：

獨立任務（主要功能）：
1. 文字 Dataset：下載 → 轉 MD → 轉 JSONL（預設）
2. 圖片 Dataset：提取圖片資訊 JSONL
3. 下載圖片：根據圖片資訊下載

單獨步驟（進階功能）：
- 下載維基百科數據
- 將 XML 轉換為 Markdown
- 將 Markdown 轉換為 JSONL
- 從 XML 提取圖片資訊
- 下載圖片
"""

import os
import sys
import argparse
import shutil


class WikiCLI:
    """維基百科數據處理統一入口"""

    def __init__(self, lang='tw'):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.lang = lang  # 'tw' for Traditional Chinese, 'cn' for Simplified Chinese
        self.download_dir = os.path.join(self.base_dir, 'downloads')
        self.md_dir = os.path.join(self.base_dir, 'markdown')  # Markdown 是語言無關的（共享）
        self.json_dir = os.path.join(self.base_dir, f'output/{lang}')
        self.image_dir = os.path.join(self.base_dir, f'images/{lang}')

    def _ensure_dir(self, dir_path):
        """確保目錄存在"""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _get_latest_md_dir(self):
        """獲取最新的 Markdown 目錄"""
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

            return xml_path, latest_date
        except Exception as e:
            print(f"✗ 下載失敗: {e}")
            return None, None

    def convert_to_md(self, xml_path=None, latest_date=None):
        """將 XML 轉換為 Markdown"""
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
        print("步驟 2/2: 將 XML 轉換為 Markdown")
        print("=" * 60)

        # 創建帶月份標記的輸出目錄（格式：YYYYMM）
        simplified_date = latest_date[:4] + latest_date[4:6]
        output_dir_with_month = os.path.join(self.md_dir, simplified_date)
        print(f"輸出目錄: {output_dir_with_month}")

        try:
            # 刪除比當前月份更早的所有舊月份資料夾
            if os.path.exists(self.md_dir):
                old_months_deleted = []
                for folder_name in os.listdir(self.md_dir):
                    folder_path = os.path.join(self.md_dir, folder_name)
                    # 檢查是否是目錄且符合6位數字格式（YYYYMM）
                    if os.path.isdir(folder_path) and len(folder_name) == 6 and folder_name.isdigit():
                        # 比較月份，刪除比當前月份早的
                        if folder_name < simplified_date:
                            print(f"正在刪除舊月份資料夾: {folder_name}")
                            shutil.rmtree(folder_path)
                            old_months_deleted.append(folder_name)

                if old_months_deleted:
                    print(f"✓ 已刪除 {len(old_months_deleted)} 個舊月份資料夾")
                else:
                    print(f"沒有需要刪除的舊月份資料夾")

            # 創建新的輸出目錄
            self._ensure_dir(output_dir_with_month)

            # 使用 WIKIParse2Doc 類解析並生成文件
            from md_converter import WIKIParse2Doc

            parser = WIKIParse2Doc(xml_path, output_dir_with_month, markdown=True)
            parser.run(num=None)

            # 統計生成的文件數量
            md_count = len([f for f in os.listdir(output_dir_with_month) if f.endswith('.md')])

            print("\n" + "=" * 60)
            print("✓ 完成！")
            print(f"  輸出目錄: {output_dir_with_month}")
            print(f"  生成的 .md 文件數量: {md_count}")
            print("=" * 60)

            return output_dir_with_month
        except Exception as e:
            print(f"✗ 轉換失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

    def convert_md_to_json(self, input_dir=None, output_file=None):
        """將 Markdown 轉換為 JSONL"""
        if input_dir is None:
            input_dir = self._get_latest_md_dir()
            if input_dir is None:
                print("✗ 找不到 Markdown 目錄，請先執行轉換")
                return None

        if output_file is None:
            self._ensure_dir(self.json_dir)
            # 預設輸出檔名為 wiki_pretrain.json（會自動分割為 part1, part2...）
            output_file = os.path.join(self.json_dir, 'wiki_pretrain.json')

        print("=" * 60)
        print("將 Markdown 轉換為 Pretrain JSON")
        print("=" * 60)
        print(f"輸入目錄: {input_dir}")
        print(f"輸出文件: {output_file}")
        print(f"語言版本: {'繁體中文' if self.lang == 'tw' else '簡體中文'}")

        try:
            from md_to_json import process_directory

            process_directory(
                input_dir=input_dir,
                output_file=output_file,
                output_format="json",
                max_size_mb=500,
                num_workers=None,
                max_files=None,
                target_keywords=None,
                use_opencc=True,
                min_length=10,
                lang=self.lang
            )

            print("\n" + "=" * 60)
            print("✓ 完成！")
            print("=" * 60)

            return output_file
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
            output_file = os.path.join(self.json_dir, 'wiki_images_dataset.jsonl')

        print("=" * 60)
        print("從 XML 提取圖片資訊")
        print("=" * 60)
        print(f"輸入文件: {xml_path}")
        print(f"輸出文件: {output_file}")
        print(f"語言版本: {'繁體中文' if self.lang == 'tw' else '簡體中文'}")

        try:
            from image_extractor import extract_wiki_images

            extract_wiki_images(xml_path, output_file, max_images=max_images, lang=self.lang)

            print("\n" + "=" * 60)
            print("✓ 完成！")
            print("=" * 60)

            return output_file
        except Exception as e:
            print(f"✗ 提取失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

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
                - 'text': 文字 Dataset（下載 → 轉 MD → 轉 JSONL）
                - 'image-dataset': 圖片 Dataset（提取圖片資訊）
                - 'download-images': 下載圖片
        """
        print("\n" + "=" * 60)
        print("維基百科中文數據處理流程")
        print("=" * 60)

        if task == 'text':
            # 文字 Dataset 任務
            print("任務：生成文字 Pretrain Dataset")
            print("步驟：下載 → 轉 MD → 轉 Pretrain JSON")
            print()

            # 步驟 1: 下載
            xml_path, latest_date = self.download_wiki()
            if xml_path is None:
                return

            # 步驟 2: 轉換為 Markdown
            md_dir = self.convert_to_md(xml_path, latest_date)
            if md_dir is None:
                return

            # 步驟 3: 轉換為 Pretrain JSON
            self.convert_md_to_json(md_dir)

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

    將 XML 轉換為 Markdown:
        python wiki_cli.py --to-md

    將 Markdown 轉換為 Pretrain JSON:
        python wiki_cli.py --to-pretrain

    提取圖片資訊:
        python wiki_cli.py --extract-images
                """
    )

    # 獨立任務參數
    parser.add_argument('--pretrain-dataset', action='store_true', help='生成文字 Pretrain Dataset（下載 → 轉 MD → 轉 Pretrain JSON，預設）')
    parser.add_argument('--image-dataset', action='store_true', help='生成圖片 Dataset（提取圖片資訊 JSONL）')

    # 單獨步驟參數（進階功能）
    parser.add_argument('--download', action='store_true', help='下載維基百科數據')
    parser.add_argument('--to-md', action='store_true', help='將 XML 轉換為 Markdown')
    parser.add_argument('--to-pretrain', action='store_true', help='將 Markdown 轉換為 Pretrain JSON')
    parser.add_argument('--extract-images', action='store_true', help='從 XML 提取圖片資訊')
    parser.add_argument('--download-images', action='store_true', help='下載圖片')

    #（已移除舊版相容參數及 skip-* 選項，介面已簡化）

    # 語言參數
    parser.add_argument('--lang', type=str, choices=['tw', 'cn'], default='tw',
                    help='輸出語言版本 (tw=繁體中文, cn=簡體中文, 預設: tw)')

    # 路徑參數
    parser.add_argument('--xml-path', type=str, help='指定 XML 文件路徑')
    parser.add_argument('--md-dir', type=str, help='指定 Markdown 目錄路徑')
    parser.add_argument('--json-file', type=str, help='指定 JSONL 輸出文件路徑')
    parser.add_argument('--image-json', type=str, help='指定圖片 JSONL 文件路徑')
    parser.add_argument('--image-dir', type=str, help='指定圖片輸出目錄')
    parser.add_argument('--max-images', type=int, help='最大圖片數量（用於 --extract-images）')

    args = parser.parse_args()

    cli = WikiCLI(lang=args.lang)

    # 如果沒有指定任何操作，預設執行文字 Pretrain Dataset 任務
    if not any([args.pretrain_dataset, args.image_dataset, args.download, args.to_md, args.to_pretrain, args.extract_images, args.download_images]):
        print("未指定任務，執行預設任務：生成文字 Pretrain Dataset\n")
        args.pretrain_dataset = True

    try:
        # 執行獨立任務
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
                cli.convert_to_md(xml_path=args.xml_path)

            if args.to_pretrain:
                cli.convert_md_to_json(input_dir=args.md_dir, output_file=args.json_file)

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
