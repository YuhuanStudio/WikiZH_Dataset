"""
維基百科中文數據下載模組
功能：下載最新的維基百科中文數據庫轉儲文件
"""

import hashlib
import os
import re
import time
import urllib.request


def _wait_for_visible_file(path, expected_size=None, attempts=50, delay=0.1):
    """等待 drvfs／9p 的重新命名結果對後續 stat 可見。

    WSL 掛載的 Windows 磁碟可能在 `os.replace()` 成功後短暫回傳 ENOENT；立即
    `getsize()` 會把已搬好的 3 GB dump 誤判成下載失敗。這裡只等待已完成的
    原子操作變得可見，不掩蓋大小錯誤或永久遺失。
    """
    for _ in range(attempts):
        try:
            size = os.path.getsize(path)
            if expected_size is None or size == expected_size:
                return size
            raise RuntimeError(
                f"重新命名後檔案大小不符: {size} != {expected_size}")
        except FileNotFoundError:
            time.sleep(delay)
    raise RuntimeError(f"重新命名後檔案仍不可見: {path}")


class WIKIDownload:
    DUMPS_WIKI = "https://dumps.wikimedia.org/zhwiki"
    LATEST_DIR = "https://dumps.wikimedia.org/zhwiki/latest"
    SHA1_FILE = "zhwiki-latest-sha1sums.txt"
    USER_AGENT = "WikiZH-Dataset/2.0 (https://github.com/YuhuanStudio/WikiZH_Dataset)"
    REQUEST_TIMEOUT = 60

    def __init__(self, output_dir):

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def __get_dump_info(self):
        """從官方 SHA-1 清單取得最新完成 dump 的日期、檔名與雜湊。

        目錄頁顯示的是檔案完成／修改日期，不是 dump 的快照日期。例如
        ``zhwiki-20260801-pages-articles.xml.bz2`` 可能在 8 月 4 日才完成。
        以修改日期命名會讓資料集把 8/1 的 dump 誤標成 8/4。
        """
        url = f"{self.LATEST_DIR}/{self.SHA1_FILE}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
            # URL 由上方固定的 Wikimedia HTTPS 常數組成，不接受外部 scheme。
            with urllib.request.urlopen(  # nosec B310
                    req, timeout=self.REQUEST_TIMEOUT) as response:
                manifest = response.read().decode("ascii")
        except Exception as e:
            raise RuntimeError(f"無法取得官方 SHA-1 清單: {e}") from e

        match = re.search(
            r"(?m)^([0-9a-f]{40})  (zhwiki-(\d{8})-pages-articles\.xml\.bz2)$",
            manifest,
        )
        if not match:
            raise RuntimeError("官方 SHA-1 清單中找不到 pages-articles dump")
        sha1, file_name, file_date = match.groups()
        return file_date, file_name, sha1

    def __get_remote_file_size(self, url):
        """獲取遠程文件的大小（字節）"""
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
            req.get_method = lambda: "HEAD"

            # 呼叫端只傳本類別組出的 Wikimedia HTTPS dump URL。
            with urllib.request.urlopen(  # nosec B310
                    req, timeout=self.REQUEST_TIMEOUT) as response:
                size = int(response.headers.get("content-length", 0))
                return size
        except Exception as e:
            print(f"無法獲取遠程文件大小: {e}")
            return 0

    def __download_with_progress(
        self, url, output_path, expected_size, expected_sha1, verbose=True
    ):
        """串流下載到暫存檔，驗證大小與 SHA-1 後再原子改名。"""
        partial_path = output_path + ".part"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
            # 呼叫端只傳本類別組出的 Wikimedia HTTPS dump URL。
            with urllib.request.urlopen(  # nosec B310
                    req, timeout=self.REQUEST_TIMEOUT) as response:
                total_size = int(response.headers.get("content-length", 0))
                if expected_size and total_size and total_size != expected_size:
                    raise RuntimeError(
                        f"HEAD 與下載回應的大小不一致: {expected_size} != {total_size}"
                    )

                chunk_size = 1024 * 1024
                downloaded = 0
                # Wikimedia 官方清單只提供 SHA-1；此處用於完整性比對，不作密碼學用途。
                digest = hashlib.sha1(usedforsecurity=False)

                if verbose:
                    print(f"\n開始下載: {os.path.basename(output_path)}")
                    size_for_display = expected_size or total_size
                    if size_for_display:
                        print(f"總大小: {size_for_display / (1024**3):.2f} GB")

                with open(partial_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        if verbose and (expected_size or total_size):
                            target = expected_size or total_size
                            progress = (downloaded / target) * 100
                            downloaded_gb = downloaded / (1024**3)
                            total_gb = target / (1024**3)
                            print(
                                f"\r下載進度: {progress:.1f}% ({downloaded_gb:.2f}/{total_gb:.2f} GB)",
                                end="",
                            )
                        elif verbose:
                            downloaded_gb = downloaded / (1024**3)
                            print(f"\r已下載: {downloaded_gb:.2f} GB", end="")

                if verbose:
                    print()

            if expected_size and downloaded != expected_size:
                raise RuntimeError(f"檔案大小不符: {downloaded} != {expected_size}")
            actual_sha1 = digest.hexdigest()
            if actual_sha1 != expected_sha1:
                raise RuntimeError(f"SHA-1 不符: {actual_sha1} != {expected_sha1}")
            os.replace(partial_path, output_path)
            _wait_for_visible_file(output_path, expected_size or downloaded)
            return output_path
        except Exception as e:
            try:
                if os.path.exists(partial_path):
                    os.remove(partial_path)
            except OSError:
                pass
            raise RuntimeError(f"下載失敗: {e}") from e

    @staticmethod
    def __sha1(file_path):
        digest = hashlib.sha1(usedforsecurity=False)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def __check_file_exists(self, file_path, expected_size=None, expected_sha1=None):
        """檢查文件是否存在且完整

        Args:
            file_path: 本地文件路徑
            expected_size: 預期的文件大小（字節），如果為None則不驗證大小

        Returns:
            tuple: (是否存在, 是否完整)
        """
        if not os.path.exists(file_path):
            return False, False

        # 檢查文件大小
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return True, False  # 文件存在但不完整

        if expected_size and file_size != expected_size:
            print("  警告: 本地檔案大小與官方不匹配")
            print(f"    本地: {file_size / (1024**3):.2f} GB")
            print(f"    官方: {expected_size / (1024**3):.2f} GB")
            return True, False
        if expected_sha1:
            print("  正在驗證既有檔案的 SHA-1…")
            if self.__sha1(file_path) != expected_sha1:
                print("  警告: 本地檔案 SHA-1 與官方不匹配")
                return True, False
        return True, True

    def __reuse_mislabeled_file(self, target, expected_size, expected_sha1):
        """沿用舊版以下載完成日誤標、但內容其實相同的本地 dump。"""
        if os.path.exists(target) or not expected_size:
            return False
        target_abs = os.path.abspath(target)
        for directory in sorted(os.listdir(self.output_dir), reverse=True):
            root = os.path.join(self.output_dir, directory)
            if not os.path.isdir(root):
                continue
            for name in os.listdir(root):
                candidate = os.path.join(root, name)
                if (os.path.abspath(candidate) == target_abs
                        or not re.fullmatch(
                            r'zhwiki-\d{8}-pages-articles\.xml\.bz2', name)
                        or os.path.getsize(candidate) != expected_size):
                    continue
                print(f"  發現舊版可能誤標日期的檔案，驗證 SHA-1: {candidate}")
                if self.__sha1(candidate) == expected_sha1:
                    os.replace(candidate, target)
                    _wait_for_visible_file(target, expected_size)
                    print(f"  ✓ 內容相同，已移到正確快照日期: {target}")
                    return True
        return False

    def run(self, verbose=True):
        """執行下載，返回文件路徑和日期"""

        file_date, remote_name, expected_sha1 = self.__get_dump_info()
        print(f"檢測到的文件日期: {file_date}")

        # 創建帶日期的輸出目錄
        date_output_dir = os.path.join(self.output_dir, file_date)
        if not os.path.isdir(date_output_dir):
            os.makedirs(date_output_dir)

        xml_file_name = remote_name
        xml_bz2_path = os.path.join(date_output_dir, xml_file_name)

        # 使用帶日期的固定 URL，避免 latest 在下載途中切換到下一版。
        download_url = f"{self.DUMPS_WIKI}/{file_date}/{remote_name}"

        # 獲取遠程文件大小
        print("正在獲取遠程文件信息...")
        remote_file_size = self.__get_remote_file_size(download_url)
        if remote_file_size > 0:
            print(f"官方文件大小: {remote_file_size / (1024 * 1024 * 1024):.2f} GB")

        reused = self.__reuse_mislabeled_file(
            xml_bz2_path, remote_file_size, expected_sha1)

        # 檢查文件是否已存在且完整
        if reused:
            file_exists, file_complete = True, True
        else:
            file_exists, file_complete = self.__check_file_exists(
                xml_bz2_path, remote_file_size, expected_sha1
            )

        if file_exists:
            if file_complete:
                local_size_gb = os.path.getsize(xml_bz2_path) / (1024 * 1024 * 1024)
                print(f"✓ 檢測到已存在的完整文件: {xml_file_name}")
                print(f"  本地文件大小: {local_size_gb:.2f} GB")
                print("  文件驗證通過，跳過下載")
                return xml_bz2_path, file_date
            else:
                print("⚠ 本地文件不完整或大小不符，需要重新下載")
                # 已確認是不完整／被破壞的檔案，刪除後重抓。
                try:
                    os.remove(xml_bz2_path)
                    print("  已刪除不完整文件")
                except OSError as e:
                    print(f"  刪除文件失敗: {e}")

        # 下載文件
        xml_bz2_path = self.__download_with_progress(
            download_url, xml_bz2_path, remote_file_size, expected_sha1, verbose
        )
        print("✓ 下載完成，檔案大小與 SHA-1 均通過驗證")
        return xml_bz2_path, file_date
