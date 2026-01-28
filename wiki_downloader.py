"""
維基百科中文數據下載模組
功能：下載最新的維基百科中文數據庫轉儲文件
"""

import os
import re
import urllib.request
from datetime import datetime


class WIKIDownload(object):

    DUMPS_WIKI = 'https://dumps.wikimedia.org/zhwiki'
    LATEST_DIR = 'https://dumps.wikimedia.org/zhwiki/latest'
    XML_FILE = 'zhwiki-latest-pages-articles.xml.bz2'

    def __init__(self, output_dir):

        self.output_dir = output_dir
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        return

    def __get_file_date(self):
        """通過訪問 zhwiki 主頁面來獲取最新日期目錄"""
        try:
            # 訪問 zhwiki 主頁面，獲取所有日期目錄
            req = urllib.request.Request(self.DUMPS_WIKI + '/')
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req) as response:
                html_content = response.read().decode('utf-8')
                
                # 匹配所有日期目錄（格式：20250101/），使用更精確的模式
                # 匹配 [20250101/] 這種格式
                date_matches = re.findall(r'\[(\d{8})/\]', html_content)
                
                # 找到最新的日期
                if date_matches:
                    latest_date = max(date_matches)
                    return latest_date
                else:
                    # 備用方案：如果無法從主頁面獲取，嘗試從 latest/ 目錄獲取
                    req_latest = urllib.request.Request(self.LATEST_DIR)
                    req_latest.add_header('User-Agent', 'Mozilla/5.0')
                    with urllib.request.urlopen(req_latest) as response_latest:
                        latest_html = response_latest.read().decode('utf-8')
                        # 從 latest/ 目錄的文件日期中提取日期
                        date_match = re.search(r'(\d{2}-[A-Za-z]{3}-\d{4})', latest_html)
                        if date_match:
                            # 轉換日期格式（例如：01-Jan-2026 -> 20260101）
                            date_str = date_match.group(1)
                            try:
                                # 使用英文月份解析
                                date_obj = datetime.strptime(date_str, '%d-%b-%Y')
                                return date_obj.strftime('%Y%m%d')
                            except Exception as parse_err:
                                print(f"日期解析失敗: {parse_err}")
                    
                    # 最後備用方案：使用當前日期
                    return datetime.now().strftime('%Y%m%d')
        except Exception as e:
            print(f"無法獲取文件日期，使用當前日期: {e}")
            return datetime.now().strftime('%Y%m%d')

    def __get_remote_file_size(self, url):
        """獲取遠程文件的大小（字節）"""
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            req.get_method = lambda: 'HEAD'
            
            with urllib.request.urlopen(req) as response:
                size = int(response.headers.get('content-length', 0))
                return size
        except Exception as e:
            print(f"無法獲取遠程文件大小: {e}")
            return 0

    def __download_with_progress(self, url, output_path, verbose=True):
        """使用 urllib 下載文件並顯示進度"""
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get('content-length', 0))
                
                if not verbose:
                    # 靜默模式，直接下載
                    with open(output_path, 'wb') as f:
                        f.write(response.read())
                    return output_path
                
                # 顯示進度條
                chunk_size = 8192
                downloaded = 0
                
                print(f"\n開始下載: {os.path.basename(output_path)}")
                if total_size > 0:
                    print(f"總大小: {total_size / (1024*1024*1024):.2f} GB")
                else:
                    print("總大小: 未知")
                
                with open(output_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 顯示進度
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            downloaded_gb = downloaded / (1024*1024*1024)
                            total_gb = total_size / (1024*1024*1024)
                            print(f"\r下載進度: {progress:.1f}% ({downloaded_gb:.2f}/{total_gb:.2f} GB)", end='')
                        else:
                            downloaded_gb = downloaded / (1024*1024*1024)
                            print(f"\r已下載: {downloaded_gb:.2f} GB", end='')
                
                print()  # 換行
                
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"下載失敗: {e}")

    def __check_file_exists(self, file_path, expected_size=None):
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
        
        # 如果提供了預期大小，進行比對
        if expected_size is not None:
            if file_size == expected_size:
                return True, True  # 文件存在且完整
            else:
                # 顯示大小差異信息
                size_diff = abs(file_size - expected_size) / (1024*1024*1024)
                print(f"  警告: 本地文件大小與官方不匹配")
                print(f"    本地: {file_size / (1024*1024*1024):.2f} GB")
                print(f"    官方: {expected_size / (1024*1024*1024):.2f} GB")
                print(f"    差異: {size_diff:.2f} GB")
                return True, False  # 文件存在但不完整
        
        return True, True  # 文件存在且未提供預期大小時假設完整

    def run(self, verbose=True):
        """執行下載，返回文件路徑和日期"""
        
        # 獲取文件日期
        file_date = self.__get_file_date()
        print(f"檢測到的文件日期: {file_date}")
        
        # 創建帶日期的輸出目錄
        date_output_dir = os.path.join(self.output_dir, file_date)
        if not os.path.isdir(date_output_dir):
            os.makedirs(date_output_dir)
        
        # 文件名（使用實際日期命名）
        xml_file_name = f'zhwiki-{file_date}-pages-articles.xml.bz2'
        xml_bz2_path = os.path.join(date_output_dir, xml_file_name)
        
        # 構建下載 URL（從 latest/ 下載，文件名固定）
        download_url = f'{self.LATEST_DIR}/{self.XML_FILE}'
        
        # 獲取遠程文件大小
        print(f"正在獲取遠程文件信息...")
        remote_file_size = self.__get_remote_file_size(download_url)
        if remote_file_size > 0:
            print(f"官方文件大小: {remote_file_size / (1024*1024*1024):.2f} GB")
        
        # 檢查文件是否已存在且完整
        file_exists, file_complete = self.__check_file_exists(xml_bz2_path, remote_file_size)
        
        if file_exists:
            if file_complete:
                local_size_gb = os.path.getsize(xml_bz2_path) / (1024*1024*1024)
                print(f"✓ 檢測到已存在的完整文件: {xml_file_name}")
                print(f"  本地文件大小: {local_size_gb:.2f} GB")
                print(f"  文件驗證通過，跳過下載")
                return xml_bz2_path, file_date
            else:
                print(f"⚠ 本地文件不完整或大小不符，需要重新下載")
                # 刪除不完整的文件
                try:
                    os.remove(xml_bz2_path)
                    print(f"  已刪除不完整文件")
                except Exception as e:
                    print(f"  刪除文件失敗: {e}")
        
        # 下載文件
        xml_bz2_path = self.__download_with_progress(download_url, xml_bz2_path, verbose)
        
        # 下載完成後驗證文件大小
        if remote_file_size > 0:
            local_size = os.path.getsize(xml_bz2_path)
            if local_size == remote_file_size:
                print(f"✓ 下載完成並通過驗證")
            else:
                print(f"⚠ 警告: 下載的文件大小與官方不符")
                print(f"  下載: {local_size / (1024*1024*1024):.2f} GB")
                print(f"  官方: {remote_file_size / (1024*1024*1024):.2f} GB")
        
        return xml_bz2_path, file_date
