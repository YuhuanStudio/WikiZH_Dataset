import os
import json
import requests
import time
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_images_from_jsonl(jsonl_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    failed_files = []
    tasks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    pbar = tqdm(total=len(lines), desc=f"下載 {os.path.basename(jsonl_path)}", unit="img", dynamic_ncols=True)

    import random
    # 產生大量 user agent
    user_agents = []
    chrome_versions = [str(v) for v in range(80, 121)]
    firefox_versions = [str(v) for v in range(70, 116)]
    safari_versions = [str(v) for v in range(10, 17)]
    os_list = [
        "Windows NT 10.0; Win64; x64",
        "Macintosh; Intel Mac OS X 10_15_7",
        "X11; Linux x86_64",
        "Linux; Android 10; SM-G975F",
        "iPhone; CPU iPhone OS 15_0 like Mac OS X"
    ]
    for os_str in os_list:
        for cv in chrome_versions:
            user_agents.append(f"Mozilla/5.0 ({os_str}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{cv}.0.0.0 Safari/537.36")
        for fv in firefox_versions:
            user_agents.append(f"Mozilla/5.0 ({os_str}; rv:{fv}.0) Gecko/20100101 Firefox/{fv}.0")
        for sv in safari_versions:
            user_agents.append(f"Mozilla/5.0 ({os_str}) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{sv}.0 Safari/605.1.15")
    # 再加一些常見 UA
    user_agents += [
        "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
        "Mozilla/5.0 (Linux; Android 9; SM-G960F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.105 Mobile Safari/537.36",
        "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15A5341f Safari/604.1"
    ]
    import shutil
    def download_one(data):
        url = data.get('url')
        file_name = data.get('file_name')
        if not url or not file_name:
            return None, None, 'no_url_or_name'
        # 處理 file_name 前的 image/ 路徑
        if file_name.startswith('image/'):
            short_name = file_name[len('image/'):]
        else:
            short_name = file_name
        out_path = os.path.join(output_dir, short_name)
        # 檢查 output_dir 是否已存在
        if os.path.exists(out_path):
            return file_name, 'exists', None
        # 檢查舊版資料夾是否有檔案
        old_image_dir = 'old_images'
        old_path = os.path.join(old_image_dir, short_name)
        if os.path.exists(old_path):
            try:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                shutil.copy2(old_path, out_path)
                return file_name, 'copied', None
            except Exception as e:
                return file_name, 'fail', f'copy_error: {e}'
        headers = {
            "User-Agent": random.choice(user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "keep-alive",
            "Referer": "https://zh.wikipedia.org/",
            "Cache-Control": "no-cache"
        }
        success = False
        last_error = None
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code == 200:
                    with open(out_path, 'wb') as img_file:
                        img_file.write(resp.content)
                    success = True
                    return file_name, 'success', None
                elif resp.status_code == 429:
                    last_error = f"狀態碼: {resp.status_code} Too Many Requests (重試 {attempt+1})"
                    time.sleep(3 + attempt * 2)  # 429 時退避更久
                else:
                    last_error = f"狀態碼: {resp.status_code} (重試 {attempt+1})"
                    time.sleep(0.5)
            except Exception as e:
                last_error = f"錯誤: {e} (重試 {attempt+1})"
                time.sleep(0.5)
        return file_name, 'fail', last_error

    with ThreadPoolExecutor(max_workers=15) as executor:  # 降低併發數，減少被封鎖
        future_to_data = {}
        for line in lines:
            try:
                data = json.loads(line)
                file_name = data.get('file_name')
                out_path = os.path.join(output_dir, file_name) if file_name else None
                if out_path and os.path.exists(out_path):
                    # 已存在，直接更新進度條
                    pbar.set_postfix({"狀態": "已存在", "檔案": file_name, "失敗累計": len(failed_files)})
                    pbar.update(1)
                else:
                    future = executor.submit(download_one, data)
                    future_to_data[future] = data
            except Exception:
                pbar.update(1)
        for future in as_completed(future_to_data):
            file_name, status, err = future.result()
            if status == 'fail':
                failed_files.append(file_name)
                pbar.set_postfix({"狀態": "下載失敗", "檔案": file_name, "失敗累計": len(failed_files)})
            elif status == 'success':
                pbar.set_postfix({"狀態": "下載成功", "檔案": file_name, "失敗累計": len(failed_files)})
            elif status == 'exists':
                pbar.set_postfix({"狀態": "已存在", "檔案": file_name, "失敗累計": len(failed_files)})
            elif status == 'copied':
                pbar.set_postfix({"狀態": "已複製(舊版)", "檔案": file_name, "失敗累計": len(failed_files)})
            pbar.update(1)
    return failed_files

