"""依圖片資料集 JSONL 安全、可恢復地下載媒體檔。"""

import hashlib
import json
import os
import shutil
import time
import unicodedata
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

USER_AGENT = (
    "WikiZH-Dataset/2.0 "
    "(https://github.com/YuhuanStudio/WikiZH_Dataset)"
)
MAX_WORKERS = 8
PENDING_LIMIT = 512
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _safe_filename(file_name):
    """把維基檔名映射成單一、不可跳出輸出目錄的本地檔名。"""
    name = str(file_name or "")
    if name.lower().startswith("image/"):
        name = name[len("image/") :]

    # `%` 先跳脫，讓 `a/b` 與原本就叫 `a%2Fb` 的檔案不會碰撞。
    unsafe = set('/\\:*?"<>|%')
    pieces = []
    for char in name:
        if char in unsafe or ord(char) < 32:
            pieces.extend(f"%{byte:02X}" for byte in char.encode("utf-8"))
        else:
            pieces.append(char)
    safe = "".join(pieces).rstrip(" .")
    if not safe:
        safe = hashlib.sha256(name.encode("utf-8")).hexdigest()

    stem = safe.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED:
        safe = "_" + safe

    # NTFS/ext4 常見的單一檔名上限是 255 bytes；預留副檔名與暫存後綴空間。
    if len(os.fsencode(safe)) > 220:
        root, ext = os.path.splitext(safe)
        suffix = "-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        while root and len(os.fsencode(root + suffix + ext)) > 220:
            root = root[:-1]
        safe = (root or "image") + suffix + ext
    return safe


def _is_complete(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _is_allowed_source_url(url):
    """圖片清單只允許本專案產生的 Wikipedia Special:FilePath URL。"""
    try:
        parsed = urllib.parse.urlsplit(str(url))
    except ValueError:
        return False
    return (parsed.scheme == 'https' and parsed.hostname == 'zh.wikipedia.org'
            and parsed.path.startswith('/wiki/Special:FilePath/'))


def _collision_safe_filename(file_name, used, resolved):
    """在大小寫不敏感的檔案系統上也維持一對一映射。"""
    if file_name in resolved:
        return resolved[file_name]
    safe = _safe_filename(file_name)
    key = unicodedata.normalize('NFC', safe).casefold()
    owner = used.get(key)
    if owner is not None and owner != file_name:
        root, ext = os.path.splitext(safe)
        digest = hashlib.sha256(file_name.encode('utf-8')).hexdigest()
        for length in (12, 20, 32, 64):
            candidate = f'{root}-{digest[:length]}{ext}'
            key = unicodedata.normalize('NFC', candidate).casefold()
            if key not in used:
                safe = candidate
                break
        else:  # SHA-256 全長仍碰撞在實務上不可達；明確失敗勝過覆寫別人的檔。
            raise ValueError(f'本地圖片檔名碰撞: {file_name!r}')
    used[key] = file_name
    resolved[file_name] = safe
    return safe


def download_images_from_jsonl(jsonl_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)
    failed_files = []
    used_local_names = {}
    resolved_local_names = {}

    # 逐行串流讀取；第一趟只計數，避免把數百 MB JSONL 一次載入記憶體。
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        total_lines = sum(1 for _ in fh)
    pbar = tqdm(
        total=total_lines,
        desc=f"下載 {os.path.basename(jsonl_path)}",
        unit="img",
        dynamic_ncols=True,
    )

    def local_path(file_name):
        local_name = _collision_safe_filename(
            file_name, used_local_names, resolved_local_names)
        path = os.path.abspath(os.path.join(output_dir, local_name))
        if os.path.commonpath((output_dir, path)) != output_dir:
            raise ValueError(f"不安全的圖片檔名: {file_name!r}")
        return path

    def download_one(data):
        url = data.get("url")
        file_name = data.get("file_name")
        if not url or not file_name:
            return file_name, "fail", "缺少 url 或 file_name"
        if not _is_allowed_source_url(url):
            return file_name, "fail", f"不允許的圖片來源 URL: {url!r}"

        try:
            out_path = local_path(file_name)
        except ValueError as exc:
            return file_name, "fail", str(exc)
        if _is_complete(out_path):
            return file_name, "exists", None

        old_path = os.path.join("old_images", os.path.basename(out_path))
        if _is_complete(old_path):
            try:
                shutil.copy2(old_path, out_path)
                return file_name, "copied", None
            except OSError as exc:
                return file_name, "fail", f"複製舊檔失敗: {exc}"

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "image/*,video/*,audio/*,application/pdf;q=0.9,*/*;q=0.1",
        }
        partial_path = out_path + ".part"
        last_error = None
        for attempt in range(3):
            try:
                with requests.get(
                    url, headers=headers, timeout=(15, 60), stream=True
                ) as response:
                    if response.status_code == 429:
                        last_error = f"HTTP 429（第 {attempt + 1} 次）"
                        time.sleep(3 + attempt * 2)
                        continue
                    if response.status_code != 200:
                        last_error = f"HTTP {response.status_code}（第 {attempt + 1} 次）"
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    content_type = response.headers.get("content-type", "").lower()
                    allowed_types = (
                        'image/', 'video/', 'audio/', 'application/pdf',
                        'application/ogg', 'application/octet-stream',
                    )
                    if (content_type
                            and not content_type.startswith(allowed_types)):
                        raise ValueError(
                            f"伺服器回傳非媒體內容: {content_type}")

                    expected = int(response.headers.get("content-length", 0) or 0)
                    written = 0
                    with open(partial_path, "wb") as image_file:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                image_file.write(chunk)
                                written += len(chunk)
                    if written == 0:
                        raise ValueError("伺服器回傳空檔案")
                    if expected and written != expected:
                        raise ValueError(f"檔案被截斷: {written} != {expected}")
                    os.replace(partial_path, out_path)
                    return file_name, "success", None
            except (OSError, requests.RequestException, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}（第 {attempt + 1} 次）"
                try:
                    if os.path.exists(partial_path):
                        os.remove(partial_path)
                except OSError:
                    pass
                time.sleep(0.5 * (attempt + 1))
        return file_name, "fail", last_error

    def report(file_name, status):
        label = {
            "fail": "下載失敗",
            "success": "下載成功",
            "exists": "已存在",
            "copied": "已複製(舊版)",
        }.get(status, status)
        pbar.set_postfix(
            {"狀態": label, "檔案": file_name or "?", "失敗累計": len(failed_files)}
        )
        pbar.update(1)

    seen = set()
    errors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        pending = {}

        def drain(limit):
            while len(pending) >= limit and pending:
                done = next(as_completed(tuple(pending)))
                file_name, status, error = done.result()
                pending.pop(done, None)
                if status == "fail":
                    failed_files.append(file_name)
                    if len(errors) < 20:
                        errors.append((file_name, error))
                report(file_name, status)

        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, 1):
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    failed_files.append(f"第 {line_number} 行")
                    if len(errors) < 20:
                        errors.append((f"第 {line_number} 行", f"JSON 錯誤: {exc}"))
                    pbar.update(1)
                    continue
                file_name = data.get("file_name")
                if not file_name:
                    failed_files.append(f"第 {line_number} 行")
                    pbar.update(1)
                    continue
                if file_name in seen:
                    report(file_name, "exists")
                    continue
                seen.add(file_name)
                try:
                    out_path = local_path(file_name)
                except ValueError as exc:
                    failed_files.append(file_name)
                    errors.append((file_name, str(exc)))
                    report(file_name, "fail")
                    continue
                if _is_complete(out_path):
                    report(file_name, "exists")
                    continue
                pending[executor.submit(download_one, data)] = file_name
                drain(PENDING_LIMIT)

        drain(1)

    pbar.close()
    if errors:
        print(f"⚠ {len(failed_files):,} 筆下載失敗；前 {len(errors)} 筆：")
        for file_name, error in errors:
            print(f"  {file_name}: {error}")
    return failed_files
