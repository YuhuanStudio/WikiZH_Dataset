# QA 工具

這個目錄只把可重現、可作為出貨閘門的工具納入版控。大型 dump 衍生檔、候選資料、
稽核日誌、Python 快取與一次性調查結果均由 `.gitignore` 排除，不應長期堆在 `qa/`。

## 核心閘門

| 工具 | 用途 | 通過條件 |
|---|---|---|
| `cases.py` | 81 條行為黃金案例 | 全部通過 |
| `stress.py` | 28 項正則與解析壓力案例 | 全部在時間預算內 |
| `validate_full.py` | 全量正文格式、殘留與完整性 | 15 類硬性缺陷全為 0 |
| `invariants.py` | 標題、程式碼圍欄、空章節等結構不變量 | 阻斷性缺陷為 0；來源公式警示不阻擋 |
| `parity.py` | 繁簡 ID 與核心結構對等 | ID 序列一致、核心結構差異為 0 |
| `omni_audit.py` | omni 圖片佔位符、圖片欄位與純文字版對照 | 所有不變量為 0 |
| `image_audit.py` | 圖片 JSONL 的 8 類缺陷 | 8 類全部為 0，空檔也失敗 |
| `verify_fixes.py` | 回出貨 Parquet 驗證歷史修正 | 指定案例全部存在且符合預期 |

最小的本機檢查：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python qa/cases.py
.venv/bin/python qa/stress.py
```

候選資料生成後，從候選根目錄執行全量閘門，避免 `parity.py` 與 `omni_audit.py`
誤讀專案根目錄下的舊 `output/`：

```bash
cd scratchpad/rebuild-YYYYMMDD
PYTHONPATH=../.. ../../.venv/bin/python ../../qa/validate_full.py output/tw tw
PYTHONPATH=../.. ../../.venv/bin/python ../../qa/validate_full.py output/cn cn
PYTHONPATH=../.. ../../.venv/bin/python ../../qa/parity.py 20000
PYTHONPATH=../.. ../../.venv/bin/python ../../qa/omni_audit.py tw 0
PYTHONPATH=../.. ../../.venv/bin/python ../../qa/omni_audit.py cn 0
PYTHONPATH=../.. ../../.venv/bin/python ../../qa/image_audit.py output/tw/wiki_images_dataset.jsonl tw
PYTHONPATH=../.. ../../.venv/bin/python ../../qa/image_audit.py output/cn/wiki_images_dataset_CN.jsonl cn
```

## 保留與清理政策

- 當次出貨的 manifest 與 QA 日誌放在隔離候選目錄，不放在 `qa/`。
- `qa/*.log`、`qa/verify*/`、`qa/__pycache__/` 與過濾後 dump 都是可重建產物，可直接清理。
- 一次性探針若證明了新缺陷，應把最小重現案例移進 `cases.py` 或 `tests/`；探針輸出本身不保留。
- 文件中的發布數字以候選 manifest 與該輪 QA 日誌為準，不沿用前一輪 README 數字。

目前公開的 `2608` 修正版以 `2026-08-01` dump 建置：正文與 omni 各 1,482,182 篇；
圖片繁體 915,171 列、簡體 915,167 列。完整發布說明見上層 [README](../README.md)。
