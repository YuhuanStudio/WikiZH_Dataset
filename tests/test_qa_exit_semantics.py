import json
import os
# Tests invoke fixed repository QA scripts without a shell.
import subprocess  # nosec B404
import sys
import tempfile
import unittest
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_qa(script, *args):
    # Interpreter and script paths are fixed by the test repository.
    return subprocess.run(  # nosec B603
        [sys.executable, os.path.join(ROOT, 'qa', script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class InvariantExitSemanticsTests(unittest.TestCase):
    def _run_rows(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            pq.write_table(pa.Table.from_pylist(rows),
                           os.path.join(directory, 'part.parquet'))
            return run_qa('invariants.py', directory, '10')

    def test_formula_source_warnings_do_not_block(self):
        result = self._run_rows([
            {
                'title': '括號不平衡',
                'text': '括號不平衡\n\n來源公式 $\\frac{a}{b$。',
            },
            {
                'title': '分隔符缺漏',
                'text': '分隔符缺漏\n\n來源直接顯示 \\frac 命令。',
            },
        ])

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('來源品質警示（不阻擋出貨）', result.stdout)
        self.assertIn('需核對來源', result.stdout)

    def test_structural_violation_still_blocks(self):
        result = self._run_rows([{
            'title': '測試',
            'text': '測試\n\n```\n未關閉的程式碼圍欄',
        }])

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('有違反的阻斷性不變量', result.stdout)


class ImageAuditExitSemanticsTests(unittest.TestCase):
    _BASE: ClassVar = {
        'url': 'https://zh.wikipedia.org/wiki/Special:FilePath/example.jpg',
        'file_name': 'example.jpg',
        'caption': '圖說',
        'alt': '替代文字',
        'page': '測試',
        'page_id': '1',
        'page_url': 'https://zh.wikipedia.org/wiki/example',
        'section': '',
    }

    def _run_content(self, content):
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.jsonl', encoding='utf-8', delete=False) as fh:
            fh.write(content)
            path = fh.name
        try:
            return run_qa('image_audit.py', path, 'tw')
        finally:
            os.unlink(path)

    def _run_row(self, row=None, **changes):
        row = dict(self._BASE if row is None else row)
        row.update(changes)
        return self._run_content(json.dumps(row, ensure_ascii=False) + '\n')

    def test_clean_row_succeeds(self):
        result = self._run_row()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_empty_dataset_blocks(self):
        result = self._run_content('')
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('圖片資料集為空', result.stdout)

    def test_every_reported_defect_blocks(self):
        missing_field = dict(self._BASE)
        del missing_field['file_name']
        cases = (
            ('私有區字元', {'caption': '\ue000'}),
            ('控制字元', {'caption': '\x01'}),
            ('殘留標記', {'caption': '{{殘留標記}}'}),
            ('欄位缺漏', missing_field),
            ('沒有任何文字', {'caption': '', 'alt': ''}),
            ('網址不合法', {'url': 'http://example.test/image.jpg'}),
        )
        for name, changes in cases:
            with self.subTest(defect=name):
                if name == '欄位缺漏':
                    result = self._run_row(row=changes)
                else:
                    result = self._run_row(**changes)
                self.assertEqual(
                    result.returncode, 1, result.stdout + result.stderr)

        result = self._run_content('{JSON 壞掉\n')
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

        duplicate = json.dumps(self._BASE, ensure_ascii=False) + '\n'
        result = self._run_content(duplicate + duplicate)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('重複記錄', result.stdout)


if __name__ == '__main__':
    unittest.main()
