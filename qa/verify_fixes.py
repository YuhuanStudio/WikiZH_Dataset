"""逐項驗證修正是否真的進到資料裡

「已修」不等於「資料裡是對的」。每個修正都對應一篇實際壞過的條目，
這支直接去出貨的 Parquet 裡查那一篇，看該在的在不在、該消失的消失沒。

改完解析邏輯、重建完成後跑一次：python qa/verify_fixes.py [output/tw]
"""
import glob
import os
import re
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zhconv

OUT = sys.argv[1] if len(sys.argv) > 1 else 'output/tw'
_S2T = zhconv._Converter([zhconv._load()['zh2Hant']]).convert

# (條目, 必須出現, 不得出現, 說明)
CHECKS = [
    ('HTML', '<!-- This is a comment -->', None,
     '<pre> 裡的註解是內容，不能當註解刪'),
    ('Smalltalk', 'Answer a new instance', None,
     '<ref> 註腳裡的程式碼要撈出來'),
    ('博揚·波格丹諾維奇', '1989年4月18日', '1989年－4月18日',
     '{{bd}} 只有生日時不加破折號'),
    ('Standard ML', None, 'template below',
     'reflist 樣板註解要整段刪，不能只刪前半'),
    ('詩巫機場', None, '</ref> tags',
     '同上（跨越逐字區邊界的註解）'),
    ('亳州市', None, 'total_width',
     '巢狀跨行模板的版面參數不會變成事實'),
    ('亳州市', None, 'image_style',
     '同上（{{multiple image}} 的內部參數）'),
    ('亳州市', '市長：汪繼宏', '導演：汪繼宏',
     '動態標籤：參數清單完整時才查得到 leader_title2'),
    ('參宿四', '10.874', None,
     '註腳裡的行內公式也要撈出來（整份推導十條）'),
    ('2channel文字人物', '　（　　　　）', None,
     '顏文字的全形空格要逐字保留'),
]

# 這幾篇要兩版都查：單一語言看不出來的 bug 靠繁簡對比才現形
PARITY_CHECKS = [
    ('捷爾諾波爾州', 3, '自閉合 <ref> 不會吃掉後面的正文（章節數要一致）'),
    # 章節跳過的身分只比層級，不比標題文字——`連結`（繁）與 `链接`（簡）
    # 正規化後仍是不同字，比文字會讓兩版各自判斷，繁體留著、簡體丟掉
    ('辛貝特', 0, '章節跳過的決定與語言無關'),
    ('清華大學歷史系', 4, '同上（參考連結節兩版都要丟）'),
]


def load(out_dir, titles):
    want = set(titles)
    got = {}
    for path in sorted(glob.glob(os.path.join(out_dir, '*.parquet'))):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=20000, columns=['title', 'text']):
            titles_col = batch.column('title').to_pylist()
            for i, t in enumerate(titles_col):
                key = _S2T(t)
                if key in want and key not in got:
                    got[key] = batch.column('text')[i].as_py()
        if len(got) == len(want):
            break
    return got


def main():
    docs = load(OUT, [c[0] for c in CHECKS] + [c[0] for c in PARITY_CHECKS])
    ok = True

    print(f'=== 逐項驗證（{OUT}）===\n')
    for title, must, must_not, desc in CHECKS:
        text = docs.get(title)
        if text is None:
            print(f'  ？ {desc}\n      [{title}] 條目不在資料集裡')
            ok = False
            continue
        good = True
        if must and must not in text:
            good = False
        if must_not and must_not in text:
            good = False
        ok &= good
        print(f'  {"✓" if good else "✗"} {desc}')
        if not good:
            print(f'      [{title}] '
                  f'{"缺少 " + repr(must) if must and must not in text else ""}'
                  f'{"殘留 " + repr(must_not) if must_not and must_not in text else ""}')

    cn_dir = OUT.replace('/tw', '/cn')
    cn_docs = load(cn_dir, [c[0] for c in PARITY_CHECKS]) if os.path.isdir(cn_dir) else {}
    for title, want_heads, desc in PARITY_CHECKS:
        tw = docs.get(title, '')
        cn = cn_docs.get(title, '')
        n_tw = len(re.findall(r'(?m)^#{2,6} .+$', tw))
        n_cn = len(re.findall(r'(?m)^#{2,6} .+$', cn))
        good = n_tw == n_cn == want_heads
        ok &= good
        print(f'  {"✓" if good else "✗"} {desc}')
        if not good:
            print(f'      [{title}] 繁體 {n_tw} 節、簡體 {n_cn} 節（應為 {want_heads}）')

    print('\n✓ 全部驗證通過' if ok else '\n✗ 有修正沒有生效')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
