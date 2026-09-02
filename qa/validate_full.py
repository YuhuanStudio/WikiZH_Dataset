"""
全量校驗：對整份資料集逐筆檢查，不抽樣

分三類：
  A. 硬性缺陷  必須為 0，任何一筆不合格就不該出貨
  B. 完整性    開頭／結尾／結構
  C. 分布      長度、章節、型態（供人工判讀，非缺陷）
"""
import collections
import os
import re
import statistics
import sys

import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.environ.get('OUTPUT_ROOT', os.path.join(REPO_ROOT, 'output'))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUTPUT_ROOT, 'tw')
LANG = sys.argv[2] if len(sys.argv) > 2 else 'tw'

PUA = re.compile('[-]')
CJK = re.compile(r'[一-鿿]')

# 逐字區塊（``` 程式碼圍欄、$…$ 公式）裡的內容是頁面上真的會顯示的文字，
# 不是殘留標記。`<pre>` 裡的 `log<sub>2</sub>k`、C++ 的 `#include <vector>`、
# LaTeX 的 `\frac{{a}}{b}` 都會命中殘留 HTML／模板／表格的規則，全是誤報。
# 檢查殘留標記之前先把這些區塊挖掉。
_FENCE_RE = re.compile(r'(?s)```.*?```')
_INLINE_CODE_RE = re.compile(r'`[^`\n]*`')
# 公式本體可以合法包含跳脫的美元符號（`n\$` 是 expofactorial 的記號）。
# 不能把 `\$` 當成分隔符，否則完整公式會被切成兩半，外側的 LaTeX 又被
# 誤報成「未標記」。`\\.` 一次吃掉跳脫序列，也涵蓋 `\{`、`\}`。


def _escaped_at(text, index):
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == '\\':
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _math_spans(text):
    r"""線性掃描 `$…$`／`$$…$$`，略過 `\$`，回傳 (起、訖、內容)。

    用 tempered-dot 正則處理沒收尾的 `$$` 會在長條目反覆回溯，單篇就可能耗時
    數分鐘。狀態機每個字元最多走一次，並保持與 Markdown 分隔符相同的語意。
    """
    spans = []
    index = 0
    size = len(text)
    while index < size:
        start = text.find('$', index)
        if start < 0:
            break
        if _escaped_at(text, start):
            index = start + 1
            continue
        display = text.startswith('$$', start)
        cursor = start + (2 if display else 1)
        line_end = size if display else text.find('\n', cursor)
        if line_end < 0:
            line_end = size
        close = -1
        while cursor < line_end:
            found = text.find('$', cursor, line_end)
            if found < 0:
                break
            if _escaped_at(text, found):
                cursor = found + 1
                continue
            if display:
                if text.startswith('$$', found):
                    close = found
                    break
            elif not text.startswith('$$', found):
                close = found
                break
            cursor = found + 1
        if close < 0:
            # 未閉合的 display span 後面仍可能有正常公式；退一個 `$` 繼續找，
            # 但不重掃已確認沒有美元符號的長區段。
            index = start + (2 if display else 1)
            continue
        end = close + (2 if display else 1)
        spans.append((start, end, text[start + (2 if display else 1):close]))
        index = end
    return spans


def _mask_math(text):
    spans = _math_spans(text)
    if not spans:
        return text
    out = []
    last = 0
    for start, end, _body in spans:
        out.append(text[last:start])
        out.append(re.sub(r'[^\n]', 'x', text[start:end]))
        last = end
    out.append(text[last:])
    return ''.join(out)


def _blank(m):
    # 換成同長度的佔位而不是空字串，才不會把前後的行接在一起；也不能換成
    # 空格，否則整行被挖空後只剩一個空格，反而觸發「行尾空白」的誤報。
    return re.sub(r'[^\n]', 'x', m.group(0))


def strip_verbatim(text):
    """挖掉程式碼圍欄與公式，只留給殘留標記檢查用"""
    if '```' in text:
        text = _FENCE_RE.sub(_blank, text)
    if '$' in text:
        text = _mask_math(text)
    return _INLINE_CODE_RE.sub(_blank, text) if '`' in text else text


def flatten_verbatim(text):
    """同上，但圍欄整塊壓成一個字元——空白類的檢查要用這個版本

    程式碼裡的空行與縮排是內容（Python 的定義之間、組語的分節、XML 的區塊），
    保留換行去查「連續空行」等於把程式碼的排版當成缺陷。
    """
    if '```' in text:
        text = _FENCE_RE.sub('x', text)
    if '$' in text:
        text = _mask_math(text)
    return _INLINE_CODE_RE.sub(_blank, text) if '`' in text else text


# 數學式：公式一律以 $…$／$$…$$ 保留 LaTeX 原始碼。
# 公式外觀檢查是「來源品質警示」，不是出貨硬閘門：中文維基來源本身就有少量
# 不平衡 LaTeX，也會在說明文字中直接展示 `\pi`、`\begin{proof}` 等命令。
# 單看輸出無法分辨來源瑕疵與轉換瑕疵；後者由 math_audit 逐式和 dump 比對。
# 用列舉而不是 `\\[A-Za-z]{2,}`：後者會命中 Windows 路徑
# （`HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft`、`c:\\source.exe`）與日文名的
# 分隔符（`\\ファイヤーコンボイ\\`），那些不是公式。
_TEX_CMD_RE = re.compile(
    r'\\(?:frac|sqrt|sum|prod|int|over|underset|overset|mathrm|mathbb|mathcal'
    r'|text|begin|left|right|cdot|times|div|pm|approx|neq|leq|geq|equiv'
    r'|alpha|beta|gamma|delta|theta|lambda|mu|nu|pi|rho|sigma|tau|phi|omega'
    r'|infty|partial|nabla|to|rightarrow|leftarrow|uparrow|downarrow'
    r'|bar|hat|vec|dot|tilde|log|ln|exp|sin|cos|tan|lim|max|min|bigtriangleup)\b')


# LaTeX 的 `\{`／`\}` 是**字面**大括號（`\left\{`、`= \{(q_1,…`），
# 不是分組符號，計數時要先拿掉，否則全是誤報。
_ESCAPED_BRACE_RE = re.compile(r'\\[{}]')


def math_defects(text):
    """回傳 (含公式, 括號不平衡, 有公式卻沒標記分隔符)"""
    # 程式碼圍欄要先挖掉再找公式：Perl 的 `$foo`、AWK 的 `$1` 都是變數不是
    # 數學分隔符，不挖的話它們會兩兩配對，把整段程式碼當成一條公式。
    body = _FENCE_RE.sub(' ', text) if '```' in text else text
    # 絕大多數條目沒有公式。避免讓帶 tempered-dot 的完整公式正則對 148 萬篇
    # 文章逐字掃描；全量驗證由約兩小時降回可日常執行的量級。
    if '$' not in body:
        has_tex = bool(_TEX_CMD_RE.search(body))
        return has_tex, False, has_tex
    spans = [span_body for _start, _end, span_body in _math_spans(body)]
    if not spans:
        has_tex = bool(_TEX_CMD_RE.search(body))
        return has_tex, False, has_tex
    unbalanced = any(
        (t := _ESCAPED_BRACE_RE.sub('', sp)).count('{') != t.count('}')
        for sp in spans)
    return True, unbalanced, False


# 這幾項要在「圍欄壓成一行」的版本上查（程式碼裡的空行與縮排是內容）
_WHITESPACE_CHECKS = {'連續空行', '行尾空白'}

# A. 硬性缺陷：出現即為 bug
HARD = [
    # `min_length=0` 代表不以長度篩選，不代表允許只剩標題的空殼。
    # 這項由 main 直接檢查正文，正則刻意永不匹配。
    ('空正文',                 re.compile(r'(?!)')),
    ('殘留模板 {{ }}',      re.compile(r'\{\{|\}\}')),
    # 表格語法一定在行首。不限行首會誤判程式碼：Ruby 的
    # `hash.delete_if {|k,value|` 是條目內容，不是殘留的表格。
    ('殘留表格 {| |}',      re.compile(r'(?m)^\s*(?:\{\||\|\})')),
    ('殘留 wiki 連結',      re.compile(r'\[\[|\]\]')),
    ('殘留 ref 標籤',       re.compile(r'</?ref')),
    # 要求標籤真的有收尾的 `>`，否則數學式 `任何一個 A<B 或 B<A` 會被
    # 當成 <b> 標籤（條目《序拓撲》）。
    ('殘留 HTML 標籤',      re.compile(r'</?(?:div|span|table|tr|td|br|p|b|i|sup|sub|small)\b[^>\n]{0,200}>', re.I)),
    ('殘留 HTML 實體',      re.compile(r'&(?:lt|gt|amp|quot|nbsp|#\d+);')),
    ('私有區字元',          PUA),
    ('控制字元',            re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')),

    ('連續空行',            re.compile(r'\n\n\n')),
    ('行尾空白',            re.compile(r'[ \t]+\n')),
    # 只抓「明顯是 wiki 參數」的形態（帶引號的樣式值或 | 前綴）。
    # 原本的 `^key=value` 會把正文誤判：下推自動機的 `s0 = , s1 = $`、
    # 氫原子光譜的 `k=1,...,N`、OCaml 範例的 `let =` 都是條目內容。
    # 只認 wiki 的樣式屬性形態。條目裡的程式碼範例（BASIC 的
    # `Console.WriteLine("Goodbye {0}", UserName)`）不該被當成殘骸。
    ('模板參數殘骸',        re.compile(r'(?:style|width|height|align|bgcolor|colspan|rowspan|scope|class)\s*=\s*["\']?[^"\'\n]{0,40}["\']?\s*\|', re.I)),
    ('表格空儲存格',        re.compile(r'｜｜')),
    # 資料查詢模板（{{wikidata|qualifier|P1082}}）沒被攔下時，參數關鍵字會黏在
    # 中文裡變成正文：「qualifier時人口數量為property人」。
    # 只認 `qualifier` 貼著中文、或兩個關鍵字同句出現的形態——單看 property
    # 會誤判正常內容（Python 的 @property、PLY 格式的 property uchar red、
    # 美學條目的 properties of qualitative degree 都是條目本身的英文）。
    ('資料模板關鍵字',      re.compile(
        r'(?:(?<=[一-鿿])qualifier|qualifier(?=[一-鿿])|qualifier[^\n]{0,40}?property)')),
    # 模板表若收進重定向頁，`{{DPP}}` 會展開成 `#重定向 Template:DPP` 出現在
    # 正文裡（實測 2.29% 的條目中招）。template_store 現在會解開或丟棄。
    # 必須指向 Template 命名空間才算殘骸——「重定向」本身是條目主題
    # （《重定向 (電腦)》、ICMP 的重定向章節），`### 重定向` 是正常的標題。
    ('重定向殘骸',        re.compile(
        r'(?i)#\s*(?:REDIRECT|重定向|重新導向)\s*\[?\[?\s*:?\s*(?:Template|模板|t)\s*:')),
    ('孤立 File: 前綴',     re.compile(r'(?m)^\s*(?:File|Image|檔案|文件)\s*:', re.I)),
]

# 先做 O(1) 的字面字串門檻，再跑較昂貴的正則。全量 148 萬篇中，模板參數
# 殘骸只有極少數候選；舊版卻讓含 `{0,40}` 的正則掃過每一個字元，單這一項
# 就佔十多分鐘。needle 只負責排除不可能命中的文章，不改變實際判定規則。
_HARD_NEEDLES = {
    '殘留模板 {{ }}': ('{{', '}}'),
    '殘留表格 {| |}': ('{|', '|}'),
    '殘留 wiki 連結': ('[[', ']]'),
    '殘留 ref 標籤': ('<ref', '</ref'),
    '殘留 HTML 標籤': ('<',),
    '殘留 HTML 實體': ('&',),
    '連續空行': ('\n\n\n',),
    '行尾空白': (' \n', '\t\n'),
    '表格空儲存格': ('｜｜',),
    '資料模板關鍵字': ('qualifier',),
}


def _could_match_hard(name, text):
    if name == '模板參數殘骸':
        return '=' in text and '|' in text
    if name == '重定向殘骸':
        return 'template' in text.casefold() or '模板' in text
    if name == '孤立 File: 前綴':
        folded = text.casefold()
        return (':' in text and any(
            word in folded for word in ('file', 'image', '檔案', '文件')))
    needles = _HARD_NEEDLES.get(name)
    return needles is None or any(needle in text for needle in needles)

# 註：U+FFFD 不列入硬性缺陷——來源文本本身就有（《中文亂碼》條目在示範亂碼、
# 《大唐西域求法高僧傳》有缺字），不是我們造成的。

# B. 完整性
END_OK = re.compile(r'[。！？…」』）\"\'.!?》\]%]$')
# 開頭合法的形態：正文字元、章節標題、列表項，或以引號／特殊字元起頭的條目
# （《深喉 (水門事件)》開頭就是「深喉嚨」，西里爾字母與罕用漢字條目也很常見）
OPEN_OK = re.compile(r'^[^\n]+\n\n(?:#{2,6} |- |[^\s#])')


def main():
    if not os.path.isdir(OUT):
        print(f'✗ 找不到資料集目錄：{OUT}', file=sys.stderr)
        return 1
    files = [os.path.join(OUT, f) for f in sorted(os.listdir(OUT)) if f.endswith('.parquet')]
    if not files:
        print(f'✗ {OUT} 沒有 parquet', file=sys.stderr)
        return 1

    n = 0
    hard = collections.Counter()
    hard_samples = collections.defaultdict(list)
    title_ok = url_ok = id_ok = 0
    bad_open = end_bad = end_bad_prose = 0
    lengths = []
    heading_counts = []
    seen_hash = collections.Counter()
    listy = noheading = tex_total = 0
    end_kinds = collections.Counter()

    for path in files:
        print(f"掃描 {os.path.basename(path)}…", file=sys.stderr, flush=True)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=20000):
            for r in batch.to_pylist():
                n += 1
                t = r['text']
                lengths.append(len(t))

                body = t.split('\n', 1)[1].strip() if '\n' in t else ''
                if not body:
                    hard['空正文'] += 1
                    if len(hard_samples['空正文']) < 3:
                        hard_samples['空正文'].append((r['title'], t[:80]))

                has_tex, tex_unbal, tex_delim = math_defects(t)
                if has_tex:
                    tex_total += 1
                    if tex_unbal:
                        hard['公式括號不平衡'] += 1
                        if len(hard_samples['公式括號不平衡']) < 3:
                            hard_samples['公式括號不平衡'].append(
                                (r['title'], f"{{ ×{t.count('{')}  }} ×{t.count('}')}"))
                    if tex_delim:
                        hard['公式未標記分隔符'] += 1
                        if len(hard_samples['公式未標記分隔符']) < 3:
                            hard_samples['公式未標記分隔符'].append(
                                (r['title'], f"$ ×{t.count('$')}"))

                # 殘留標記一律在「挖掉逐字區塊」的版本上檢查
                plain = strip_verbatim(t)
                # 空白類的檢查要用「圍欄整塊壓成一行」的版本：程式碼裡的空行
                # 與縮排是內容（Python 的定義之間、組語的分節），保留換行去查
                # 「連續空行」等於把程式碼的排版當成缺陷。
                flat = flatten_verbatim(t)
                for name, pat in HARD:
                    if name in _WHITESPACE_CHECKS:
                        if not _could_match_hard(name, flat):
                            continue
                        m = pat.search(flat)
                        if m:
                            hard[name] += 1
                            if len(hard_samples[name]) < 3:
                                lo = max(0, m.start() - 45)
                                hard_samples[name].append((r['title'], flat[lo:m.end() + 45]))
                        continue
                    if not _could_match_hard(name, plain):
                        continue
                    m = pat.search(plain)
                    if m:
                        hard[name] += 1
                        if len(hard_samples[name]) < 3:
                            lo = max(0, m.start() - 45)
                            hard_samples[name].append((r['title'], plain[lo:m.end() + 45]))

                if t.startswith(r['title'] + '\n'):
                    title_ok += 1
                if r['url'].startswith('https://zh.wikipedia.org/wiki/'):
                    url_ok += 1
                if r['id'].isdigit():
                    id_ok += 1

                # 「標題 → 空行 → 正文」。以 `## 章節` 或 `- 列表` 開頭是合法的
                # 條目形態（年份條目沒有前言，直接就是「大事記」），不算異常。
                if not OPEN_OK.match(t):
                    bad_open += 1

                last = t.rstrip().split('\n')[-1]
                if not END_OK.search(last):
                    end_bad += 1
                    if '｜' in last:
                        end_kinds['表格列'] += 1
                    elif last.startswith('- '):
                        end_kinds['列表項'] += 1
                    elif re.search(r'[A-Za-z0-9)\]}]$', last):
                        end_kinds['英數/程式碼'] += 1
                    elif len(last) < 12:
                        end_kinds['極短行'] += 1
                    else:
                        end_kinds['★散文但無句號'] += 1
                        end_bad_prose += 1

                h = len(re.findall(r'(?m)^#{2,6} ', t))
                heading_counts.append(h)
                if h == 0:
                    noheading += 1
                lines = t.split('\n')
                if sum(1 for line in lines
                       if line.startswith('- ')) > len(lines) * 0.5:
                    listy += 1
                seen_hash[hash(t)] += 1

    if not n:
        print(f'✗ {OUT} 的 parquet 沒有任何記錄', file=sys.stderr)
        return 1

    if n == 0:
        print(f'✗ {OUT} 的 parquet 沒有任何記錄', file=sys.stderr)
        return 1

    percentiles = statistics.quantiles(lengths, n=100)

    def q(percent):
        return percentiles[percent - 1]
    dup = sum(c - 1 for c in seen_hash.values() if c > 1)

    print(f"=== 全量校驗 {OUT}（lang={LANG}）：{n:,} 筆 ===\n")
    print("【A. 硬性缺陷（必須為 0）】")
    worst = 0
    for name, _ in HARD:
        c = hard.get(name, 0)
        worst = max(worst, c)
        flag = '' if c == 0 else '   ← 需修正'
        print(f"  {name:<22}{c:>8,} ({c/n:7.4%}){flag}")

    print("\n【數學式來源品質警示（不阻擋出貨）】")
    for name in ('公式括號不平衡', '公式未標記分隔符'):
        c = hard[name]
        flag = '' if c == 0 else '   ← 需核對來源'
        print(f"  {name:<22}{c:>8,} ({c/max(tex_total,1):7.4%} of 含公式 {tex_total:,}){flag}")

    print("\n【B. 完整性】")
    print(f"  標題開頭一致          {title_ok:>8,} ({title_ok/n:7.2%})")
    print(f"  URL 合法              {url_ok:>8,} ({url_ok/n:7.2%})")
    print(f"  id 為數字             {id_ok:>8,} ({id_ok/n:7.2%})")
    print(f"  完全重複文檔          {dup:>8,} ({dup/n:7.4%})")
    print(f"  開頭異常              {bad_open:>8,} ({bad_open/n:7.2%})")
    print(f"  結尾無句讀            {end_bad:>8,} ({end_bad/n:7.2%})")
    for k, c in end_kinds.most_common():
        print(f"      {k:<18}{c:>8,} ({c/max(end_bad,1):6.1%})")

    print("\n【C. 分布（供判讀）】")
    print(f"  長度 p5={q(5):.0f} p25={q(25):.0f} p50={q(50):.0f} p75={q(75):.0f} "
          f"p95={q(95):.0f} p99={q(99):.0f} max={max(lengths):,}")
    print(f"  平均章節數 {statistics.mean(heading_counts):.1f}｜無章節 {noheading:,} ({noheading/n:.1%})")
    print(f"  列表為主的條目 {listy:,} ({listy/n:.1%})")
    print(f"  總字元 {sum(lengths):,}")

    if worst:
        print("\n【硬性缺陷樣本】")
        for name, _ in HARD:
            for title, snippet in hard_samples.get(name, []):
                print(f"  [{name}] {title}: {snippet[:110]!r}")

    print(f"\n{'✓ 全部硬性檢查通過' if worst == 0 else '✗ 有硬性缺陷，不應出貨'}")
    # 有缺陷就非零退出。原本一律 exit 0，chain 腳本再怎麼 `||` 也攔不住，
    # 這支「出貨閘門」實際上從來沒擋過任何東西。
    return 1 if worst else 0


if __name__ == '__main__':
    sys.exit(main())
