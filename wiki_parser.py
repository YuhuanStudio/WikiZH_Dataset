"""
維基百科中文數據解析模組
功能：解析維基百科 XML 數據並生成純文本內容
"""

import re
import bz2file
from gensim.corpora.wikicorpus import extract_pages, filter_wiki


class WIKIParse(object):

    KEYWORDS = [
        'Template', 'Category', 'Wikipedia',
        'File', 'Topic', 'Portal',
        'MediaWiki', '模块', 'Draft', 'Help'
    ]
    
    def __init__(self, input_file, markdown=False):
        try:
            bz2_file = bz2file.open(input_file)
            self.wiki_content = extract_pages(bz2_file)
        except Exception as e:
            raise RuntimeError(e)

        self.markdown = markdown
        self.nl = '\n\n' if markdown else '\n'

    def __is_not_word(self, word):
        word_items = word.split(':')
        if len(word_items) > 1 and \
                word_items[0] in self.KEYWORDS:
            return True
        return False

    def __is_redirect(self, text):
        return re.findall(r'^#', text)

    def __clean_synonym(self, s): # 处理同义词
        t1 = r'-{(.*?)}-'
        t2 = r'.*zh-(?:hans|cn):(.*?)(;|}-|;zh).*'

        while True:
            match1 = re.search(t1, s, re.DOTALL)
            if match1 is None:
                break

            start, end = match1.span()
            sub_s = s[start:end].replace(' ', '')
            match2 = re.match(t2, sub_s, re.DOTALL)

            if match2 is not None:
                sub_s = match2.group(1)
            else:
                sub_s = sub_s[2:-2]
                if 'zh-hans' in sub_s or 'zh-cn' in sub_s:
                    sub_s = ''

            s = s[:start] + sub_s + s[end:]
        return s

    def __clean_template(self, s):
        # 处理 {{Le|文本1|文本2|文本3}} 类型模板，优先保留第三个参数(通常是中文解释)，否则保留第一个参数
        def le_replacement(match):
            text1 = match.group(1)
            text3 = match.group(3) if match.group(3) else None
            return text3 if text3 else text1

        # 先处理Le模板
        le_template = r'{{Le\|(.*?)\|(.*?)(?:\|(.*?))?}}'
        s = re.sub(le_template, le_replacement, s)

        # 處理所有 link-xx 類模板，保留第二個參數（顯示文字）
        def link_lang_replacement(match):
            # {{link-de|顯示文字|原文}}，保留顯示文字
            return match.group(1)
        s = re.sub(r'{{link-[a-z]{2}\|([^|{}]+)\|([^|{}]+)}}', link_lang_replacement, s)

        # 处理其他简单模板，直接删除（不刪 link-xx）
        t = r'{{(?!lang|link-[a-z]{2})(.*?)}}'
        while True:
            match = re.search(t, s)
            if match is None:
                break
            start, end = match.span()
            s = s[:start] + s[end:]
        return s

    def __clean_wiki_links(self, s):
        """自定义的链接清理函数，保留链接文本"""
        # 处理 [[链接|显示文本]] 格式，保留显示文本
        s = re.sub(r'\[\[([^\|\]]+)\|([^\]]+)\]\]', r'\2', s) #繁體
        
        # 处理 [[链接]] 格式，保留链接文本
        s = re.sub(r'\[\[([^\]]+)\]\]', r'\1', s)
        
        # 处理外部链接 [http://... 显示文本] 格式
        s = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', s)
        
        # 移除单独的外部链接 [http://...]
        s = re.sub(r'\[https?://[^\s\]]+\]', '', s)
        
        # 移除其他wiki标记但不使用filter_wiki
        s = re.sub(r"'''([^']+)'''", r'\1', s)  # 粗体
        s = re.sub(r"''([^']+)''", r'\1', s)    # 斜体
        s = re.sub(r'<ref[^>]*>.*?</ref>', '', s, flags=re.DOTALL)  # 移除引用
        s = re.sub(r'<ref[^>]*/?>', '', s)  # 移除单独的ref标签
        s = re.sub(r'<!--.*?-->', '', s, flags=re.DOTALL)  # 移除注释
        s = re.sub(r'<[^>]+>', '', s)  # 移除其他HTML标签
        
        return s

    def __clean(self, s):
        s = self.__clean_synonym(s)
        s = self.__clean_template(s)

        s = re.sub(r':*{\|[\s\S]*?\|}', '', s)
        s = re.sub(r'\[\[File:.*\]\]', '', s)
        s = re.sub(r'<gallery[\s\S]*?</gallery>', '', s)
        s = re.sub(r'(.){{([^{}\n]*?\|[^{}\n]*?)}}',
                   r'\\1[[\\2]]', s)
        
        # 使用 filter_wiki
        s = filter_wiki(s)
        
        s = re.sub(r'\* *\n|\'{2,}', '', s)
        s = re.sub('\n+', '\n', s)
        s = re.sub('\n[:;]|\n +', '\n', s)
        s = re.sub('\n==', '\n\n==', s)
        s = s.replace('\n。', '。\n')
        return s

    def __clean_surrogates(self, text):
        """清理文本中的代理對字符"""
        if isinstance(text, str):
            return text.encode('utf-8', 'ignore').decode('utf-8')
        return text

    def __fresh(self, word, text):
        def update(cn):
            return str(int(cn) + 1)

        def get_title(line, symbol):
            temp = '{}(.+?){}'.format(symbol, symbol)
            match = re.search(temp, line)
            if match is None:
                return ''

            title = match.group(1)
            title = title.strip()
            return title

        def form_line(catalog, title, level):
            if catalog:
                level = len(catalog) - catalog.count('0') + 1
                catalog = [c for c in catalog if c != '0']
                line = '.'.join(catalog) + ' ' + title
            else:
                line = title + self.nl

            if self.markdown:
                line = '#' * level + ' ' + line
            return line

        fresh_text = form_line(None, word, 1)
        prev_item_line = False
        c2, c3, c4, c5 = '0', '0', '0', '0'
        for line in text.split('\n'):
            item_line = False
            if line.startswith('====='):
                c5 = update(c5)
                title = get_title(line, '=====')
                line = form_line([c2, c3, c4, c5], title, 5)
            elif line.startswith('===='):
                c4, c5 = update(c4), '0'
                title = get_title(line, '====')
                line = form_line([c2, c3, c4], title, 4)
            elif line.startswith('==='):
                c3, c4, c5 = update(c3), '0', '0'
                title = get_title(line, '===')
                line = form_line([c2, c3], title, 3)
            elif line.startswith('=='):
                c2, c3, c4, c5 = update(c2), '0', '0', '0'
                title = get_title(line, '==')
                line = form_line([c2], title, 2)
            elif line.startswith('***'):
                line = '  * ' + line[3:].strip()
                item_line, prev_item_line = True, True
            elif line.startswith('**'):
                line = ' * ' + line[2:].strip()
                item_line, prev_item_line = True, True
            elif line.startswith('*') or line.startswith('#'):
                line = '* ' + line[1:].strip()
                item_line, prev_item_line = True, True
            else:
                pass

            if not item_line and prev_item_line:
                fresh_text += '\n'
                prev_item_line = False

            nl = '\n' if item_line else self.nl
            fresh_text += line + nl
        return fresh_text

    def parse(self, content):
        word, text, ID = content

        if self.__is_not_word(word) or \
           self.__is_redirect(text):
            return None, None

        #if 內文長度小於 30 則不處理
        if len(text) < 30:
            return None, None
        text = self.__clean(text)
        
        # 在處理前先清理代理字符
        word = self.__clean_surrogates(word)
        text = self.__clean_surrogates(text)
        
        text = self.__fresh(word, text)
        
        return ID, text
