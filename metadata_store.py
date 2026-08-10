"""單次掃描 dump，同時建立模板表與資訊框標籤表。"""

import bz2

from gensim.corpora.wikicorpus import extract_pages
from tqdm import tqdm

import infobox_labels
import template_store


def build(xml_path, out_dir):
    """共用同一次 XML 解析與 bz2 解壓，輸出內容與分開建表相同。"""
    opener = bz2.open if xml_path.endswith('.bz2') else open
    template_state = template_store._new_collector()
    label_state = infobox_labels._new_collector()

    with opener(xml_path, 'rb') as f:
        for title, text, _pid in tqdm(extract_pages(f), desc='收集模板與資訊框標籤'):
            template_store._collect_page(template_state, title, text)
            infobox_labels._collect_page(label_state, title, text)

    store = template_store._finish_collector(template_state, out_dir)
    labels = infobox_labels._finish_collector(label_state, out_dir)
    return store, labels
