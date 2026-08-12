import os
import tempfile
import unittest
from unittest import mock

import image_extractor
import md_to_dataset
import wikidata_store
from hf_uploader import (HFUploader, dump_date_to_version, find_image_files,
                         run_upload, shift_month)
from image_downloader import (
    _collision_safe_filename,
    _is_allowed_source_url,
    _safe_filename,
)
from image_extractor import _file_url, parse_usage
from md_converter import ParseFailuresError, WIKIParse2Doc
from page_store import DONE_MARKER, FAILURE_REPORT, is_complete
from wiki_cli import WikiCLI
from wiki_downloader import _wait_for_visible_file
from md_to_dataset import _drop_orphan_fences, process_page
from qa.validate_full import math_defects, strip_verbatim
from wiki_parser import (
    _eval_expr,
    _fence_code_blocks,
    _keep_math_all,
    _wrap_indented_pre,
    IMAGE_MARK,
    strip_image_marks,
    WIKIParse,
)
from wiki_text import (
    drop_empty_brackets,
    separate_adjacent_math,
    strip_leftover_markup,
    strip_restored_markup,
)


class OperationalSafetyTests(unittest.TestCase):
    def test_image_file_discovery_is_language_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            names = ('wiki_images_dataset.jsonl',
                     'wiki_images_dataset_2.jsonl',
                     'wiki_images_dataset_CN.jsonl',
                     'wiki_images_dataset_CN_2.jsonl')
            for name in names:
                with open(os.path.join(directory, name), 'w', encoding='utf-8'):
                    pass
            tw = {os.path.basename(path)
                  for path in find_image_files(directory, lang='tw')}
            cn = {os.path.basename(path)
                  for path in find_image_files(directory, lang='cn')}
            self.assertEqual(tw, set(names[:2]))
            self.assertEqual(cn, set(names[2:]))

    def test_upload_rejects_empty_scope_before_network_access(self):
        unused_path = os.path.join(tempfile.gettempdir(), 'wikizh-not-used')
        with self.assertRaisesRegex(ValueError, '至少要指定一個語言'):
            run_upload(unused_path, '20260801', langs=())
        with self.assertRaisesRegex(ValueError, '至少要選擇一種'):
            run_upload(unused_path, '20260801',
                       upload_pretrain=False, upload_images=False,
                       upload_omni=False)

    def test_upload_refuses_to_overwrite_newer_root_version(self):
        uploader = HFUploader.__new__(HFUploader)
        uploader.ensure_repo = mock.Mock(return_value=False)
        uploader.detect_archive_version = mock.Mock(return_value='2609')
        uploader.archive_root = mock.Mock()
        uploader.upload = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, '拒絕以 2608 覆寫'):
            unused_file = os.path.join(
                tempfile.gettempdir(), 'wikizh-not-read.parquet')
            uploader.upload_pretrain(
                'tw', [unused_file], '2608', '20260801')
        uploader.archive_root.assert_not_called()
        uploader.upload.assert_not_called()

    def test_page_processing_exception_is_not_counted_as_skip(self):
        with mock.patch('md_to_dataset._build_one',
                        side_effect=ValueError('broken')):
            with self.assertRaisesRegex(RuntimeError, '處理條目 7 時出錯'):
                process_page({'id': '7', 'text': '# 測試\n\n正文'}, lang='tw')

    def test_dataset_promotion_replaces_only_managed_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = tempfile.mkdtemp(
                prefix=md_to_dataset._STAGING_PREFIX, dir=directory)
            old = os.path.join(directory, 'train-00000-of-00002.parquet')
            stale = os.path.join(directory, 'train-00001-of-00002.parquet')
            keep = os.path.join(directory, 'wiki_images_dataset.jsonl')
            new = os.path.join(staging, 'train-00000-of-00001.parquet')
            for path, content in ((old, 'old'), (stale, 'stale'),
                                  (keep, 'image'), (new, 'new')):
                with open(path, 'w', encoding='utf-8') as fh:
                    fh.write(content)
            state = {
                'staging_dir': staging,
                'final_output_dir': directory,
                'final_files': [new],
            }

            md_to_dataset._promote_output_states({'tw': state})

            published = os.path.join(
                directory, 'train-00000-of-00001.parquet')
            with open(published, encoding='utf-8') as fh:
                self.assertEqual(fh.read(), 'new')
            self.assertFalse(os.path.exists(old))
            self.assertFalse(os.path.exists(stale))
            self.assertTrue(os.path.exists(keep))
            self.assertFalse(os.path.exists(staging))

    def test_dataset_promotion_failure_restores_previous_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = tempfile.mkdtemp(
                prefix=md_to_dataset._STAGING_PREFIX, dir=directory)
            old = os.path.join(directory, 'train-00000-of-00001.parquet')
            new = os.path.join(staging, 'train-00000-of-00001.parquet')
            with open(old, 'w', encoding='utf-8') as fh:
                fh.write('old')
            with open(new, 'w', encoding='utf-8') as fh:
                fh.write('new')
            state = {
                'staging_dir': staging,
                'final_output_dir': directory,
                'final_files': [new],
            }
            real_replace = os.replace

            def fail_publish(source, target):
                if source == new:
                    raise OSError('publish failed')
                return real_replace(source, target)

            with mock.patch('md_to_dataset.os.replace', side_effect=fail_publish):
                with self.assertRaisesRegex(OSError, 'publish failed'):
                    md_to_dataset._promote_output_states({'tw': state})

            with open(old, encoding='utf-8') as fh:
                self.assertEqual(fh.read(), 'old')
            self.assertFalse(os.path.exists(staging))

    def test_empty_image_variant_build_preserves_previous_outputs(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch('image_extractor._install_word_guard'), \
                mock.patch('image_extractor.extract_pages', return_value=[]):
            paths = {
                lang: os.path.join(directory, lang, f'{lang}.jsonl')
                for lang in ('tw', 'cn')
            }
            for path in paths.values():
                os.makedirs(os.path.dirname(path))
                with open(path, 'w', encoding='utf-8') as fh:
                    fh.write('previous')
            xml_path = os.path.join(directory, 'dump.xml')
            with open(xml_path, 'wb'):
                pass

            with self.assertRaisesRegex(RuntimeError, '圖片資料集為空'):
                image_extractor.extract_wiki_images_variants(xml_path, paths)

            for path in paths.values():
                with open(path, encoding='utf-8') as fh:
                    self.assertEqual(fh.read(), 'previous')
                leftovers = [
                    name for name in os.listdir(os.path.dirname(path))
                    if name.startswith(image_extractor._IMAGE_STAGE_PREFIX)
                ]
                self.assertEqual(leftovers, [])

    def test_image_promotion_failure_restores_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = tempfile.mkdtemp(
                prefix=image_extractor._IMAGE_STAGE_PREFIX, dir=directory)
            old = os.path.join(directory, 'images.jsonl')
            new = os.path.join(staging, 'images_1.jsonl')
            with open(old, 'w', encoding='utf-8') as fh:
                fh.write('old')
            with open(new, 'w', encoding='utf-8') as fh:
                fh.write('new')
            state = {
                'staging_dir': staging,
                'final_output_dir': directory,
                'final_base': os.path.join(directory, 'images'),
                'ext': '.jsonl',
                'staged_files': [new],
            }
            real_replace = os.replace

            def fail_publish(source, target):
                if source == new:
                    raise OSError('publish failed')
                return real_replace(source, target)

            with mock.patch('image_extractor.os.replace',
                            side_effect=fail_publish):
                with self.assertRaisesRegex(OSError, 'publish failed'):
                    image_extractor._promote_image_states({'tw': state})

            with open(old, encoding='utf-8') as fh:
                self.assertEqual(fh.read(), 'old')
            self.assertFalse(os.path.exists(staging))

    def test_single_language_image_upload_only_archives_and_deletes_its_files(self):
        cases = (
            ('tw', 'wiki_images_dataset_2.jsonl',
             {'wiki_images_dataset_CN.jsonl',
              'wiki_images_dataset_CN_2.jsonl'}),
            ('cn', 'wiki_images_dataset_CN_2.jsonl',
             {'wiki_images_dataset.jsonl',
              'wiki_images_dataset_2.jsonl'}),
        )
        existing = {
            'wiki_images_dataset.jsonl': 100,
            'wiki_images_dataset_2.jsonl': 50,
            'wiki_images_dataset_CN.jsonl': 90,
            'wiki_images_dataset_CN_2.jsonl': 40,
        }

        for lang, expected_delete, preserved in cases:
            with self.subTest(lang=lang), tempfile.TemporaryDirectory() as directory:
                local = os.path.join(directory, f'new-{lang}.jsonl')
                with open(local, 'w', encoding='utf-8') as f:
                    f.write('{}\n')

                uploader = HFUploader.__new__(HFUploader)
                uploader.api = mock.Mock()
                uploader.dry_run = False
                uploader.refresh_cards = False
                uploader._tree_cache = {}
                uploader.ensure_repo = mock.Mock(return_value=False)
                uploader._read_repo_file = mock.Mock(return_value=None)
                uploader._month_dirs = mock.Mock(return_value=[])
                uploader._root_data_files = mock.Mock(side_effect=lambda _repo, pattern: {
                    name: size for name, size in existing.items()
                    if pattern.match(name)
                })

                with mock.patch('hf_uploader.dataset_card', return_value='card'):
                    uploader.upload_images(
                        {lang: [local]}, '2608', '20260801',
                        archive=True, archive_as='2607')

                commits = uploader.api.create_commit.call_args_list
                archived = {
                    op.src_path_in_repo for op in commits[0].kwargs['operations']
                    if op.__class__.__name__ == 'CommitOperationCopy'
                }
                operations = commits[1].kwargs['operations']
                deleted = {
                    op.path_in_repo for op in operations
                    if op.__class__.__name__ == 'CommitOperationDelete'
                }
                expected_base = expected_delete.replace('_2.jsonl', '.jsonl')
                self.assertEqual(archived, {expected_base, expected_delete})
                self.assertTrue(archived.isdisjoint(preserved))
                self.assertEqual(deleted, {expected_delete})
                self.assertTrue(deleted.isdisjoint(preserved))

    def test_empty_bracket_symbols_inside_table_are_preserved(self):
        line = '符號｜（ ）｜“ ”｜『 』｜說明'
        self.assertEqual(drop_empty_brackets(line), line)
        self.assertEqual(drop_empty_brackets('內容｜“”'), '內容｜')

    def test_english_line_with_wiki_links_is_not_deleted(self):
        self.assertEqual(
            strip_leftover_markup('See [[Alpha]] and [[Beta|B]]'),
            'See Alpha and B',
        )

    def test_dump_date_uses_real_calendar(self):
        self.assertEqual(dump_date_to_version('20260801'), '2608')
        with self.assertRaises(ValueError):
            dump_date_to_version('20260230')

    def test_month_version_rejects_invalid_month(self):
        self.assertEqual(shift_month('2612', 1), '2701')
        with self.assertRaises(ValueError):
            shift_month('2699', 0)

    def test_template_expression_rejects_python_power_operator(self):
        self.assertEqual(_eval_expr('2 + 3 * 4'), '14')
        self.assertEqual(_eval_expr('9**999999999999999999'), '')

    def test_dump_rename_waits_for_drvfs_visibility(self):
        with mock.patch(
                'wiki_downloader.os.path.getsize',
                side_effect=[FileNotFoundError, 123]), mock.patch(
                    'wiki_downloader.time.sleep') as sleep:
            self.assertEqual(_wait_for_visible_file('/candidate', 123), 123)
            sleep.assert_called_once()

    def test_download_filename_cannot_escape_output_directory(self):
        escaped = _safe_filename('../../outside.jpg')
        self.assertNotIn('/', escaped)
        self.assertNotIn('\\', escaped)
        self.assertNotEqual(escaped, _safe_filename('..%2F..%2Foutside.jpg'))

    def test_case_insensitive_download_name_collision_is_disambiguated(self):
        used, resolved = {}, {}
        first = _collision_safe_filename('Map.PNG', used, resolved)
        second = _collision_safe_filename('map.png', used, resolved)
        self.assertNotEqual(first.casefold(), second.casefold())
        self.assertEqual(
            first, _collision_safe_filename('Map.PNG', used, resolved))

    def test_file_url_encodes_path_and_query_delimiters(self):
        url = _file_url('A/B?C.jpg')
        self.assertTrue(url.endswith('A%2FB%3FC.jpg'))

    def test_image_word_guard_uses_explicit_page_dir(self):
        candidate = '/candidate/parsed-root/202608'
        words = {'候選詞'}
        with mock.patch('title_words.load', return_value=words) as load, \
                mock.patch('tw_vocab.load_guard') as install:
            image_extractor._install_word_guard(candidate)

        load.assert_called_once_with(candidate)
        install.assert_called_once_with(words)

    def test_image_extractors_forward_explicit_page_dir(self):
        candidate = '/candidate/parsed-root/202608'
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch('image_extractor._install_word_guard') as install, \
                mock.patch('image_extractor.extract_pages',
                           return_value=[('測試', '[[File:X.jpg|圖說]]', '1')]), \
                mock.patch('image_extractor.find_image_tags',
                           return_value=[(0, 'File:X.jpg|圖說')]), \
                mock.patch('image_extractor.iter_gallery_bodies',
                           return_value=[]), \
                mock.patch('image_extractor.section_positions',
                           return_value=[]), \
                mock.patch('image_extractor.parse_usage',
                           return_value=('X.jpg', '圖說', '')):
            xml_path = os.path.join(directory, 'dump.xml')
            with open(xml_path, 'wb'):
                pass
            image_extractor.extract_wiki_images(
                xml_path, os.path.join(directory, 'tw.jsonl'),
                page_dir=candidate)
            image_extractor.extract_wiki_images_variants(
                xml_path, {
                    'tw': os.path.join(directory, 'variant-tw.jsonl'),
                    'cn': os.path.join(directory, 'variant-cn.jsonl'),
                }, page_dir=candidate)

        self.assertEqual(install.call_args_list,
                         [mock.call(candidate), mock.call(candidate)])

    def test_cli_infers_matching_image_word_guard_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            cli = WikiCLI()
            cli.md_dir = os.path.join(directory, 'parsed-root')
            page_dir = os.path.join(cli.md_dir, '202608')
            os.makedirs(page_dir)
            with open(os.path.join(page_dir, 'pages-00000.jsonl'), 'w',
                      encoding='utf-8') as f:
                f.write('{}\n')
            with open(os.path.join(page_dir, DONE_MARKER), 'w',
                      encoding='utf-8') as f:
                f.write('1\n')

            dump_dir = os.path.join(directory, 'downloads', '20260801')
            os.makedirs(dump_dir)
            xml_path = os.path.join(
                dump_dir, 'zhwiki-20260801-pages-articles.xml.bz2')
            with open(xml_path, 'wb'):
                pass
            output = os.path.join(directory, 'image.jsonl')

            with mock.patch(
                    'image_extractor.extract_wiki_images') as extract:
                self.assertEqual(cli.extract_images(xml_path, output), output)

        extract.assert_called_once_with(
            xml_path, output, max_images=None, lang='tw', page_dir=page_dir)

    def test_image_downloader_rejects_non_wikipedia_urls(self):
        self.assertTrue(_is_allowed_source_url(
            'https://zh.wikipedia.org/wiki/Special:FilePath/A.jpg'))
        self.assertFalse(_is_allowed_source_url('http://127.0.0.1/private'))

    def test_completion_marker_must_be_valid_and_have_a_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(is_complete(directory))
            with open(os.path.join(directory, DONE_MARKER), 'w', encoding='utf-8') as f:
                f.write('10\n')
            self.assertFalse(is_complete(directory))
            with open(os.path.join(directory, 'pages-00000.jsonl'), 'w',
                      encoding='utf-8') as f:
                f.write('{}\n')
            self.assertTrue(is_complete(directory))

            with open(os.path.join(directory, FAILURE_REPORT), 'w',
                      encoding='utf-8') as f:
                f.write('壞條目\tValueError: broken\n')
            self.assertFalse(is_complete(directory))

    def test_page_parse_failure_writes_report_without_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            parser = WIKIParse2Doc.__new__(WIKIParse2Doc)
            parser.output_dir = directory
            parser.markdown = True
            parser.wiki_content = [('好條目', 'text', '1'),
                                   ('壞條目', 'text', '2')]

            pool = mock.Mock()
            pool.imap_unordered.return_value = iter([
                ('1', '# 好條目\n\n正文', [], None),
                (None, None, None, ('壞條目', 'ValueError: broken')),
            ])
            context = mock.Mock()
            context.Pool.return_value = pool
            with mock.patch('md_converter.multiprocessing.get_context',
                            return_value=context):
                with self.assertRaises(ParseFailuresError):
                    parser.run(workers=1)

            report = os.path.join(directory, FAILURE_REPORT)
            self.assertTrue(os.path.isfile(report))
            with open(report, encoding='utf-8') as f:
                self.assertIn('壞條目\tValueError: broken', f.read())
            self.assertFalse(os.path.exists(os.path.join(directory, DONE_MARKER)))
            self.assertFalse(is_complete(directory))

    def test_cli_rejects_parser_result_without_valid_completion_state(self):
        with tempfile.TemporaryDirectory() as directory:
            xml_path = os.path.join(directory, 'dump.xml.bz2')
            with open(xml_path, 'wb') as f:
                f.write(b'placeholder')
            cli = WikiCLI()
            cli.md_dir = os.path.join(directory, 'parsed')

            parser = mock.Mock()
            parser.run.return_value = 1
            with mock.patch('md_converter.WIKIParse2Doc', return_value=parser), \
                    mock.patch.object(cli, '_prune_old_versions') as prune:
                result = cli.convert_to_md(
                    xml_path=xml_path, latest_date='20260801', force=True,
                    fetch_wikidata=False)

            self.assertIsNone(result)
            prune.assert_not_called()

    def test_custom_image_cleanup_does_not_touch_other_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, 'custom.jsonl')
            paths = [target, os.path.join(directory, 'custom_1.jsonl'),
                     os.path.join(directory, 'official.jsonl')]
            for path in paths:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write('x')
            WikiCLI._clean_image_output_path(target)
            self.assertFalse(os.path.exists(paths[0]))
            self.assertFalse(os.path.exists(paths[1]))
            self.assertTrue(os.path.exists(paths[2]))

    def test_multiline_nowiki_becomes_fenced_code(self):
        body = '<nowiki><p>Hello</p>\n  <span>world</span></nowiki>'
        result = _fence_code_blocks(body)
        self.assertIn(
            '```\n<p>Hello</p>\n  <span>world</span>\n```', result)

    def test_indented_pre_is_protected_before_template_cleanup(self):
        result = _wrap_indented_pre(' class Hello {{\n   return {};\n }}')
        self.assertEqual(
            result, '<pre>\nclass Hello {{\n  return {};\n}}\n</pre>')

    def test_indented_code_survives_the_complete_pipeline(self):
        parser = WIKIParse.__new__(WIKIParse)
        parser.markdown = True
        parser.nl = '\n\n'
        _id, document, _images = parser.parse((
            '範例', '前言。\n\n class Hello {\n   return {};\n }\n\n結尾。', '1'))
        record = process_page({'id': '1', 'text': document}, lang='tw')
        self.assertIn(
            '```\nclass Hello {\n  return {};\n}\n```', record['text'])

    def test_indented_template_parameters_are_not_code(self):
        source = '{{Infobox\n | name = value\n}}'
        self.assertEqual(_wrap_indented_pre(source), source)

    def test_orphan_fence_does_not_consume_next_block(self):
        doc = '前文\n\n```md 範例\n\n```html\n<p>ok</p>\n```\n\n後文'
        cleaned = _drop_orphan_fences(doc)
        self.assertNotIn('```md', cleaned)
        self.assertIn('```html\n<p>ok</p>\n```', cleaned)

    def test_glued_generated_fence_is_split_back_to_its_own_line(self):
        cleaned = _drop_orphan_fences(
            '- 小數部分：```\n0.25 * 2 = 0.5\n```')
        self.assertIn('- 小數部分：\n```\n0.25 * 2 = 0.5\n```', cleaned)

    def test_math_ending_in_single_backslash_cannot_escape_delimiter(self):
        converted = _keep_math_all(
            r'<math display="block">x = \frac{a}{b}\</math>')
        self.assertEqual(converted, r'$$x = \frac{a}{b}\ $$')
        self.assertEqual(len(math_defects(converted)), 3)
        self.assertEqual(math_defects(converted), (True, False, False))

    def test_adjacent_inline_math_is_separated_without_changing_display_math(self):
        text = r'前文 $x_{1}$$y_{2}$。\n\n$$z = x + y$$'
        self.assertEqual(
            separate_adjacent_math(text),
            r'前文 $x_{1}$ $y_{2}$。\n\n$$z = x + y$$')

    def test_image_removed_between_math_spans_keeps_separator(self):
        text = '$2[$' + IMAGE_MARK + '26' + IMAGE_MARK + '$] = 2[2[5]]$'
        self.assertEqual(strip_image_marks(text), '$2[$ $] = 2[2[5]]$')

    def test_image_caption_keeps_readable_prefix_before_unclosed_template(self):
        body = (
            'File:clinic.jpg|thumb|由[[湯姆·艾金斯]]在1875年所繪的《'
            '{{le|格罗斯医师的临床课|The Gross Clinic》，目前收藏於博物館')
        self.assertEqual(
            parse_usage(body, '外科醫師', 'tw'),
            ('clinic.jpg', '由湯姆·艾金斯在1875年所繪的《格羅斯醫師的臨床課', ''))

    def test_image_caption_recovers_text_inside_unclosed_wrapper(self):
        body = 'File:cycle.png|thumb|{{center|顯示慢速碳循環的繪圖。'
        self.assertEqual(
            parse_usage(body, '碳循環', 'tw'),
            ('cycle.png', '顯示慢速碳循環的繪圖。', ''))

    def test_math_layout_tags_are_not_kept_as_latex(self):
        converted = _keep_math_all(
            r'<math>x = \begin{cases}<pre>a & b\\c & d</pre>\end{cases}</math>')
        self.assertEqual(
            converted, r'$$x = \begin{cases} a & b\\c & d \end{cases}$$')

    def test_final_markup_cleanup_preserves_verbatim_content(self):
        text = (
            'Title\n\n$\\frac{{a}}{b}$\n\n'
            'Visible [[Target|標籤]] {{broken|value}} <br/>\n\n'
            '```html\n<div>{{literal}}</div>\n```')
        cleaned = strip_restored_markup(text)
        self.assertIn(r'$\frac{{a}}{b}$', cleaned)
        self.assertIn('標籤', cleaned)
        self.assertNotIn('broken', cleaned)
        self.assertNotIn('<br', cleaned)
        self.assertIn('<div>{{literal}}</div>', cleaned)

    def test_final_markup_cleanup_drops_collapsed_template_soup(self):
        soup = (
            'Prefix {{#ifexpr:$>0|[[File:graph.svg]]|'
            + ('{{loop|$|rowspan=2|&#8203;}}' * 20)
            + '}} suffix')
        text = f'Title\n\n正文。\n\n{soup}\n\n後文。'
        cleaned = strip_restored_markup(text)
        self.assertEqual(cleaned, 'Title\n\n正文。\n\n後文。')

    def test_escaped_dollar_stays_inside_math_span(self):
        text = r'記號 $n\$ = n^{(n-1)\$}$ 是一條完整公式。'
        self.assertEqual(math_defects(text), (True, False, False))
        self.assertNotIn(r'\$', strip_verbatim(text))

    def test_wikidata_fetch_resumes_after_interruption(self):
        need = {f'Title{i:03d}': ['P1'] for i in range(51)}

        def entity(value):
            return {
                'claims': {
                    'P1': [{
                        'rank': 'normal',
                        'mainsnak': {
                            'datavalue': {'type': 'string', 'value': value},
                        },
                    }],
                },
            }

        first_batch = {
            'entities': [entity(f'value-{i}') for i in range(50)],
        }
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                    wikidata_store, '_api',
                    side_effect=[first_batch, KeyboardInterrupt]), \
                    mock.patch.object(wikidata_store.time, 'sleep'):
                with self.assertRaises(KeyboardInterrupt):
                    wikidata_store.fetch(
                        need, directory, sleep=0, checkpoint_every=100)

            progress = os.path.join(directory, wikidata_store.PROGRESS_NAME)
            final = os.path.join(directory, wikidata_store.STORE_NAME)
            self.assertTrue(os.path.isfile(progress))
            self.assertFalse(os.path.exists(final))

            with mock.patch.object(
                    wikidata_store, '_api',
                    return_value={'entities': [entity('value-50')]}) as api, \
                    mock.patch.object(wikidata_store.time, 'sleep'):
                result = wikidata_store.fetch(
                    need, directory, sleep=0, checkpoint_every=100)

            self.assertEqual(api.call_count, 1)
            self.assertEqual(len(result), 51)
            self.assertEqual(result['Title000']['P1'], 'value-0')
            self.assertEqual(result['Title050']['P1'], 'value-50')
            self.assertTrue(os.path.isfile(final))
            self.assertFalse(os.path.exists(progress))


if __name__ == '__main__':
    unittest.main()
