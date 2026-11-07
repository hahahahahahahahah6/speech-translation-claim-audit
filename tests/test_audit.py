import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from speech_translation_claim_audit import phrases
from speech_translation_claim_audit.audit import audit, evidence, match_phrases, prose, main

EXPECTED_POSITIVE = ['speech translation', 'speech-translation', 'speechtranslation', 'speech translation model', 'speech translation models', 'speech translation system', 'speech translation systems', 'speech translation feature', 'speech translation capability', 'speech translation support', 'supports speech translation', 'speech translation tool', 'speech translation api', 'speech translation pipeline', 'speech translation algorithm', 'speech translation library', 'speech translation framework', 'speech translation service', 'speech translation function', 'speech translation engine', 'speech translation engines', 'speech translator', 'speech translators', 'translate speech', 'translates speech', 'translating speech', 'translate spoken', 'translates spoken', 'translating spoken', 'speech to text translation', 'speech-to-text translation', 'spoken language translation', 'spoken-language translation', 'audio translation', 'audio translator', 'translate audio', 'translates audio', 'translating audio', 'voice translation', 'voice translator', 'translate voice', 'speech to speech translation', 'speech-to-speech translation', 'cross-lingual speech translation', 'multilingual speech translation', 'real-time speech translation', 'realtime speech translation', 'live speech translation', 'speech translation task', 'speech translation benchmark']
EXPECTED_NEGATIVE = ['does not support speech translation', 'do not support speech translation', "doesn't support speech translation", "don't support speech translation", 'no speech translation', 'without speech translation', 'never performs speech translation', 'does not perform speech translation', 'speech translation not supported', 'speech translation unsupported', 'no speech translation support', 'lacks speech translation', 'missing speech translation', 'speech translation disabled', 'no speech translation capability', 'without speech translation support', 'does not provide speech translation', 'no speech translation model', 'speech translation not available', 'does not use speech translation', 'no speech translator', 'without speech translator', 'does not support speech translator', 'no spoken language translation']
EXPECTED_GENERIC = ['translate_speech', 'speech_translate', 'speech_translation', 'translate_audio', 'audio_translate', 'translate_spoken', 'spoken_translate', 'translate_voice', 'voice_translate', 'translate_speech_to_text', 'speech_to_text_translation', 'audio_translation', 'speech_translator', 'translate_audio_to_text', 'speech_to_speech_translation', 'translate_speech_audio', 'cross_lingual_speech_translation', 'translate_recorded_speech', 'translate_voice_audio', 'speech_translate_audio', 'translate_speech_input', 'speech_language_translation', 'translate_spoken_audio', 'translate_speech_file']

class AuditTests(unittest.TestCase):
    def test_phrase_oracle(self):
        self.assertEqual(phrases.SPEECHTRANSLATION, EXPECTED_POSITIVE)
        self.assertEqual(phrases.NOSPEECHTRANSLATION, EXPECTED_NEGATIVE)

    def test_generic_oracle(self):
        self.assertEqual(phrases.GENERIC_NAMES, EXPECTED_GENERIC)

    def test_dedicated_oracle(self):
        self.assertEqual(phrases.WHISPER_CONSTRUCTORS, ['load_model', 'WhisperModel'])
        self.assertEqual(phrases.SEAMLESS_MODELS, ['SeamlessM4TModel', 'AutoModelForSpeechSeq2Seq'])
        self.assertEqual(phrases.SPEECH2TEXT_MODELS, ['Speech2TextForConditionalGeneration'])

    def fixture(self, files):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
        return root

    def cli(self, *args):
        output = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = main(list(args))
        return code, output.getvalue(), errors.getvalue()

    def test_all_phrases_match(self):
        for phrase in EXPECTED_POSITIVE:
            with self.subTest(phrase=phrase):
                positive, negative = match_phrases(phrase)
                self.assertEqual(positive, [{'phrase': phrase, 'offset': 0}])
                self.assertEqual(negative, [])
        for phrase in EXPECTED_NEGATIVE:
            with self.subTest(phrase=phrase):
                positive, negative = match_phrases(phrase)
                self.assertEqual(positive, [])
                self.assertEqual(negative, [{'phrase': phrase, 'offset': 0}])

    def test_generic_routes(self):
        for name in EXPECTED_GENERIC:
            for source in ('x.' + name + '()', 'factory().' + name + '()',
                           'from api import ' + name + '\n' + name + '()',
                           'from api import ' + name + ' as t\nt()'):
                with self.subTest(source=source):
                    self.assertTrue(evidence(source))

    def test_generic_any_base(self):
        self.assertTrue(evidence('def factory(): pass\nfactory().translate_speech()'))
        self.assertTrue(evidence('class Object: pass\nObject().translate_speech()'))

    def test_assigned_alias(self):
        self.assertTrue(evidence('m=SeamlessM4TModel()\nn=m\nn.generate()'))
        self.assertTrue(evidence('m: object = WhisperModel()\nm.transcribe(task="translate")'))

    def test_all_routes_shadowed(self):
        for name in EXPECTED_GENERIC:
            with self.subTest(name=name):
                self.assertFalse(evidence('from api import ' + name + '\nclass ' + name + ': pass\nx.' + name + '()'))
        for name in ['WhisperModel', 'SeamlessM4TModel', 'AutoModelForSpeechSeq2Seq', 'Speech2TextForConditionalGeneration']:
            call = '.transcribe(task="translate")' if name == 'WhisperModel' else '.generate()'
            with self.subTest(name=name):
                self.assertFalse(evidence('from api import ' + name + ' as M\ndef ' + name + '(): pass\nM()' + call))

    def test_positive_model_routes(self):
        sources = [
            'import whisper\nm = whisper.load_model("base")\nm.transcribe("a", task="translate")',
            'import whisper as w\nw.load_model("base").transcribe("a", task="translate")',
            'from whisper import load_model\nm=load_model("base")\nm.transcribe(task="translate")',
            'from whisper import load_model as lm\nlm("base").transcribe(task="translate")',
            'from faster_whisper import WhisperModel\nm=WhisperModel("base")\nm.transcribe(task="translate")',
            'import faster_whisper as fw\nfw.WhisperModel("base").transcribe(task="translate")',
            'WhisperModel("base").transcribe(task="translate")',
            'from faster_whisper import WhisperModel as WM\nWM("base").transcribe(task="translate")',
        ]
        for model in ['SeamlessM4TModel', 'AutoModelForSpeechSeq2Seq', 'Speech2TextForConditionalGeneration']:
            sources.extend([
                model + '().generate()',
                'm=' + model + '()\nm.generate()',
                model + '.from_pretrained("x").generate()',
                'm=' + model + '.from_pretrained("x")\nm.generate()',
                'from transformers import ' + model + ' as M\nM.from_pretrained("x").generate()',
                'import transformers as t\nt.' + model + '.from_pretrained("x").generate()',
                'self.model=' + model + '()\nself.model.generate()',
            ])
        for source in sources:
            with self.subTest(source=source):
                self.assertTrue(evidence(source))

    def test_rejections(self):
        sources = [
            'import whisper, transformers, faster_whisper, torch, speech_recognition',
            'import whisper\nm=whisper.load_model("base")\nm.transcribe("a")',
            'import whisper\nwhisper.load_model("base").transcribe(task="transcribe")',
            'pipeline("automatic-speech-recognition")',
            'Recognizer().recognize_google(audio)',
            'pipeline("translation")', 'MarianMTModel.from_pretrained("x")',
            'x.translate_text("hi")', 'pipeline("text-to-speech")',
            'gTTS("hello")', 'import pyttsx3\npyttsx3.init().say("hi")',
            'import langdetect\nlangdetect.detect("hi")', 'detect_language(audio)',
            'import whisper\nwhisper.load_model("x")', 'WhisperModel("x")',
            'SeamlessM4TModel()', 'AutoModelForSpeechSeq2Seq.from_pretrained("x")',
            'Speech2TextForConditionalGeneration()',
            'm.transcribe(task="translate")', 'm.generate()',
            '"speech translation"\n# speech translator\n"""speech translation"""',
            'from api import translate_speech\ntranslate_speech',
            'translate_speech()',
            'import subprocess\nsubprocess.run(["speech-translation", "a.wav"])',
            'import other\nother.load_model("x").transcribe(task="translate")',
            'import whisper\nm=whisper.load_model("x")\nother.transcribe(task="translate")',
            'SeamlessM4TModel()\nother.generate()',
            'import whisper\nwhisper.load_model("x").transcribe(task=TASK)',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertFalse(evidence(source))

    def test_shadowing(self):
        sources = [
            'from api import translate_speech\ndef translate_speech(): pass\ntranslate_speech()',
            'def translate_speech(): pass\nx.translate_speech()',
            'from api import translate_speech as t\ndef t(): pass\nt()',
            'from api import translate_speech as t\ndef translate_speech(): pass\nt()',
            'from transformers import SeamlessM4TModel\nclass SeamlessM4TModel: pass\nSeamlessM4TModel().generate()',
            'from whisper import load_model\ndef load_model(): pass\nload_model().transcribe(task="translate")',
            'import whisper\ndef load_model(): pass\nwhisper.load_model().transcribe(task="translate")',
            'class WhisperModel: pass\nWhisperModel().transcribe(task="translate")',
            'class Speech2TextForConditionalGeneration: pass\nSpeech2TextForConditionalGeneration().generate()',
            'class AutoModelForSpeechSeq2Seq: pass\nAutoModelForSpeechSeq2Seq().generate()',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertFalse(evidence(source))

    def test_split_modules(self):
        root = self.fixture({'README.md': 'speech translation',
                             'a.py': 'import whisper\nm=whisper.load_model("base")',
                             'b.py': 'm.transcribe(task="translate")'})
        self.assertEqual(len(audit(root)), 1)

    def test_span_removal(self):
        self.assertEqual(match_phrases('does not support speech translation')[0], [])

    def test_longest_first(self):
        for phrase in ['speech translation model', 'speech to text translation',
                       'cross-lingual speech translation']:
            self.assertEqual(match_phrases(phrase)[0], [{'phrase': phrase, 'offset': 0}])
        self.assertEqual(len(match_phrases('speech translation model; speech translation')[0]), 2)

    def test_case_insensitive_high(self):
        findings = audit(self.fixture({'README.md': 'SPEECH TRANSLATION'}))
        self.assertEqual(findings[0]['severity'], 'HIGH')

    def test_documented_collision(self):
        self.assertEqual(match_phrases('audio translation of the user interface strings')[0],
                         [{'phrase': 'audio translation', 'offset': 0}])

    def test_fences(self):
        self.assertEqual(audit(self.fixture({'README.md': '```text\nspeech translation\n```\n'})), [])
        self.assertEqual(prose('before\n````\n```\nspeech translation\n````\nafter'), 'before\n\n\n\n\nafter')

    def test_inline_code_counts(self):
        self.assertEqual(len(audit(self.fixture({'README.md': '`speech translation`'}))), 1)

    def test_exact_substrings(self):
        self.assertEqual(match_phrases('speech_translation')[0], [])
        self.assertEqual(match_phrases('speech--translation')[0], [])
        self.assertEqual(len(match_phrases('xxspeechtranslationyy')[0]), 1)

    def test_cli_good(self):
        root = self.fixture({'README.md': 'speech translation', 'app.py': 'x.translate_speech()'})
        self.assertEqual(self.cli('audit', str(root))[0], 0)

    def test_cli_bad(self):
        root = self.fixture({'README.md': 'speech translation'})
        code, out, _ = self.cli('audit', str(root))
        self.assertEqual(code, 2)
        self.assertIn('HIGH documented-speech-translation-claim-false', out)

    def test_cli_test_only(self):
        root = self.fixture({'tests/fixture.md': 'speech translation'})
        code, out, _ = self.cli('audit', str(root))
        self.assertEqual(code, 0)
        self.assertIn('MEDIUM', out)
        self.assertEqual(self.cli('audit', str(root), '--fail-on', 'medium')[0], 2)

    def test_cli_test_text_fixture(self):
        root = self.fixture({'tests/fixture.txt': 'speech translation'})
        self.assertEqual(self.cli('audit', str(root))[0], 0)
        self.assertEqual(self.cli('audit', str(root), '--fail-on', 'medium')[0], 2)

    def test_cli_test_python_fixture(self):
        root = self.fixture({'tests/test_sample.py': 'fixture="speech translation"'})
        self.assertEqual(self.cli('audit', str(root))[0], 0)
        self.assertEqual(self.cli('audit', str(root), '--fail-on', 'medium')[0], 2)

    def test_cli_denial(self):
        root = self.fixture({'README.md': 'does not support speech translation',
                             'app.py': 'x.translate_speech()'})
        code, out, _ = self.cli('audit', str(root))
        self.assertEqual(code, 2)
        self.assertIn('HIGH documented-no-speech-translation-claim-false', out)

    def test_denial_no_sources(self):
        root = self.fixture({'README.md': 'does not support speech translation'})
        self.assertEqual(audit(root), [])

    def test_cli_no_claim(self):
        root = self.fixture({'README.md': 'Ordinary project', 'app.py': 'x.translate_speech()'})
        self.assertEqual(self.cli('audit', str(root))[0], 0)

    def test_cli_json(self):
        root = self.fixture({'README.md': 'speech translation'})
        code, out, _ = self.cli('audit', str(root), '--json')
        self.assertEqual(code, 2)
        findings = json.loads(out)
        self.assertEqual(findings[0]['verdict'], 'FAIL')
        self.assertTrue(findings[0]['message'])
        self.assertEqual(findings[0]['rule'], 'documented-speech-translation-claim-false')

    def test_version(self):
        with self.assertRaises(SystemExit) as caught:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                main(['--version'])
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(out.getvalue(), '0.1.0\n')

    def test_help(self):
        with self.assertRaises(SystemExit) as caught:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                main(['--help'])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn('audit', out.getvalue())

    def test_ignore(self):
        root = self.fixture({'docs/draft.md': 'speech translation'})
        self.assertEqual(self.cli('audit', str(root), '--ignore', 'docs/')[0], 0)
        self.assertEqual(self.cli('audit', '--ignore', 'docs/', str(root))[0], 0)

    def test_repeatable_ignore(self):
        root = self.fixture({'docs/a.md': 'speech translation', 'docs/b.md': 'speech translation'})
        self.assertEqual(self.cli('audit', '--ignore', 'docs/a', '--ignore', 'docs/b', str(root))[0], 0)

    def test_ignore_evidence(self):
        root = self.fixture({'README.md': 'speech translation', 'app.py': 'x.translate_speech()'})
        self.assertEqual(self.cli('audit', str(root), '--ignore', 'app.py')[0], 2)

    def test_verbose(self):
        root = self.fixture({'README.md': 'Nothing claimed', 'app.py': 'pass'})
        code, _, err = self.cli('audit', str(root), '--verbose')
        self.assertEqual(code, 0)
        self.assertIn('Scope:', err)
        self.assertIn('PROSE README.md', err)
        self.assertIn('PYTHON app.py', err)

    def test_usage_and_io_errors(self):
        self.assertEqual(self.cli('audit', '/this/path/does/not/exist')[0], 1)
        self.assertEqual(self.cli('audit', '--fail-on', 'low')[0], 1)
        self.assertEqual(self.cli('audit', '--unknown')[0], 1)
        self.assertEqual(self.cli()[0], 1)
        root = self.fixture({'README.md': 'ok'})
        (root / 'README.md').write_bytes(b'\xff')
        self.assertEqual(self.cli('audit', str(root))[0], 1)

    def test_scan_scope(self):
        root = self.fixture({'readme.md': 'speech translation', 'other.md': 'speech translation',
                             'docs/nested/guide.md': 'speech translation'})
        self.assertEqual([f['path'] for f in audit(root)], ['docs/nested/guide.md'])

    def test_excluded_directories(self):
        files = {'README.md': 'speech translation'}
        for folder in ['.git', '__pycache__', '.venv', 'venv', 'env', '.env', 'virtualenv']:
            files[folder + '/app.py'] = 'x.translate_speech()'
        files['customenv/pyvenv.cfg'] = 'home = /usr/bin'
        files['customenv/app.py'] = 'x.translate_speech()'
        self.assertEqual(len(audit(self.fixture(files))), 1)

    def test_syntax_error_is_not_evidence(self):
        root = self.fixture({'README.md': 'speech translation', 'broken.py': 'x.translate_speech('})
        self.assertEqual(len(audit(root)), 1)

    def test_self_audit_readme(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(match_phrases(prose((root / 'README.md').read_text())) , ([], []))
        self.assertEqual(self.cli('audit', str(root))[0], 0)

if __name__ == '__main__':
    unittest.main()
