# Documentation claim auditor

An offline, deterministic Python 3.9+ CLI for checking documentation against
Python call patterns for turning spoken audio in one language into text or
audio in another. Runtime uses only the standard library: no LLM, network,
telemetry, code execution, or file writes. Output goes to stdout, with verbose
decisions and errors on stderr.

## Install and run

```sh
python3 -m pip install .
speech-translation-claim-audit --version
speech-translation-claim-audit audit /path/to/repo
python3 -m speech_translation_claim_audit audit /path/to/repo --json
speech-translation-claim-audit audit --ignore docs/drafts/ /path/to/repo
speech-translation-claim-audit audit . --ignore vendor/ --ignore examples/ --fail-on medium --verbose
python3 -m unittest discover
```

The repository defaults to the current directory. Each ignore option takes one
literal relative path prefix, using forward slashes, and may be repeated.
Ignored paths contribute neither claims nor evidence. The default threshold is
high. Exit codes are 0 for no findings at the threshold, 2 for findings at the
threshold, and 1 for usage or IO errors. Version output is 0.1.0.

JSON output is a list of finding objects, each with rule, severity, path,
phrase, verdict, and message fields. A finding has verdict FAIL; an empty list
means no findings. A medium finding can coexist with a successful exit code.

```text
documented-speech-translation-claim-false  HIGH (MEDIUM for test-only claims)
documented-no-speech-translation-claim-false  HIGH
```

## Scan rules

Shipped prose is the root README.md plus Markdown files recursively under docs.
Backtick fenced code blocks are excluded; inline backticks are stripped but
their content counts. Matching uses exact, case-insensitive substrings without
stemming, paraphrase detection, or hyphen normalization. Denial spans are removed
first. Longest-first matching makes each occurrence match once under its
longest phrase. A positive claim without evidence is a finding; a denial with
evidence is a finding. No claims is a pass, and a denial without any sources
is a pass.

Fixture files anywhere under a tests directory are also scanned as UTF-8 text.
A positive claim appearing only in those files has medium severity. Findings
from shipped documentation have high severity. Every Python module is checked
independently, including tests; paired routes never join evidence across files.
Git metadata, bytecode caches, conventional virtualenv folders, directories
containing pyvenv.cfg, and symbolic links are skipped. Python syntax errors are
skipped as evidence and reported in verbose mode; unreadable or non-UTF-8 scoped
files cause an IO error.

The vocabulary and evidence names are exposed in the package's phrases module.
Imports and aliases are resolved structurally; locally defined functions and
classes shadow matching route names throughout a module. Dedicated model routes
require a constructed instance plus a matching instance call, assigned or chained.

```text
whisper.load_model(...) + instance.transcribe(task="translate")
WhisperModel(...) + instance.transcribe(task="translate")
SeamlessM4TModel(...) + instance.generate(...)
AutoModelForSpeechSeq2Seq.from_pretrained(...) + instance.generate(...)
Speech2TextForConditionalGeneration.from_pretrained(...) + instance.generate(...)
```

## Differentiation

The neighboring tools check distinct capabilities:

```text
speech-recognition-claim-audit
```

Checks transcribing speech in the same language. This auditor checks turning
spoken audio in one language into text or audio in another.

```text
translation-claim-audit
```

Checks translating written words. This auditor checks converting spoken input across languages.

```text
text-to-speech-claim-audit
```

Checks producing speech from text. This auditor checks converting spoken input across languages.

```text
speech-emotion-recognition-claim-audit
```

Checks detecting emotion in speech. This auditor checks carrying meaning across
languages from spoken input, rather than detecting emotion.

## Limitations

This is static pattern matching, not semantic judgement. Exact phrases miss
restatements. Detected calls do not prove working models, correct language
selection, actual audio input, execution, or quality. Recognition alone, written
text conversion alone, synthesis alone, and language detection alone are not
evidence. A model construction alone and an unrelated instance call are not a
pair. Dynamic imports, runtime monkey patching, and intermodule data flow are
outside the analysis. Simple assigned aliases are recognized, but this is not
a control-flow or full type analysis.

A Whisper transcribe call with no task keyword is deliberately not counted,
because the default task is recognition. Only the exact constant keyword below
qualifies for that route.

```python
model.transcribe(audio, task="translate")
```

A generic call on an unrelated object counts as evidence when its method name
is shared with a recognized generic route. That is a deliberate naming heuristic.

```python
unrelated.translate_speech(data)
```

These exact substrings can also describe converting audio-related interface
text, voice-assistant responses, or audio file metadata. They still match; the
auditor cannot infer the intended meaning. Longest-first span blanking prevents
overlapping counts, but does not remove this ambiguity.

```text
audio translation of the user interface strings
voice translation of the assistant reply
translate audio metadata into French
```

## Distribution checks

The source distribution includes the unittest suite and its discovery marker.
Build requirements use setuptools 77 or newer; the runtime has no dependencies.

```sh
python3 -m build --outdir ../speech-translation-claim-audit-dists
python3 -m unittest discover
```
