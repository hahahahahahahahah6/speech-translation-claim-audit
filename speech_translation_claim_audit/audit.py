"""Deterministic documentation and Python call-pattern analysis."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys

from . import __version__
from .phrases import (SPEECHTRANSLATION, NOSPEECHTRANSLATION, GENERIC_NAMES,
                      WHISPER_CONSTRUCTORS, SEAMLESS_MODELS, SPEECH2TEXT_MODELS)


def prose(text):
    """Remove backtick fences, preserving newlines; retain inline content."""
    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r'^\s*(`{3,})', line)
        if fence is None and marker:
            fence = len(marker.group(1))
            lines.append('\n')
        elif fence is not None:
            if re.match(r'^\s*`{' + str(fence) + r',}\s*$', line):
                fence = None
            lines.append('\n')
        else:
            lines.append(line.replace('`', ''))
    return ''.join(lines)


def match_phrases(text):
    """Return (positive matches, denial matches), denying before claiming."""
    remaining = text.lower()

    def consume(phrases):
        nonlocal remaining
        found = []
        for phrase in sorted(phrases, key=len, reverse=True):
            start = remaining.find(phrase)
            while start != -1:
                found.append({'phrase': phrase, 'offset': start})
                remaining = (remaining[:start] + ' ' * len(phrase)
                             + remaining[start + len(phrase):])
                start = remaining.find(phrase)
        return sorted(found, key=lambda item: item['offset'])

    negative = consume(NOSPEECHTRANSLATION)
    return consume(SPEECHTRANSLATION), negative


def evidence(source):
    """Whether a single module contains one of the specified call routes.

    This is structural analysis, not execution or a type/flow proof. Local
    function/class definitions shadow their names throughout the module.
    """
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    shadowed = {n.name for n in nodes if isinstance(
        n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    aliases = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split('.')[0]] = (
                    alias.name if alias.asname else alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = node.module + '.' + alias.name

    def resolve(node):
        if isinstance(node, ast.Name):
            if node.id in shadowed:
                return None
            value = aliases.get(node.id, node.id)
            if value.split('.')[-1] in shadowed:
                return None
            return value
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            if base is None or node.attr in shadowed:
                return None
            return base + '.' + node.attr
        return None

    instances = {}

    def instance_key(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = instance_key(node.value)
            return base + '.' + node.attr if base else None
        return None

    def constructed(node):
        if isinstance(node, (ast.Name, ast.Attribute)):
            if any(isinstance(n, ast.Name) and n.id in shadowed for n in ast.walk(node)):
                return None
            return instances.get(instance_key(node))
        if not isinstance(node, ast.Call):
            return None
        name = resolve(node.func)
        if not name:
            return None
        if name.endswith('.from_pretrained'):
            name = name.rsplit('.', 1)[0]
        terminal = name.split('.')[-1]
        if terminal == 'load_model' and name == 'whisper.load_model':
            return 'whisper'
        if terminal == WHISPER_CONSTRUCTORS[1]:
            return 'whisper'
        if terminal in SEAMLESS_MODELS + SPEECH2TEXT_MODELS:
            return 'generate'
        return None

    # Resolve assigned models and simple aliases without relying on AST walk order.
    assignments = [n for n in nodes if isinstance(n, (ast.Assign, ast.AnnAssign))]
    for _ in range(len(assignments) + 1):
        changed = False
        for node in assignments:
            kind = constructed(node.value)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if kind and isinstance(target, (ast.Name, ast.Attribute)) and instances.get(instance_key(target)) != kind:
                    instances[instance_key(target)] = kind
                    changed = True
        if not changed:
            break

    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        name = resolve(node.func)
        if isinstance(node.func, ast.Attribute) and node.func.attr in GENERIC_NAMES:
            if node.func.attr not in shadowed:
                return True
        if (name and name.split('.')[-1] in GENERIC_NAMES
                and isinstance(node.func, ast.Name) and node.func.id in aliases):
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr not in shadowed:
            kind = constructed(node.func.value)
            if kind == 'generate' and node.func.attr == 'generate':
                return True
            if kind == 'whisper' and node.func.attr == 'transcribe':
                if any(k.arg == 'task' and isinstance(k.value, ast.Constant)
                       and k.value.value == 'translate' for k in node.keywords):
                    return True
    return False


EXCLUDED = {'.git', '__pycache__', '.venv', 'venv', 'env', '.env', 'virtualenv'}


def audit(repo='.', ignore=(), decisions=None):
    root = Path(repo)
    if not root.is_dir():
        raise OSError('Repository is not a directory: ' + str(root))
    decisions = decisions if decisions is not None else []
    docs = []
    tests = []
    has_evidence = False

    def skipped(relative):
        return any(relative.startswith(prefix) for prefix in ignore)

    def walk_error(error):
        raise error

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        current = Path(directory)
        kept = []
        for name in sorted(dirs):
            path = current / name
            rel = path.relative_to(root).as_posix()
            if (name in EXCLUDED or (path / 'pyvenv.cfg').is_file()
                    or path.is_symlink() or skipped(rel)):
                decisions.append('SKIP ' + rel + '/')
            else:
                kept.append(name)
        dirs[:] = kept
        for name in sorted(files):
            path = current / name
            rel = path.relative_to(root).as_posix()
            parts = path.relative_to(root).parts
            if skipped(rel) or path.is_symlink():
                decisions.append('SKIP ' + rel)
                continue
            is_doc = rel == 'README.md' or (parts[0] == 'docs' and path.suffix == '.md')
            is_test = 'tests' in parts[:-1]
            is_python = path.suffix == '.py'
            if not (is_doc or is_test or is_python):
                decisions.append('OUT OF SCOPE ' + rel)
                continue
            text = path.read_text(encoding='utf-8')
            if is_python:
                try:
                    present = evidence(text)
                except SyntaxError:
                    decisions.append('PYTHON SYNTAX SKIP ' + rel)
                else:
                    has_evidence = has_evidence or present
                    decisions.append('PYTHON ' + rel + (' evidence' if present else ' no evidence'))
            if is_doc or is_test:
                positive, negative = match_phrases(prose(text))
                record = (rel, positive, negative)
                (tests if is_test else docs).append(record)
                decisions.append('PROSE ' + rel + ': ' + str(len(positive))
                                 + ' claims, ' + str(len(negative)) + ' denials')
    findings = []
    # Test-only means the positive feature is absent from shipped documentation.
    shipped_claim = any(positive for _, positive, _ in docs)
    for rel, positive, negative in docs + tests:
        is_test = any(rel == item[0] for item in tests)
        if positive and not has_evidence and (not is_test or not shipped_claim):
            for match in positive:
                severity = 'MEDIUM' if is_test else 'HIGH'
                findings.append(dict(rule='documented-speech-translation-claim-false',
                                     severity=severity, path=rel, phrase=match['phrase'],
                                     verdict='FAIL', message='Documented claim has no Python evidence.'))
        if negative and has_evidence:
            for match in negative:
                findings.append(dict(rule='documented-no-speech-translation-claim-false',
                                     severity='HIGH', path=rel, phrase=match['phrase'],
                                     verdict='FAIL', message='Documented denial contradicts Python evidence.'))
    return findings


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    parser = Parser(description='Audit documentation claims against local Python call patterns.')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True, parser_class=Parser)
    command = commands.add_parser('audit')
    command.add_argument('repo', nargs='?', default='.')
    command.add_argument('--json', action='store_true')
    command.add_argument('--verbose', action='store_true')
    command.add_argument('--ignore', action='append', default=[], metavar='PREFIX')
    command.add_argument('--fail-on', choices=['high', 'medium'], default='high')
    try:
        args = parser.parse_args(argv)
        decisions = []
        findings = audit(args.repo, args.ignore, decisions)
        if args.verbose:
            print('Scope: README.md, docs/**/*.md, tests fixtures; all Python modules.', file=sys.stderr)
            for decision in decisions:
                print(decision, file=sys.stderr)
        if args.json:
            print(json.dumps(findings, indent=2))
        elif findings:
            for finding in findings:
                print('{severity} {rule} {path}: {message}'.format(**finding))
        else:
            print('PASS: no contradictory claims found.')
        return 2 if any(f['severity'] == 'HIGH' or args.fail_on == 'medium'
                        for f in findings) else 0
    except (OSError, UnicodeError, ValueError) as error:
        print('error: ' + str(error), file=sys.stderr)
        return 1
