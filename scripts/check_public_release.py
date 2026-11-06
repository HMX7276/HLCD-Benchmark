"""Conservative public-tree checks; not a proof that arbitrary text is deidentified.

Scans all non-cache files, not just git-tracked files. No network or data loading.
Run from any directory. Inspect git diff --cached separately before publishing.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORE_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache'}
ROOT_FILES = {'.env.example', '.gitignore', '.gitattributes', 'LICENSE', 'NOTICE',
              'README.md', 'requirements.txt', 'CITATION.cff', 'CHANGELOG.md',
              'CONTRIBUTING.md', 'RELEASE_MANIFEST.json'}
PATTERNS = [
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'\b(?:sk-[A-Za-z0-9_-]{24,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
    re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    re.compile(r'https?://[^\s/]+:[^\s/]+@'),
]


def allowed(rel):
    if len(rel.parts) == 1:
        return rel.name in ROOT_FILES
    if rel.parts[0] in {'src', 'tests', 'scripts'}:
        return rel.suffix == '.py'
    if rel.parts[0] == 'docs':
        return rel.suffix in {'.md', '.txt'}
    return rel.as_posix() == '.github/workflows/tests.yml'


def main():
    failures, count = [], 0
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if any(part in IGNORE_DIRS for part in rel.parts):
            continue
        if path.is_symlink():
            failures.append(f'{rel}: symlinks are not included in this release policy')
            continue
        if not path.is_file():
            continue
        count += 1
        if not allowed(rel):
            failures.append(f'{rel}: unexpected public file; review or move out of repository')
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeError:
            failures.append(f'{rel}: not UTF-8 text')
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if any(pattern.search(line) for pattern in PATTERNS):
                failures.append(f'{rel}:{i}: possible credential; content suppressed')
            if rel.name == '.env.example' and '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                if (key.endswith('_KEY') or key.endswith('_PASSWORD')
                        or re.search(r'_KEY\d+$', key)) and value.strip():
                    failures.append(f'{rel}:{i}: example credential must be empty')
    if failures:
        print('\n'.join(failures))
        return 1
    print(f'Public-tree checks passed ({count} files). No data artifacts or common credential patterns found.')
    print('This pattern scan does not replace review of newly added text and staged files.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
