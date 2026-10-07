#!/usr/bin/env python3
"""Synchronize shared C++ sources and fixtures into independently installable packages."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def expected_files():
    files = {}
    for package in ('python', 'julia'):
        sources = [ROOT / 'cpp/CMakeLists.txt', ROOT / 'cpp/THIRD_PARTY_NOTICES.md']
        sources += [p for p in (ROOT / 'cpp/src/native').rglob('*')
                    if p.is_file() and p.suffix in ('.cpp', '.h', '.hpp')]
        for source in sources:
            files[ROOT / package / 'cpp' / source.relative_to(ROOT / 'cpp')] = source
        files[ROOT / package / 'LICENSE'] = ROOT / 'LICENSE'
    for source in (ROOT / 'cpp/src/native').rglob('*'):
        if source.is_file() and source.suffix in ('.cpp', '.h', '.hpp'):
            files[ROOT / 'R/src/native' / source.relative_to(ROOT / 'cpp/src/native')] = source
    files[ROOT / 'R/inst/THIRD_PARTY_NOTICES'] = ROOT / 'cpp/THIRD_PARTY_NOTICES.md'
    for folder in ('R/tests', 'python/tests', 'julia/examples'):
        files[ROOT / folder / 'data.csv'] = ROOT / 'examples/data.csv'
    return files


def sync(check=False):
    expected = expected_files()
    stale = [str(target.relative_to(ROOT)) for target, source in expected.items()
             if not target.is_file() or target.read_bytes() != source.read_bytes()]
    managed = ('R/src/native', 'python/cpp', 'julia/cpp')
    extras = [p for folder in managed for p in (ROOT / folder).rglob('*')
              if p.is_file() and p not in expected and p.name != '.DS_Store']
    if check:
        if stale or extras:
            raise SystemExit(f'Shared package files out of sync: changed={stale}, extra={[str(p) for p in extras]}')
        print('Shared C++ sources, licenses and fixtures match all packages.')
        return
    for target, source in expected.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for p in extras:
        p.unlink()
    print('Shared package files synchronized. Edit language sources directly in R/, julia/ and python/.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    sync(parser.parse_args().check)
