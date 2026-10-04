#!/usr/bin/env python3
"""Generate/check the self-contained Julia registration tree from shared sources."""
import argparse
from pathlib import Path
import shutil
import tempfile
from build_packages import ROOT, assemble, version

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
target = ROOT / "julia" / "SpectralMM"
with tempfile.TemporaryDirectory() as work:
    source = assemble(Path(work), "Julia", version())
    expected = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    if args.check:
        missing = [str(p) for p, data in expected.items()
                   if not (target / p).is_file() or (target / p).read_bytes() != data]
        extra = [str(p.relative_to(target)) for p in target.rglob("*")
                 if p.is_file() and p.relative_to(target) not in expected
                 and p.name not in ("Manifest.toml", "build.log")
                 and p.suffix != ".cov" and "build" not in p.relative_to(target).parts]
        if missing or extra:
            raise SystemExit(f"Julia snapshot is out of sync: changed={missing}, extra={extra}")
        print("Julia registration tree matches canonical sources.")
    else:
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
