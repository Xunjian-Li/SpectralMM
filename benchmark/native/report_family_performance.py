"""Compatibility entry point; reports and LaTeX now share one exporter."""
import argparse
from pathlib import Path
import subprocess
import sys
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--source',choices=('initial','repeat'),default='initial')
args=parser.parse_args()
subprocess.run([sys.executable,str(Path(__file__).with_name('export_latex.py')),
                '--results',str(args.output),'--source',args.source],check=True)
