"""Public entry point for the Python/R experiment; run with --help for options."""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parent/'python_r/run_experiment.py'), run_name='__main__')
