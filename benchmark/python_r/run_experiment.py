"""Build current local packages and run reproducible Python/R benchmarks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'benchmark/python_r'


def call(command, env=None, cwd=ROOT):
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    paths = []
    for folder in ('cpp', 'R', 'python/spectralmm', 'benchmark/python_r'):
        paths.extend(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and p.suffix in ('.cpp', '.hpp', '.h', '.py', '.R', '.jl')
                     and 'results' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts)
    paths.extend([ROOT/'cpp/CMakeLists.txt', ROOT/'R/DESCRIPTION', ROOT/'R/src/Makevars'])
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths))}


def build(env):
    # Canonical C++ edits are propagated before building either interface.
    call([sys.executable, ROOT/'tools/sync_package_sources.py'], env)
    target = ROOT/'benchmark/build'
    core = target/'core'
    command = ['cmake', '-S', ROOT/'cpp', '-B', core,
               '-DCMAKE_BUILD_TYPE=Release', '-DSPECTRALMM_USE_BLAS=AUTO',
               '-DSPECTRALMM_CHECK_PCG_ALLOCATIONS=OFF']
    eigen = os.environ.get('EIGEN3_INCLUDE_DIR')
    if not eigen:
        eigen = next((p for p in ('/opt/homebrew/include/eigen3','/usr/local/include/eigen3','/usr/include/eigen3')
                      if Path(p,'Eigen/Core').exists()), None)
    command += ([f'-DEIGEN3_INCLUDE_DIR={eigen}', '-DSPECTRALMM_FETCH_EIGEN=OFF'] if eigen
                else ['-DSPECTRALMM_FETCH_EIGEN=ON'])
    call(command, env)
    call(['cmake','--build',core,'--parallel',str(min(4,os.cpu_count() or 1))],env)
    library = next(p for p in (core/'libspectralmm.dylib',core/'libspectralmm.so',core/'spectralmm.dll') if p.exists())
    env['SPECTRALMM_LIBRARY'] = str(library)
    env['PYTHONPATH'] = str(ROOT/'python')
    rlib=target/'Rlib'; rlib.mkdir(parents=True,exist_ok=True)
    env['R_LIBS'] = str(rlib)
    rfiles=[p for p in (ROOT/'R').rglob('*') if p.is_file() and p.suffix not in ('.o','.so','.dll','.dylib','.rds') and p.name!='.DS_Store']
    signature=hashlib.sha256(''.join(str(p.relative_to(ROOT))+digest(p) for p in sorted(rfiles)).encode()).hexdigest()
    stamp=target/'R-source.sha256'
    if not stamp.exists() or stamp.read_text()!=signature or not (rlib/'SpectralMM').exists():
        staged=target/'Rsrc'
        if staged.exists(): shutil.rmtree(staged)
        shutil.copytree(ROOT/'R',staged,ignore=shutil.ignore_patterns('*.o','*.so','*.dll','*.dylib','symbols.rds','.DS_Store'))
        call(['R','CMD','INSTALL',f'--library={rlib}',staged],env)
        stamp.write_text(signature)
    rlibs=list((rlib/'SpectralMM/libs').rglob('*'))
    binaries={str(library):digest(library)}
    binaries.update({str(p):digest(p) for p in rlibs if p.is_file() and p.suffix in ('.so','.dll','.dylib')})
    return binaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',type=Path,default=HERE/'settings.json')
    parser.add_argument('--output',type=Path,required=True,help='New result directory; existing timing CSVs are never overwritten')
    parser.add_argument('--inputs',type=Path,help='Existing fixture directory containing data/ and optionally Julia.csv')
    parser.add_argument('--stage',choices=('all','initial','repeat','export'),default='all')
    args=parser.parse_args(); out=args.output.resolve()
    if args.stage=='export':
        call([sys.executable,HERE/'export_latex.py','--results',out,'--source','initial'])
        return
    settings=json.loads(args.settings.read_text())
    expected=set(json.loads((HERE/'settings.json').read_text()))
    if set(settings)!=expected: raise ValueError(f'Settings must contain these keys: {sorted(expected)}')
    if settings['threads']=='max': settings['threads']=os.cpu_count() or 1
    for key in ('threads','size','samples','repeat_samples','maxiter','inner_maxiter','max_batch','rank','krylovdim'):
        if type(settings[key]) is not int or settings[key]<=0:raise ValueError(f'{key} must be a positive integer')
    if not settings['rank']<settings['krylovdim']<=settings['size']+1: raise ValueError('Require rank < krylovdim <= size+1')
    for key in ('accept_negligible','accept_stalled'):
        if type(settings[key]) is not bool:raise ValueError(f'{key} must be boolean')
    if args.stage in ('all','initial') and any((out/f).exists() for f in ('python.csv','R.csv','python_repeat.csv','R_repeat.csv')):
        raise ValueError('Choose a fresh --output directory; timings already exist here')
    if args.stage=='repeat' and any((out/f).exists() for f in ('python_repeat.csv','R_repeat.csv')):
        raise ValueError('Repeat timings already exist; use a fresh experiment directory')
    out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key in ('SPECTRALMM_BENCH_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','JULIA_NUM_THREADS'):
        env[key]=str(settings['threads'])
    binaries=build(env)
    snapshot=sources()
    data=out/'data'
    if args.stage in ('all','initial'):
        if args.inputs:
            origin=args.inputs.resolve()
            if not (origin/'data').is_dir():raise ValueError('--inputs must contain data/')
            if data.exists():raise ValueError('Output already has data/; use a fresh directory')
            data.symlink_to(origin/'data',target_is_directory=True)
            if (origin/'Julia.csv').exists():shutil.copy2(origin/'Julia.csv',out/'Julia.csv')
        else:
            call(['julia','--startup-file=no',f'--project={ROOT / "julia"}',HERE/'family_performance.jl',settings['size'],out],env)
    inputs={p.name:digest(p) for p in sorted(data.glob('*.bin'))}
    if not inputs:raise ValueError('No input binaries found')
    # Validate dimensions before any native fit can read malformed fixture arrays.
    p=settings['size']; n=2*p
    families=['gaussian','bernoulli','probit','poisson','gamma','negative_binomial','smooth_quantile','expectile','pseudo_huber','student_t']
    for family in families:
        for storage in ('dense','sparse'):
            stem=data/f'{family}_{storage}_p{p}'
            for suffix,expected_bytes in [('_y.bin',8*n),('_beta0.bin',8*(p+1))]:
                if Path(str(stem)+suffix).stat().st_size!=expected_bytes:raise ValueError('Input size mismatch')
            if storage=='dense' and Path(str(stem)+'_X.bin').stat().st_size!=8*n*(p+1):raise ValueError('Dense input size mismatch')
    state=dict(settings=settings,sources=snapshot,binaries=binaries,input_sha256=inputs)
    statepath=out/'run_state.json'
    if args.stage=='repeat':
        if not statepath.exists() or json.loads(statepath.read_text())!=state:
            raise ValueError('Settings, sources, binaries or inputs changed since initial timing; start a fresh experiment')
    else: statepath.write_text(json.dumps(state,indent=2)+'\n')
    settingsfile=out/'settings.json';settingsfile.write_text(json.dumps(settings,indent=2)+'\n')
    stages=['initial','repeat'] if args.stage=='all' else [args.stage]
    for stage in stages:
        started=datetime.now(timezone.utc).isoformat()
        script='family_performance.py' if stage=='initial' else 'repeat_family_native.py'
        command=[sys.executable,HERE/script,'--output',out,'--language','both']
        if stage=='initial':command+=['--size',p,'--settings',settingsfile]
        call(command,env)
        names=([f'{lang}{suffix}.csv' for lang in ('python','R') for suffix in ('','_samples')]
               if stage=='initial' else
               [f'{lang}{suffix}.csv' for lang in ('python','R') for suffix in ('_repeat','_repeat_samples')])
        records={name:digest(out/name) for name in names}
        record=dict(state,started_utc=started,completed_utc=datetime.now(timezone.utc).isoformat(),
                    result_sha256=records)
        (out/f'provenance_{stage}.json').write_text(json.dumps(record,indent=2)+'\n')
    call([sys.executable,HERE/'export_latex.py','--results',out,'--source','initial'],env)
    if (out/'provenance_repeat.json').exists():
        call([sys.executable,HERE/'export_latex.py','--results',out,'--source','repeat'],env)


if __name__=='__main__':main()
