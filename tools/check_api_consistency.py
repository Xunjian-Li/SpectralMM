#!/usr/bin/env python3
"""Compare deterministic statistical-API exports from installed language packages."""
import argparse
import csv
import math
from pathlib import Path


def read(path):
    groups={}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            backend=row.get('backend','cpp')
            key=(row['family'],row['quantity'],int(row['index']))
            values=groups.setdefault(backend,{})
            if key in values: raise AssertionError(f'duplicate row: {path}: {key}')
            values[key]=float(row['value'])
    return groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python',required=True); p.add_argument('--r',required=True); p.add_argument('--julia',required=True)
    args=p.parse_args()
    reference=read(args.python)['cpp']
    assert len(reference)==6*(4+100+100+4), 'incomplete six-model fixture export'
    targets={'R':read(args.r)['cpp']}
    targets.update({'Julia/'+k:v for k,v in read(args.julia).items()})
    assert 'Julia/julia' in targets and 'Julia/cpp' in targets
    for label,values in targets.items():
        assert values.keys()==reference.keys(),f'{label}: mismatched model quantities'
        maximum=0.
        for key,expected in reference.items():
            actual=values[key]
            assert math.isfinite(actual) and math.isfinite(expected),f'{label}: nonfinite {key}'
            if key[1]=='converged': assert actual==expected==1.,f'{label}: nonconvergence {key}'
            else:
                # Existing 1e-7 absolute / 1e-8 relative gradient criteria allow
                # slightly different trajectories across BLAS libraries/backends.
                # These well-conditioned small designs use a fixed 1e-6 output
                # tolerance, not a per-case adjusted threshold.
                assert math.isclose(actual,expected,rel_tol=1e-7,abs_tol=1e-6),f'{label}: {key}: {actual} != {expected}'
                maximum=max(maximum,abs(actual-expected))
        print(f'{label}: {len(values)} quantities passed; maximum absolute difference {maximum:.6g}')

if __name__=='__main__': main()
