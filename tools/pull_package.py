#!/usr/bin/env python3
"""Populate degp/ with the artifact package files from a Kaggle export.

Usage:
    python tools/pull_package.py /path/to/exported_saved_files [--force]

Copies the eight package .py files into degp/, printing SHA-256 hashes.
Refuses to overwrite existing files unless --force."""
import argparse, glob, hashlib, os, shutil, sys

FILES = ('__init__.py', 'planner.py', 'leash.py', 'planning.py',
         'phasec2.py', 'phaseb.py', 'predictor.py', 'action_diagnostics.py')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    dst = os.path.join(os.path.dirname(__file__), '..', 'degp')
    os.makedirs(dst, exist_ok=True)
    missing = []
    for f in FILES:
        hits = sorted(set(glob.glob(os.path.join(a.src, f),
                                    recursive=False))
                      | set(glob.glob(os.path.join(a.src, '**', 'degp', f),
                                      recursive=True)))
        if not hits:
            missing.append(f); continue
        out = os.path.join(dst, f)
        if os.path.exists(out) and not a.force:
            print(f'  = {f} (exists, kept)'); continue
        shutil.copy2(hits[0], out)
        h = hashlib.sha256(open(out, 'rb').read()).hexdigest()
        print(f'  + {f}  sha256:{h[:16]}...')
    if missing:
        print(f'!! missing: {missing}'); sys.exit(1)
    print('degp/ complete.')

if __name__ == '__main__':
    main()
