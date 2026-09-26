#!/usr/bin/env python3
"""Sanity-check notebooks/degp_battery.ipynb: the six cell blocks are
present, in order, and the 3-line e1_reward_r2 fix is applied."""
import json, sys

MARKERS = ['PRE-FLIGHT', 'DBATCH-DEF', 'CELL 2 SUPPLEMENT',
           'E-BASELINES', 'DRIVER', 'CELL V', 'K30']
FIX = 'rh.append(rewT(zt, a).item())'          # fixed dim version
BUG = 'rewT(zt.unsqueeze(0), a.unsqueeze(0))'  # buggy version

nb = json.load(open(sys.argv[1] if len(sys.argv) > 1
                    else 'notebooks/degp_battery.ipynb'))
src = '\n'.join(''.join(c.get('source', [])) for c in nb['cells'])
pos, ok = -1, True
for m in MARKERS:
    i = src.find(m)
    if i < 0 or i < pos:
        print(f'  MISSING/OUT-OF-ORDER: {m}'); ok = False
    else:
        pos = i; print(f'  ok: {m}')
if BUG in src:
    print('  !! e1_reward_r2 dim bug still present — apply the 3-line fix')
    ok = False
elif FIX in src:
    print('  ok: e1_reward_r2 fix applied')
print('NOTEBOOK VALID' if ok else 'NOTEBOOK INVALID'); sys.exit(0 if ok else 1)
