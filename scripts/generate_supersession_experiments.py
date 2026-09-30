#!/usr/bin/env python3
"""Generate fresh supersession controls, update-status pairs, or version chains."""
import argparse, json
from pathlib import Path
from src.data.supersession import make_control_history, make_status_history, make_version_chain, render_control_example, render_status_example, render_version_chain

p=argparse.ArgumentParser(); p.add_argument('--kind',choices=('controls','status','chains'),required=True); p.add_argument('--values',required=True,help='JSON array of candidate strings'); p.add_argument('--output',required=True); p.add_argument('--n',type=int,default=96); p.add_argument('--seed',type=int,default=73021); p.add_argument('--depths',default='0,1,2,4,8'); a=p.parse_args()
values=json.loads(Path(a.values).read_text()); rows=[]
for i in range(a.n):
    if a.kind=='controls':
        for condition in ('live','superseded','irrelevant'):
            h=make_control_history(f'control{i:05d}',values,condition,a.seed+i)
            h['rendered_prompt']=render_control_example(h); rows.append(h)
    elif a.kind=='status':
        for h in make_status_history(f'status{i:05d}',values,a.seed+i):
            h['rendered_prompt_x']=render_status_example(h,'x'); h['rendered_prompt_z']=render_status_example(h,'z'); rows.append(h)
    else:
        for depth in map(int,a.depths.split(',')):
            h=make_version_chain(f'chain{i:05d}:d{depth}',values,depth,a.seed+i*19+depth)
            h['rendered_prompt_x']=render_version_chain(h,'x'); h['rendered_prompt_z']=render_version_chain(h,'z'); rows.append(h)
out=Path(a.output)
if out.exists(): raise FileExistsError(f'{out} exists; choose a fresh output path')
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
