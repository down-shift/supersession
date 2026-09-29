#!/usr/bin/env python3
"""Export compact, direction-separated correct-answer diagnostics from patch JSONL."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--patches',required=True,help='held-out raw patch JSONL')
p.add_argument('--output',required=True,help='compact CSV output path')
p.add_argument('--summary',help='optional JSON summary path')
a=p.parse_args()
required=('history_id','edited_binding','query_id','site_role','layer','direction','patch_correct_logit_delta','correct_answer','recipient_correct_logit','donor_correct_logit','patched_correct_logit')
groups=defaultdict(lambda: defaultdict(list)); missing=set(); rows=0
with Path(a.patches).open(encoding='utf8') as f:
    for line_no,line in enumerate(f,1):
        if not line.strip(): continue
        row=json.loads(line); rows+=1
        if row.get('edited_binding') not in ('old_x','old_z'): continue
        absent=[key for key in required if key not in row]
        if absent: missing.update(absent); continue
        key=(row['history_id'],row['edited_binding'],row['query_id'],row['site_role'],row['direction'])
        for name in ('patch_correct_logit_delta','patch_correct_margin_delta','recipient_correct_logit','donor_correct_logit','patched_correct_logit'):
            val=row.get(name)
            if val is not None: groups[key][name].append(float(val))
        groups[key]['layers'].append(int(row['layer']))
        groups[key]['correct_answer'].append(row['correct_answer'])
if missing:
    raise SystemExit('Raw patch rows lack diagnostic fields: '+', '.join(sorted(missing))+'. These diagnostics require patch records generated with the diagnostic-enabled runner.')
fields=['history_id','edited_binding','query_id','site_role','direction','correct_answer','n_layers','recipient_correct_logit_mean','donor_correct_logit_mean','patched_correct_logit_mean','patch_correct_logit_delta_mean','patch_correct_logit_delta_abs_mean','patch_correct_margin_delta_mean']
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
with Path(a.output).open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for key,vals in sorted(groups.items()):
        def mean(name):
            xs=vals.get(name,[]); return sum(xs)/len(xs) if xs else ''
        deltas=vals.get('patch_correct_logit_delta',[])
        w.writerow(dict(zip(fields[:5],key)) | {'correct_answer':'|'.join(sorted(set(vals['correct_answer']))),'n_layers':len(set(vals['layers'])),'recipient_correct_logit_mean':mean('recipient_correct_logit'),'donor_correct_logit_mean':mean('donor_correct_logit'),'patched_correct_logit_mean':mean('patched_correct_logit'),'patch_correct_logit_delta_mean':mean('patch_correct_logit_delta'),'patch_correct_logit_delta_abs_mean':sum(abs(x) for x in deltas)/len(deltas) if deltas else '', 'patch_correct_margin_delta_mean':mean('patch_correct_margin_delta')})
if a.summary:
    Path(a.summary).parent.mkdir(parents=True,exist_ok=True)
    Path(a.summary).write_text(json.dumps({'input_rows':rows,'exported_history_binding_query_site_direction_groups':len(groups),'grouping':'history_id × edited_binding × query_id × site_role × direction; mean over frozen held-out layers','directions_kept_separate':True},indent=2)+'\n',encoding='utf8')
print(f'Wrote {len(groups)} compact diagnostic rows to {a.output}')
