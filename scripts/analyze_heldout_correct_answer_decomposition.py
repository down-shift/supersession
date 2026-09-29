#!/usr/bin/env python3
"""Summarize signed/absolute correct-logit and margin patch changes."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def summarize(x,seed=20260930,n_boot=10000):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]; n=len(x)
    if not n:return {'n_histories':0,'mean':None,'ci95_bootstrap':None}
    rng=np.random.default_rng(seed); b=x[rng.integers(0,n,(n_boot,n))].mean(1)
    return {'n_histories':n,'mean':float(x.mean()),'ci95_bootstrap':[float(np.quantile(b,.025)),float(np.quantile(b,.975))]}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--diagnostics',required=True); p.add_argument('--output-dir',required=True); a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.diagnostics); required={'history_id','edited_binding','query_id','site_role','direction','patch_correct_logit_delta_mean','patch_correct_logit_delta_abs_mean','patch_correct_margin_delta_mean'}
    if not required<=set(d): raise ValueError(f'diagnostic fields missing: {sorted(required-set(d))}')
    # Since margin = correct - old-source, this difference reconstructs
    # the old-source logit change history by history.
    d['patch_old_source_logit_delta_mean']=d.patch_correct_logit_delta_mean-d.patch_correct_margin_delta_mean
    fields=['patch_correct_logit_delta_mean','patch_correct_logit_delta_abs_mean','patch_correct_margin_delta_mean','patch_old_source_logit_delta_mean']
    rows=[]; result={}
    for keys,g in d.groupby(['edited_binding','query_id','site_role','direction'],dropna=False):
        ent=dict(zip(['edited_binding','query_id','site_role','direction'],keys)); ent['exploratory_diagnostic']=True
        result['/'.join(map(str,keys))]={}
        for field in fields:
            vals=g[field].dropna(); s=summarize(vals,20260930+len(result)*13+fields.index(field)); result['/'.join(map(str,keys))][field]=s
            rows.append(ent|{'metric':field,**s})
    pd.DataFrame(rows).to_csv(out/'heldout_correct_answer_decomposition.csv',index=False)
    (out/'heldout_correct_answer_decomposition.json').write_text(json.dumps({'exploratory_diagnostic':True,'unit':'history_id','interpretation':'Signed correct-answer logit change measures the average confidence shift. Mean absolute change measures intervention magnitude. A larger correct-minus-old-source margin does not by itself mean the correct-answer logit increased; inspect the old-source component.','margin_identity':'delta(correct) - delta(old_source) = delta(correct_minus_old_source)','decomposition':'old_source_delta = correct_logit_delta - correct_minus_old_source_margin_delta','groups':result},indent=2)+'\n')
if __name__=='__main__': main()
