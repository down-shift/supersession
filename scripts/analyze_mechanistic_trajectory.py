#!/usr/bin/env python3
"""Exploratory history-bootstrap layer trajectory from targeted discovery."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROLES = ('edited_binding_value','same_variable_other_value','current_assignment_variable','query_variable','final_preanswer')

def boot_stats(values, seed=20260929, n_boot=10000):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]; n=len(x)
    if not n: return dict(n_histories=0,mean=None,median=None,trimmed_mean_10pct=None,fraction_positive=None,ci95_bootstrap=None)
    rng=np.random.default_rng(seed); b=x[rng.integers(0,n,(n_boot,n))].mean(axis=1)
    return dict(n_histories=n,mean=float(x.mean()),median=float(np.median(x)),trimmed_mean_10pct=float(pd.Series(x).sort_values().iloc[int(n*.1):int(np.ceil(n*.9))].mean()),fraction_positive=float(np.mean(x>0)),ci95_bootstrap=[float(np.quantile(b,.025)),float(np.quantile(b,.975))])

def trajectory(patches, output_dir, selected_layers=(33,34,35), n_boot=10000):
    p=Path(patches); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    if p.suffix=='.csv': f=pd.read_csv(p)
    else:
        f=pd.read_json(p,lines=True)
        # Reduce each raw direction to history × semantic site × layer effect,
        # then form current-x minus current-z with history as the pairing unit.
        keys=['history_id','query_id','site_role','layer','direction']
        e=f[f.edited_binding.eq('old_x')].groupby(keys,as_index=False).patch_delta_toward_donor.mean()
        e=e.groupby(['history_id','query_id','site_role','layer'],as_index=False).patch_delta_toward_donor.mean()
        f=e.pivot(index=['history_id','site_role','layer'],columns='query_id',values='patch_delta_toward_donor').reset_index()
        if not {'current_x','current_z'} <= set(f.columns): raise ValueError('discovery input needs old_x/current_x and old_x/current_z')
        f['R_x_patch']=f.current_x-f.current_z
    if 'R_x_patch' not in f: raise ValueError('input must contain R_x_patch')
    f=f[f.site_role.isin(ROLES)].copy(); f['layer']=f.layer.astype(int)
    rows=[]
    for (layer,role),g in f.groupby(['layer','site_role']):
        s=boot_stats(g.R_x_patch,20260929+layer*31+ROLES.index(role),n_boot)
        rows.append(dict(layer=int(layer),site_role=role,**s))
    tab=pd.DataFrame(rows).sort_values(['site_role','layer']); tab.to_csv(out/'trajectory_by_layer_site.csv',index=False)
    late=f[(f.site_role=='final_preanswer')&f.layer.isin(selected_layers)].groupby('history_id').R_x_patch.mean()
    late_mean=float(late.mean()) if len(late) else float('nan')
    diagnostics=[]
    for role,g in tab.groupby('site_role'):
        roledata=f[f.site_role.eq(role)]
        for threshold in (.10,.25,.50):
            candidates=g[g['mean'] >= threshold*late_mean] if late_mean>0 else g.iloc[0:0]
            diagnostics.append({'site_role':role,'diagnostic':f'first_crossing_{int(threshold*100)}pct_late_effect','value':int(candidates.layer.min()) if len(candidates) else None})
        if len(g):
            peak=g.loc[g['mean'].abs().idxmax()]
            diagnostics.append({'site_role':role,'diagnostic':'maximum_absolute_mean_Rx_layer','value':int(peak.layer)})
        for r in g.to_dict('records'):
            diagnostics.append({'site_role':role,'layer':int(r['layer']),'diagnostic':'fraction_of_late_effect','value':r['mean']/late_mean if late_mean else None})
    summary={'exploratory':True,'input':str(p),'pairing_unit':'history_id','late_reference':'mean per-history final_preanswer R_x_patch averaged across independently patched frozen layers 33, 34, 35','late_reference_mean':late_mean,'selected_layers':list(selected_layers),'layer_aggregation':'mean single-layer patch effect across frozen 33–35 region; layers are never patched simultaneously','interpretation_caveat':'A late-layer historical-token patch cannot alter the already-computed final-token representation from that same layer. Near-zero historical-token patch effects in layers 33–35 do not show that historical tokens were irrelevant earlier; the information may already have propagated to the final token. A large final_preanswer patch at layer 35 intervenes close to the readout representation and does not identify the upstream pathway.','landmarks':diagnostics,'no_layerwise_hypothesis_tests':True}
    (out/'trajectory_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    candidates=[]
    for r in tab.to_dict('records'):
        candidates.append({'site':r['site_role'],'layer':int(r['layer']),'mean_R_x_patch':r['mean'],'ci95_history_bootstrap':r['ci95_bootstrap'],'fraction_of_late_effect':r['mean']/late_mean if late_mean else None,'n_histories':int(r['n_histories']),'exploratory':True})
    (out.parent/'stage2_candidates.json').write_text(json.dumps({'exploratory':True,'selection_policy':'descriptive candidates only; no automatic ranking or intervention selection','candidates':candidates},indent=2)+'\n')
    # Pillow keeps artifact generation independent of matplotlib's local GUI/font stack.
    from PIL import Image, ImageDraw
    image=Image.new('RGB',(1200,720),'white'); draw=ImageDraw.Draw(image)
    left,right,top,bottom=90,1150,55,630
    finite=np.r_[tab['mean'].to_numpy(),*[np.asarray(v,dtype=float) for v in tab.ci95_bootstrap if isinstance(v,list)]]
    ymin,ymax=float(np.nanmin(finite)),float(np.nanmax(finite)); pad=max(.5,(ymax-ymin)*.08); ymin-=pad; ymax+=pad
    def xy(layer,value): return (left+(int(layer)-int(tab.layer.min()))/max(1,int(tab.layer.max())-int(tab.layer.min()))*(right-left),bottom-(value-ymin)/(ymax-ymin)*(bottom-top))
    for tick in np.linspace(ymin,ymax,7):
        y=xy(int(tab.layer.min()),tick)[1]; draw.line((left,y,right,y),fill='#dddddd'); draw.text((8,y-7),f'{tick:.1f}',fill='black')
    colors=['#1f77b4','#d62728','#2ca02c','#9467bd','#ff7f0e']
    for color,(role,g) in zip(colors,tab.groupby('site_role')):
        g=g.sort_values('layer'); pts=[]
        for r in g.to_dict('records'):
            lo,hi=r['ci95_bootstrap']; pts.append(xy(r['layer'],r['mean']))
            draw.line((*xy(r['layer'],lo),*xy(r['layer'],hi)),fill=color,width=2)
        if len(pts)>1: draw.line(pts,fill=color,width=3)
        for pt in pts: draw.ellipse((pt[0]-4,pt[1]-4,pt[0]+4,pt[1]+4),fill=color)
    draw.line((left,xy(int(tab.layer.min()),0)[1],right,xy(int(tab.layer.min()),0)[1]),fill='black',width=1)
    draw.text((left,12),'Exploratory targeted discovery trajectory; bars are 95% history-bootstrap CIs',fill='black')
    draw.text((left,bottom+20),'Transformer layer',fill='black'); draw.text((8,35),'Mean history-level R_x_patch',fill='black')
    for role,(color,_) in zip(tab.site_role.unique(),zip(colors,range(5))):
        pass
    legend_y=bottom+45
    for color,role in zip(colors,tab.site_role.drop_duplicates()):
        draw.line((left,legend_y+6,left+24,legend_y+6),fill=color,width=3); draw.text((left+30,legend_y),role,fill='black'); left+=230
    image.save(out/'Rx_by_layer_site.png')
    return tab,summary

def main():
    a=argparse.ArgumentParser(); a.add_argument('--patches',required=True); a.add_argument('--output-dir',required=True); a.add_argument('--selected-layers',default='33,34,35'); x=a.parse_args()
    trajectory(x.patches,x.output_dir,tuple(map(int,x.selected_layers.split(','))))
if __name__=='__main__': main()
