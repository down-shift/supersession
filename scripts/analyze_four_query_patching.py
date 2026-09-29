#!/usr/bin/env python3
"""History-level analysis of donor-oriented four-query residual patches."""
import argparse, collections, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.io import read_jsonl
from src.utils import save_json
from src.analysis.metrics import trimmed_mean
from src.experiments.patching import r_x_patch, symmetric_r_patch

def stats(values,seed=0,n_boot=10000,n_perm=10000,inferential=True):
 x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]; n=len(x)
 if not n:return {'n_histories':0,'mean':None,'median':None,'trimmed_mean_10pct':None,'fraction_positive':None,'ci95_cluster_bootstrap':None,**({'sign_flip_p_two_sided':None} if inferential else {})}
 rng=np.random.default_rng(seed); boot=np.mean(rng.choice(x,(n_boot,n),replace=True),axis=1)
 result={'n_histories':n,'mean':float(x.mean()),'median':float(np.median(x)),'trimmed_mean_10pct':float(trimmed_mean(x,.1)),'fraction_positive':float(np.mean(x>0)),'ci95_cluster_bootstrap':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
 if inferential:
  observed=abs(float(x.mean())); flips=rng.choice(np.array([-1.,1.]),(n_perm,n)); result['sign_flip_p_two_sided']=float((1+np.sum(np.abs(np.mean(flips*x,axis=1))>=observed))/(n_perm+1))
 return result

def main():
 p=argparse.ArgumentParser(); p.add_argument('--patches',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--stage',choices=('discovery','heldout'),required=True); p.add_argument('--seed',type=int,default=20260929); p.add_argument('--selection-site',default='final_preanswer'); p.add_argument('--region-width',type=int,choices=(3,),default=3); a=p.parse_args()
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); rows=read_jsonl(a.patches)
 required={'history_id','query_id','edited_binding','direction','layer','patch_delta_toward_donor'}
 if rows and not required.issubset(rows[0]): raise ValueError(f'missing patch fields: {sorted(required-set(rows[0]))}')
 for r in rows:
  r.setdefault('position',0)
  r.setdefault('site_role',r.get('site','other_position')); r.setdefault('site_label',r.get('site','other_position'))
  aliases={'donor_to_recipient':'baseline_to_edited','recipient_to_donor':'edited_to_baseline'}
  r['direction']=aliases.get(r['direction'],r['direction'])
 frame=pd.DataFrame(rows)
 # Each absolute position remains its own cell in all-position data. Semantic
 # multi-token spans are averaged within direction by role.
 allpos=bool(frame.site_role.eq('other_position').any())
 keys=['history_id','edited_binding','query_id','site_role','layer']+(['position'] if allpos else [])
 d=frame.groupby(keys+['direction'],dropna=False).patch_delta_toward_donor.mean().reset_index()
 pivot=d.pivot(index=keys,columns='direction',values='patch_delta_toward_donor').reset_index()
 needed={'baseline_to_edited','edited_to_baseline'}
 if not needed.issubset(pivot.columns): raise ValueError('both donor-oriented patch directions are required')
 pivot['mean_donor_oriented_patch_effect']=pivot[list(needed)].mean(axis=1)
 pivot['effect']=pivot['mean_donor_oriented_patch_effect']
 if a.stage=='heldout':
  derived=[]
  groupkeys=['history_id','site_role','layer']+(['position'] if allpos else [])
  for key,g in pivot.groupby(groupkeys,dropna=False):
   by={(r.edited_binding,r.query_id):float(r.effect) for r in g.itertuples()}
   if all(k in by for k in [('old_x','current_x'),('old_x','current_z'),('old_z','current_z'),('old_z','current_x')]):
    entry=dict(zip(groupkeys,key if isinstance(key,tuple) else (key,)))
    entry.update({'measure':'primary_contrasts','R_x_patch':r_x_patch(by['old_x','current_x'],by['old_x','current_z']),'R_z_patch':r_x_patch(by['old_z','current_z'],by['old_z','current_x'])})
    entry['symmetric_R_patch']=symmetric_r_patch(entry['R_x_patch'],entry['R_z_patch']); derived.append(entry)
  if derived:
   # Separate derived contrasts make the held-out history-level primary easy to audit.
   pd.DataFrame(derived).to_csv(out/'heldout_R_by_history.csv',index=False)
 pivot.to_csv(out/('heldout_patch_effects_by_history.csv' if a.stage=='heldout' else 'patch_effects_by_history.csv'),index=False)
 def cell(binding,query,layer,role,position=None):
  f=pivot[(pivot.edited_binding==binding)&(pivot.query_id==query)&(pivot.layer==layer)&(pivot.site_role==role)]
  if position is not None:f=f[f.position==position]
  return dict(zip(f.history_id,f.effect))
 layers=sorted(int(x) for x in pivot.layer.unique()); roles=sorted(set(pivot.site_role)); results={}; perlayer={}
 positions=sorted(int(x) for x in pivot.position.unique()) if allpos else [None]
 for role in roles:
  for layer in layers:
   for position in positions:
    c={(b,q):cell(b,q,layer,role,position) for b in ('old_x','current_x','old_z','current_z') for q in ('current_x','initial_x','current_z','initial_z')}
    rxs={h:r_x_patch(c['old_x','current_x'][h],c['old_x','current_z'][h]) for h in set(c['old_x','current_x'])&set(c['old_x','current_z'])}
    rzs={h:r_x_patch(c['old_z','current_z'][h],c['old_z','current_x'][h]) for h in set(c['old_z','current_z'])&set(c['old_z','current_x'])}
    both=set(rxs)&set(rzs); sym={h:symmetric_r_patch(rxs[h],rzs[h]) for h in both}
    sxids=set(c['old_x','initial_x'])&set(c['old_x','current_x'])&set(c['current_x','current_x'])&set(c['current_x','initial_x'])
    szids=set(c['old_z','initial_z'])&set(c['old_z','current_z'])&set(c['current_z','current_z'])&set(c['current_z','initial_z'])
    sx={h:.5*((c['old_x','initial_x'][h]-c['old_x','current_x'][h])+(c['current_x','current_x'][h]-c['current_x','initial_x'][h])) for h in sxids}
    sz={h:.5*((c['old_z','initial_z'][h]-c['old_z','current_z'][h])+(c['current_z','current_z'][h]-c['current_z','initial_z'][h])) for h in szids}
    key=f'{role}/layer_{layer}'+(f'/position_{position}' if position is not None else ''); results[key]={'R_x_patch':stats(list(rxs.values()),a.seed,inferential=not allpos),'R_z_patch':stats(list(rzs.values()),a.seed+1,inferential=not allpos),'symmetric_R_patch':stats(list(sym.values()),a.seed+2,inferential=not allpos),'S_x_control':stats(list(sx.values()),a.seed+3,inferential=not allpos),'S_z_control':stats(list(sz.values()),a.seed+4,inferential=not allpos),'cells_present':{f'{b}/{q}':len(v) for (b,q),v in c.items()}}
    perlayer[(role,layer,position)]=rxs
 region_primary={}; region_rows=[]
 correct_diagnostics={}
 if a.stage=='heldout':
  regionkeys=['history_id','edited_binding','query_id','site_role']+(['position'] if allpos else [])
  grouped=pivot.groupby(regionkeys,dropna=False).effect.agg(['mean','count']).reset_index()
  if not grouped.empty and not (grouped['count']==len(layers)).all(): raise ValueError('held-out data are incomplete across the frozen layer region')
  diagnostic_fields=('patch_correct_logit_delta','patch_correct_margin_delta')
  if set(diagnostic_fields).issubset(frame.columns):
   dx=frame[frame.edited_binding.isin(['old_x','old_z'])]
   dg=dx.groupby(['history_id','edited_binding','query_id','site_role','direction','layer'],dropna=False)[list(diagnostic_fields)].mean().reset_index()
   dg=dg.groupby(['history_id','edited_binding','query_id','site_role','direction'],dropna=False)[list(diagnostic_fields)].mean().reset_index()
   dg.to_csv(out/'heldout_correct_answer_diagnostics_by_history.csv',index=False)
   for keys_,g in dg.groupby(['edited_binding','query_id','site_role','direction'],dropna=False):
    label='/'.join(str(x) for x in keys_)
    correct_diagnostics[label]={}
    for field in diagnostic_fields:
     values=g[field].dropna().to_numpy()
     correct_diagnostics[label][field]=stats(values,a.seed)
     correct_diagnostics[label][field+'_absolute']=stats(np.abs(values),a.seed)
  for role in sorted(set(grouped.site_role)):
   subset=grouped[grouped.site_role==role]
   positions_for_role=sorted(int(x) for x in subset.position.unique()) if allpos else [None]
   for position in positions_for_role:
    f=subset if position is None else subset[subset.position==position]
    cells={}
    for b in ('old_x','old_z','current_x','current_z'):
     for q in ('current_x','initial_x','current_z','initial_z'):
      z=f[(f.edited_binding==b)&(f.query_id==q)]; cells[(b,q)]=dict(zip(z.history_id,z['mean']))
    rxids=set(cells['old_x','current_x'])&set(cells['old_x','current_z']); rzids=set(cells['old_z','current_z'])&set(cells['old_z','current_x'])
    rx={h:r_x_patch(cells['old_x','current_x'][h],cells['old_x','current_z'][h]) for h in rxids}; rz={h:r_x_patch(cells['old_z','current_z'][h],cells['old_z','current_x'][h]) for h in rzids}
    both=set(rx)&set(rz); sy={h:symmetric_r_patch(rx[h],rz[h]) for h in both}; key=role+(f'/position_{position}' if position is not None else '')
    region_primary[key]={'R_x_patch':stats(list(rx.values()),a.seed),'R_z_patch':stats(list(rz.values()),a.seed+1),'symmetric_R_patch':stats(list(sy.values()),a.seed+2),'frozen_layers':layers}
    for h in both: region_rows.append({'history_id':h,'layer':'selected_region','site_role':role,'position':position,'edited_binding':'symmetric','query_id':'symmetric_R_patch','R_x_patch':rx[h],'R_z_patch':rz[h],'symmetric_R_patch':sy[h]})
 if a.stage=='discovery' and not allpos:
  sel={}; siteframe=pivot[(pivot.site_role==a.selection_site)]
  actual_layers=sorted(int(x) for x in siteframe.layer.unique()); by={}
  for layer in actual_layers:
   c={(b,q):cell(b,q,layer,a.selection_site) for b,q in [('old_x','current_x'),('old_x','current_z')]}
   ids=set(c['old_x','current_x'])&set(c['old_x','current_z']); by[layer]={h:r_x_patch(c['old_x','current_x'][h],c['old_x','current_z'][h]) for h in ids}
  windows=[]
  for i in range(len(actual_layers)-a.region_width+1):
   win=actual_layers[i:i+a.region_width]
   if win==list(range(win[0],win[0]+a.region_width)):
    idsets=[set(by[l]) for l in win]; common=set.intersection(*idsets) if idsets else set()
    vals={h:float(np.mean([by[l][h] for l in win])) for h in common}
    windows.append((float(np.mean(list(vals.values()))) if vals else float('-inf'),win[0],win,len(common)))
  if not windows: raise ValueError('no complete contiguous three-layer selection window')
  best=max(windows,key=lambda x:(x[0],-x[1]))
  sel={'selection_statistic':{'name':'R_x_patch','formula':'old_x/current_x - old_x/current_z','site':a.selection_site,'window_width':a.region_width,'selected_layers':best[2],'tie_break':'lowest_start_layer'},'per_layer_R_x_patch':{str(l):stats(list(by[l].values()),a.seed) for l in actual_layers},'selected_window_mean_R_x_patch':best[0],'complete_histories_in_window':best[3]}
  save_json(sel,out/'discovery_selection.json')
 summary={'stage':a.stage,'exploratory':allpos,'results':results,'selected_region_primary':region_primary if a.stage=='heldout' else None,'correct_answer_diagnostics':correct_diagnostics if a.stage=='heldout' else None,'primary':'R_x_patch at discovery; selected-region symmetric_R_patch at heldout','frozen_layer_aggregation':'mean single-layer patch effect across frozen layers; layers are patched independently, never simultaneously','cells_present':sorted({f"{r.edited_binding}/{r.query_id}" for r in frame.itertuples()}),'bootstrap_unit':'history_id','direction_order':'donor-oriented each direction, then averaged','selection':sel if a.stage=='discovery' and not allpos else None,'per_layer_inference':'not performed' if allpos else 'existing targeted discovery procedure'}
 save_json(summary,out/'four_query_patch_summary.json' if a.stage=='discovery' else out/'heldout_patch_summary.json')
 if allpos:
  # Absolute position is part of the analysis key; never pool token locations.
  # Token identity can differ between query variants at the same position
  # (especially the queried x/z token), so it must not participate in pairing.
  ps=pivot[pivot.edited_binding=='old_x']
  # Pair solely by history, layer, and absolute position. A token may receive
  # different semantic labels in the two query variants when tokenization shifts.
  abs_effect=ps.groupby(['history_id','layer','position','query_id'],as_index=False).effect.mean()
  wide=abs_effect.pivot(index=['history_id','layer','position'],columns='query_id',values='effect').reset_index()
  wide['R_x_patch']=wide['current_x']-wide['current_z']
  roles=ps.groupby(['history_id','layer','position','query_id']).site_role.agg(lambda x:'|'.join(sorted(set(map(str,x))))).unstack('query_id').reset_index()
  roles=roles.rename(columns={q:f'semantic_site_role_{q}' for q in ('current_x','current_z')})
  wide=wide.merge(roles,on=['history_id','layer','position'],how='left')
  wide.to_csv(out/'all_positions_Rx_by_history.csv',index=False)
  agg=wide.groupby(['layer','position']).R_x_patch.agg(['mean','count']).reset_index()
  tokmeta=frame[frame.edited_binding=='old_x'].groupby(['position','query_id'],as_index=False).agg(
   token_id=('token_id',lambda x:'|'.join(map(str,sorted(set(x.dropna()))))),
   token_text=('token_text',lambda x:' | '.join(sorted(set(str(v) for v in x.dropna())))))
  tokwide=tokmeta.pivot(index='position',columns='query_id',values=['token_id','token_text'])
  tokwide.columns=[f'{field}_{query}' for field,query in tokwide.columns]
  agg=agg.merge(tokwide.reset_index(),on='position',how='left')
  agg.to_csv(out/'all_positions_Rx.csv',index=False)
  try:
   from PIL import Image,ImageDraw
   mat=agg.pivot(index='layer',columns='position',values='mean').sort_index(); vals=mat.to_numpy(dtype=float); lo=float(np.nanmin(vals)); hi=float(np.nanmax(vals)); span=max(1e-12,hi-lo)
   im=Image.new('RGB',(max(700,mat.shape[1]*14),max(450,mat.shape[0]*10)),'white'); d=ImageDraw.Draw(im); cw=max(1,im.width//mat.shape[1]); ch=max(1,(im.height-60)//mat.shape[0])
   for yi,row in enumerate(vals):
    for xi,v in enumerate(row):
     t=0 if not np.isfinite(v) else (v-lo)/span; color=(int(40+215*t),int(70+100*(1-abs(2*t-1))),int(255-215*t)); d.rectangle((xi*cw,yi*ch,(xi+1)*cw,(yi+1)*ch),fill=color)
   d.text((8,im.height-45),'x: absolute input token position; y: transformer layer; color: mean history-level R_x_patch',fill='black'); im.save(out/'all_positions_Rx.png')
  except ImportError: pass
  legend=frame[frame.edited_binding=='old_x'].groupby(['position','query_id'],as_index=False).agg(
   token_text=('token_text',lambda x:' | '.join(sorted(set(str(v) for v in x.dropna())))),
   semantic_site_role=('site_role',lambda x:' | '.join(sorted(set(map(str,x.dropna()))))))
  token_texts=tokmeta.pivot(index='position',columns='query_id',values='token_text')
  token_texts.columns=[f'token_text_{q}' for q in token_texts.columns]
  rolewide=legend.pivot(index='position',columns='query_id',values='semantic_site_role')
  rolewide.columns=[f'semantic_site_role_{q}' for q in rolewide.columns]
  tokenlegend=token_texts.join(rolewide,how='outer').reset_index()
  rough={'old_x_value':'initial_x_assignment','old_x_variable':'initial_x_assignment','old_z_value':'initial_z_assignment','old_z_variable':'initial_z_assignment','current_x_value':'current_x_assignment','current_x_variable':'current_x_assignment','current_z_value':'current_z_assignment','current_z_variable':'current_z_assignment','query_variable':'question','final_preanswer':'answer_prefix','edited_binding_value':'assignment_value','same_variable_other_value':'assignment_value','current_assignment_variable':'current_assignment','other_position':'rendered_text_unclassified'}
  semantic_cols=[c for c in tokenlegend if c.startswith('semantic_site_role_')]
  tokenlegend['semantic_site_role']=' | '.join([])
  tokenlegend['rough_rendered_region']='rendered_text_unclassified'
  for i,row in tokenlegend.iterrows():
   roles_here=sorted({role for col in semantic_cols if pd.notna(row[col]) for role in str(row[col]).split(' | ') if role!='other_position'})
   tokenlegend.at[i,'semantic_site_role']=' | '.join(roles_here) if roles_here else 'other_position'
   regions=sorted({rough.get(role,'rendered_text_unclassified') for role in roles_here})
   tokenlegend.at[i,'rough_rendered_region']=' | '.join(regions) if regions else 'rendered_text_unclassified'
  tokenlegend['site_role']=tokenlegend['semantic_site_role']
  tokenlegend.to_csv(out/'all_positions_token_legend.csv',index=False)
  # Semantic aggregation averages token positions within each semantic role
  # separately for each history and layer before group summaries.
  semantic_rows=[]
  for r in wide.itertuples():
   roles_here=set()
   for name in ('semantic_site_role_current_x','semantic_site_role_current_z'):
    value=getattr(r,name,None)
    if isinstance(value,str): roles_here.update(value.split('|'))
   for role in roles_here-{'other_position'}:
    semantic_rows.append({'history_id':r.history_id,'layer':r.layer,'site_role':role,'R_x_patch':r.R_x_patch})
  sem=pd.DataFrame(semantic_rows).groupby(['history_id','layer','site_role'],as_index=False).R_x_patch.mean()
  sem.to_csv(out/'all_positions_Rx_semantic_by_history.csv',index=False)
  semagg=sem.groupby(['layer','site_role']).R_x_patch.agg(['mean','median','count']).reset_index().rename(columns={'count':'n_histories'})
  semagg.to_csv(out/'all_positions_Rx_semantic.csv',index=False)
  mapping=wide.melt(id_vars=['history_id','layer','position'],value_vars=[c for c in wide if c.startswith('semantic_site_role_')],value_name='site_roles').dropna(subset=['site_roles'])
  mapping=mapping.assign(site_role=mapping.site_roles.str.split('|')).explode('site_role'); mapping=mapping[mapping.site_role.ne('other_position')]
  mapping=mapping.drop_duplicates(['history_id','layer','position','site_role'])
  varying=mapping.groupby('site_role').agg(histories=('history_id','nunique'),min_position=('position','min'),max_position=('position','max'),absolute_positions=('position','nunique')).reset_index()
  (out/'all_positions_semantic_position_mapping.json').write_text(json.dumps({'pairing_key':['history_id','layer','absolute_position'],'token_identity_in_pairing_key':False,'absolute_position_variation_by_role':varying.to_dict('records'),'semantic_aggregation':'within history × layer × semantic_site_role, mean over positions before group mean'},indent=2)+'\n')
  try:
   from PIL import Image,ImageDraw
   im=Image.new('RGB',(1000,520),'white'); d=ImageDraw.Draw(im); roles=list(semagg.site_role.unique()); layers=sorted(semagg.layer.unique()); colors=['#1f77b4','#d62728','#2ca02c','#9467bd','#ff7f0e','#17becf']; ys=[]
   lo=float(semagg['mean'].min()); hi=float(semagg['mean'].max()); span=max(1e-12,hi-lo); left,right,top,bottom=85,970,30,450
   for ri,role in enumerate(roles):
    g=semagg[semagg.site_role==role].sort_values('layer'); pts=[]
    for r in g.itertuples(): pts.append((left+(r.layer-min(layers))/max(1,max(layers)-min(layers))*(right-left),bottom-(r.mean-lo)/span*(bottom-top)))
    if len(pts)>1:d.line(pts,fill=colors[ri%len(colors)],width=3)
    d.text((left+ri*145,bottom+20),role,fill=colors[ri%len(colors)])
   d.text((left,im.height-25),'Semantic role; mean history-level R_x_patch',fill='black'); im.save(out/'all_positions_Rx_semantic.png')
  except ImportError: pass
 # compact per-history layer/site table for R contrasts
 outrows=[]
 for (role,layer,position),vals in perlayer.items():
  for h,v in vals.items():outrows.append({'history_id':h,'layer':layer,'site_role':role,'position':position,'edited_binding':'old_x','query_id':'current_x-current_z','mean_donor_oriented_patch_effect':v,'R_x_patch':v})
 pd.DataFrame(outrows).to_csv(out/'patch_Rx_by_layer_site.csv',index=False)
 if region_rows: pd.DataFrame(region_rows).to_csv(out/'heldout_R_by_history.csv',index=False)
 print(f'wrote {a.stage} patch analysis to {out}')
if __name__=='__main__': main()
