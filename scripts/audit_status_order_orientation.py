#!/usr/bin/env python3
"""CPU-only factorial audit of literal-variable, block-position, and orientation effects."""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from src.analysis.metrics import bootstrap_mean_ci
from src.analysis.supersession_behavior import audited_effects, history_contrasts
from src.data.io import read_jsonl, sha256_file
from src.utils import provenance, save_json

METRICS=('R_accepted_current','R_superseded_initial','R_retained_initial_after_rejection','R_rejected_update')

def _summary(values,seed,draws):
    values=np.asarray(values,dtype=float); ci=bootstrap_mean_ci(values,n_boot=draws,seed=seed)
    return {'n_histories':len(values),'mean':float(values.mean()),'median':float(np.median(values)),
            'fraction_positive':float(np.mean(values>0)),'ci95_history_bootstrap':[ci['ci_low'],ci['ci_high']]}

def analyze_order(dataset,scored,seed=20261003,draws=3000):
    effects=audited_effects(dataset,scored,'status')
    contrasts=history_contrasts(effects,'status')
    histories={}
    for row in dataset:
        old=histories.setdefault(row['history_id'],(tuple(row['variables']),int(row['orientation'])))
        if old!=(tuple(row['variables']),int(row['orientation'])):
            raise ValueError('history has inconsistent variable mapping/orientation')
    for hid,(variables,orientation) in histories.items():
        expected=('x','z') if orientation==0 else ('z','x')
        if variables!=expected: raise ValueError(f'{hid}: orientation does not match literal variable mapping')
    component_rows=[]; asymmetry_rows=[]
    for h in contrasts:
        variables,orientation=histories[h['history_id']]
        for metric in METRICS:
            rx,rz=h[metric+'_x'],h[metric+'_z']
            for role,value in (('x',rx),('z',rz)):
                literal=variables[0 if role=='x' else 1]
                component_rows.append({'history_id':h['history_id'],'orientation':orientation,'metric':metric,
                    'semantic_edited_variable':role,'literal_variable':literal,
                    'source_block':'update' if metric in ('R_accepted_current','R_rejected_update') else 'assignment',
                    'block_position':'first' if role=='x' else 'second','relevance_effect':value})
            d=rx-rz
            asymmetry_rows.append({'history_id':h['history_id'],'orientation':orientation,'metric':metric,
                'first_minus_second':d,
                'literal_x_minus_literal_z':d if orientation==0 else -d,
                'R_first':rx,'R_second':rz})
    f=pd.DataFrame(component_rows)
    grouped=[]
    group_cols=['metric','literal_variable','source_block','block_position','orientation']
    for key,g in f.groupby(group_cols,sort=True):
        record=dict(zip(group_cols,key)); record.update(_summary(g.relevance_effect,seed,draws)); grouped.append(record)
    asym=pd.DataFrame(asymmetry_rows); asym_summaries=[]
    for (metric,orientation),g in asym.groupby(['metric','orientation'],sort=True):
        for name in ('first_minus_second','literal_x_minus_literal_z'):
            asym_summaries.append({'metric':metric,'orientation':int(orientation),'contrast':name,
                                   **_summary(g[name],seed+orientation+len(name),draws)})
    factorial=[]
    rng=np.random.default_rng(seed)
    for metric,g in asym.groupby('metric',sort=True):
        by_orientation=g.groupby('orientation')
        values0=by_orientation.get_group(0).first_minus_second.to_numpy(float)
        values1=by_orientation.get_group(1).first_minus_second.to_numpy(float)
        d0=float(values0.mean()); d1=float(values1.mean())
        b0=rng.choice(values0,size=(draws,len(values0)),replace=True).mean(axis=1)
        b1=rng.choice(values1,size=(draws,len(values1)),replace=True).mean(axis=1)
        pos=(b0+b1)/4; lit=(b0-b1)/4
        factorial.append({'metric':metric,'n_histories':int(g.history_id.nunique()),
            'mean_first_minus_second_orientation0':d0,'mean_first_minus_second_orientation1':d1,
            'position_main_effect_half_scale':(d0+d1)/4,
            'literal_x_vs_z_main_effect_half_scale':(d0-d1)/4,
            'position_main_effect_ci95':[float(np.quantile(pos,.025)),float(np.quantile(pos,.975))],
            'literal_x_vs_z_main_effect_ci95':[float(np.quantile(lit,.025)),float(np.quantile(lit,.975))],
            'interpretation_note':'Factorial decomposition assumes additive position and literal-name effects; inspect orientation-specific contrasts for interaction.'})
    # One baseline score per unique task prompt. The legacy status data repeat these prompts
    # across counterfactual edit fields, so deduplicate and verify score diagnostics agree.
    by_prompt=defaultdict(list)
    for r in scored:
        if r['pair_direction']!=0: continue
        by_prompt[(r['history_id'],r['condition'],r['query'])].append(r)
    task_rows=[]
    for (hid,condition,query),items in by_prompt.items():
        for k in ('full_vocab_next_token_accuracy','accuracy','full_vocab_rank','candidate_rank'):
            if len({x[k] for x in items})!=1: raise ValueError('duplicated baseline task prompt has inconsistent score diagnostics')
        variables,orientation=histories[hid]; literal=variables[0 if query=='x' else 1]
        task_rows.append({'history_id':hid,'condition':condition,'query_role':query,'literal_variable':literal,
                          'orientation':orientation,'binding_position':'first' if query=='x' else 'second',
                          'full_vocab_next_token_accuracy':items[0]['full_vocab_next_token_accuracy'],
                          'candidate_accuracy':items[0]['accuracy'],'full_vocab_rank':items[0]['full_vocab_rank'],
                          'candidate_rank':items[0]['candidate_rank']})
    task=pd.DataFrame(task_rows); task_summary=[]
    for keys,g in task.groupby(['condition','literal_variable','orientation','binding_position'],sort=True):
        condition,literal,orientation,position=keys
        task_summary.append({'condition':condition,'literal_variable':literal,'orientation':int(orientation),
             'binding_position':position,'n_histories':int(g.history_id.nunique()),
             'full_vocab_accuracy':float(g.full_vocab_next_token_accuracy.mean()),
             'candidate_accuracy':float(g.candidate_accuracy.mean()),
             'full_vocab_rank_mean':float(g.full_vocab_rank.mean()),'full_vocab_rank_median':float(g.full_vocab_rank.median())})
    return {'per_history_components':f,'component_strata':pd.DataFrame(grouped),
            'per_history_asymmetry':asym,'asymmetry_strata':pd.DataFrame(asym_summaries),
            'factorial_decomposition':pd.DataFrame(factorial),'task_rows':task,
            'task_competence_strata':pd.DataFrame(task_summary)}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--behavior',required=True)
    p.add_argument('--output-dir',required=True); p.add_argument('--seed',type=int,default=20261003)
    p.add_argument('--bootstrap-draws',type=int,default=3000); a=p.parse_args()
    out=Path(a.output_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'{out} is nonempty; choose a fresh output directory')
    dataset,scored=read_jsonl(a.dataset),read_jsonl(a.behavior)
    result=analyze_order(dataset,scored,a.seed,a.bootstrap_draws)
    out.mkdir(parents=True,exist_ok=True)
    file_map={'history_R_components.csv':'per_history_components','R_component_strata.csv':'component_strata',
              'history_xz_asymmetry.csv':'per_history_asymmetry','xz_asymmetry_strata.csv':'asymmetry_strata',
              'factorial_order_decomposition.csv':'factorial_decomposition','unique_task_rows.csv':'task_rows',
              'task_competence_strata.csv':'task_competence_strata'}
    for filename,key in file_map.items(): result[key].to_csv(out/filename,index=False)
    save_json({'analysis':'cpu_status_order_orientation_audit','n_histories':len({r['history_id'] for r in dataset}),
        'design_mapping':'orientation 0: semantic x/literal x/first and semantic z/literal z/second; orientation 1 reverses literal names while retaining semantic block positions.',
        'position_main_effect_half_scale':'(D_orientation0 + D_orientation1)/4 where D is per-history R_first-R_second',
        'literal_name_main_effect_half_scale':'(D_orientation0 - D_orientation1)/4; additive decomposition, inspect interactions',
        'task_rows':'NO baseline prompt score deduplicated over edit fields; accuracy/ranks only',
        'bootstrap_unit':'history_id','seed':a.seed,'bootstrap_draws':a.bootstrap_draws,
        'inputs_sha256':{'dataset':sha256_file(a.dataset),'behavior':sha256_file(a.behavior)},
        'provenance':provenance({'seed':a.seed},a.dataset)},out/'audit.json')
    print(f'wrote CPU order/orientation audit for {len({r["history_id"] for r in dataset})} histories to {out}')


if __name__=='__main__': main()
