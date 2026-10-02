#!/usr/bin/env python3
"""Generate/audit relational v2 histories or reanalyze archived v1 score files."""
import argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from src.cross_model.robustness_v2 import generate, render, validate, VERSION
from src.cross_model.protocol import write_new, sealed, digest
from src.data.io import read_jsonl, sha256_file

def fast_summary(values):
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all(): raise ValueError('summary requires nonempty finite observations')
    rng=np.random.default_rng(73021); idx=rng.integers(0,len(x),size=(2000,len(x)))
    boot=x[idx].mean(axis=1)
    return {'n':int(len(x)),'mean':float(x.mean()),'ci95_history_bootstrap':[float(v) for v in np.quantile(boot,[.025,.975])]}

def cluster_summary(items):
    grouped=defaultdict(list)
    for hid,value in items: grouped[hid].append(float(value))
    if not grouped: return {'n_histories':0,'mean':None,'ci95_history_bootstrap':None}
    summary=fast_summary([np.mean(grouped[h]) for h in sorted(grouped)])
    summary['n_histories']=summary.pop('n')
    return summary


def reanalyze(dataset_path, scores_path):
    rows, scores = read_jsonl(dataset_path), read_jsonl(scores_path)
    ds = {r['example_id']:r for r in rows}; sc = {r['example_id']:r for r in scores}
    if len(ds)!=len(rows) or ds.keys()!=sc.keys(): raise ValueError('saved dataset/score IDs do not match')
    pairs=defaultdict(dict)
    for eid, row in ds.items():
        s=sc[eid]
        if any(s.get(k)!=v for k,v in row.items()): raise ValueError(f'saved score metadata mismatch at {eid}')
        if 'semantic_log_mass' not in s or 'surface_likelihoods' not in s:
            raise ValueError('required bounded-mass or surface decomposition absent; scores cannot be regenerated without inference')
        pairs[row['pair_id']][row['pair_direction']]=s
    effects=[]
    for pid,m in pairs.items():
        if set(m)!={0,1}: raise ValueError(f'incomplete pair {pid}')
        b,e=m[0],m[1]; r=ds[b['example_id']]
        lm0,lm1=b['semantic_log_mass'],e['semantic_log_mass']
        alt=max((v for v in lm0 if v!=r['answer']),key=lm0.get)
        effects.append({'pair_id':pid,'history_id':r['history_id'],'condition':r['condition'],'orientation':r['orientation'],
          'edited_variable':r['edited_variable'],'query':r['query'],'slot_order':r.get('unassigned_slot_order'),
          'base_accuracy':int(b['semantic_rank']==1),'edited_accuracy':int(e['semantic_rank']==1),
          'base_rank':b['semantic_rank'],'edited_rank':e['semantic_rank'],
          'current_mass_base':lm0[r['answer']],'current_mass_edit':lm1[e['answer']],
          'source_mass_base':lm0[r['source_value']],'source_mass_edit':lm1[r['source_value']],
          'replacement_mass_base':lm0[r['replacement_value']],'replacement_mass_edit':lm1[r['replacement_value']],
          'current_score_change':lm1[e['answer']]-lm0[r['answer']],
          'current_margin_base':lm0[r['answer']]-lm0[alt],
          'current_margin_edit':lm1[e['answer']]-lm1[alt],
          'decision_margin_change':(lm1[e['answer']]-lm1[alt])-(lm0[r['answer']]-lm0[alt]),
          'identity_transfer':(lm1[r['replacement_value']]-lm1[r['source_value']])-(lm0[r['replacement_value']]-lm0[r['source_value']]),
          'absolute_mass':{k:{'base':float(np.exp(lm0[k])),'edited':float(np.exp(lm1[k]))} for k in (r['answer'],r['source_value'],r['replacement_value'])},
          'correct_pair':int(b['semantic_rank']==1 and e['semantic_rank']==1)})
    # Existing v1 history contrast estimator preserves variable/query relevance and both orders.
    grouped=defaultdict(list)
    for e in effects: grouped[(e['history_id'],e['condition'],e['edited_variable'],e['query'],e['slot_order'])].append(e['identity_transfer'])
    history_vals=defaultdict(dict)
    for (h,c,v,q,slot), xs in grouped.items():
        if len(xs)!=1: raise ValueError('expected one paired identity-transfer row per cell')
        history_vals[h][(c,v,q,slot)]=xs[0]
    hrows=[]
    for hid,cells in sorted(history_vals.items()):
        out={'history_id':hid}
        for condition in ('live','superseded','irrelevant','irrelevant_counterbalanced'):
            label='irrelevant_counterbalanced' if condition=='irrelevant_counterbalanced' else condition
            if label=='irrelevant_counterbalanced':
                for v in ('x','z'):
                    for q in ('x','z'):
                        out[f'{label}_{v}_{q}']=float(np.mean([cells[(condition,v,q,o)] for o in ('xz','zx')]))
            else:
                for v in ('x','z'):
                    for q in ('x','z'): out[f'{label}_{v}_{q}']=cells[(condition,v,q,None)]
            out['R_'+label]=.5*sum(out[f'{label}_{v}_{v}']-out[f'{label}_{v}_'+('z' if v=='x' else 'x')] for v in ('x','z'))
        out['R_superseded_minus_R_irrelevant_counterbalanced']=out['R_superseded']-out['R_irrelevant_counterbalanced']
        out['R_live_minus_R_superseded']=out['R_live']-out['R_superseded']
        out['R_live_minus_R_irrelevant_counterbalanced']=out['R_live']-out['R_irrelevant_counterbalanced']
        for v in ('x','z'):
            out[f'R_superseded_{v}']=out[f'superseded_{v}_{v}']-out[f'superseded_{v}_'+('z' if v=='x' else 'x')]
            out[f'R_irrelevant_counterbalanced_{v}']=out[f'irrelevant_counterbalanced_{v}_{v}']-out[f'irrelevant_counterbalanced_{v}_'+('z' if v=='x' else 'x')]
        hrows.append(out)
    for e in effects:
        b,e1=pairs[e['pair_id']][0],pairs[e['pair_id']][1]
        r=ds[b['example_id']]
        e['surface_log_likelihoods']={value:{'baseline':b['surface_likelihoods'][value],
           'edited':e1['surface_likelihoods'][value]} for value in (r['source_value'],r['replacement_value'])}
    primary=[h['R_superseded_minus_R_irrelevant_counterbalanced'] for h in hrows]
    live=[h['R_live_minus_R_irrelevant_counterbalanced'] for h in hrows]
    rlive=fast_summary(live)
    normalized={'available':False,'reason':'irrelevant-corrected live-effect 95% bootstrap lower bound <= 1 nat'}
    if rlive['ci95_history_bootstrap'][0] > 1:
        rng=np.random.default_rng(73021); draws=rng.integers(0,len(live),size=(2000,len(live)))
        den=np.asarray(live)[draws].mean(axis=1)
        if np.all(den>1):
            ratios=np.asarray(primary)[draws].mean(axis=1)/den
            normalized={'available':True,'ratio_of_means':float(np.mean(primary)/np.mean(live)),
                        'ci95_history_bootstrap':[float(x) for x in np.quantile(ratios,[.025,.975])]}
    return {'protocol':'archived_cross_model_v1_reanalysis', 'dataset_sha256':sha256_file(dataset_path), 'scores_sha256':sha256_file(scores_path),
      'n_trials':len(rows),'n_pairs':len(effects),'baseline_accuracy':float(np.mean([e['base_accuracy'] for e in effects])),
      'edited_accuracy':float(np.mean([e['edited_accuracy'] for e in effects])),
      'paired_current_score_change':cluster_summary([(e['history_id'],e['current_score_change']) for e in effects]),
      'current_answer_margin':{'baseline':cluster_summary([(e['history_id'],e['current_margin_base']) for e in effects]),
          'edited':cluster_summary([(e['history_id'],e['current_margin_edit']) for e in effects])},
      'paired_decision_margin_change':cluster_summary([(e['history_id'],e['decision_margin_change']) for e in effects]),
      'candidate_ranking':{'base_rank_mean':float(np.mean([e['base_rank'] for e in effects])),'edited_rank_mean':float(np.mean([e['edited_rank'] for e in effects])),
          'rank_changed_n':sum(e['base_rank']!=e['edited_rank'] for e in effects),'both_correct_n':sum(e['correct_pair'] for e in effects)},
      'condition_accuracy':{c:{'baseline':float(np.mean([e['base_accuracy'] for e in effects if e['condition']==c])),'edited':float(np.mean([e['edited_accuracy'] for e in effects if e['condition']==c]))} for c in sorted({e['condition'] for e in effects})},
      'orientation':{str(o):fast_summary([h['R_superseded']-h['R_irrelevant_counterbalanced'] for h in hrows if next(r['orientation'] for r in rows if r['history_id']==h['history_id'])==o]) for o in (0,1)},
      'ordinary_irrelevant_by_mention_order':{o:cluster_summary([(e['history_id'],e['identity_transfer']) for e in effects if e['condition']=='irrelevant_counterbalanced' and e['slot_order']==o]) for o in ('xz','zx')},
      'history_level_R':hrows,'paired_effect_details':effects,
      'predefined_R_contrasts':{'v1_superseded_minus_counterbalanced_irrelevant':fast_summary(primary),
        'v1_live_minus_superseded':fast_summary([h['R_live_minus_R_superseded'] for h in hrows]),
        'requested_relational_contrasts':{'available':False,'reason':'entity_mention, other_attribute, early_unassigned and late_unassigned were not conditions in archived v1 data'}},
      'normalized_relative_sensitivity':normalized,
      'secondary_conditioned_both_correct':{'label':'secondary; conditioned on both baseline and edited semantic rank one',
          'n_pairs':sum(e['correct_pair'] for e in effects),'mean_identity_transfer':float(np.mean([e['identity_transfer'] for e in effects if e['correct_pair']])) if any(e['correct_pair'] for e in effects) else None},
      'limitation':'Entity-mention, other-attribute, early-unassigned and late-unassigned contrasts are unavailable in cross_model_v1 because those conditions were not collected. Per-surface bounded log likelihoods for source/replacement are retained for available edit decompositions; no unavailable condition is imputed.'}


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['generate','audit','reanalyze']); p.add_argument('--stage',choices=list(__import__('src.cross_model.robustness_v2',fromlist=['COUNTS']).COUNTS)); p.add_argument('--dataset'); p.add_argument('--scores'); p.add_argument('--output',required=True); p.add_argument('--model',default='')
    a=p.parse_args()
    if a.command in ('generate','audit'):
        if not a.stage: p.error('--stage is required')
        histories,rows=generate(a.stage); validate(histories,rows,a.stage)
        if a.command=='audit':
            samples={}
            for c in __import__('src.cross_model.robustness_v2',fromlist=['CONDITIONS']).CONDITIONS:
                for ho in (0,1):
                    for co in (0,1):
                        row=next(r for r in rows if r['condition']==c and r['historical_entity_order']==ho and r['current_entity_order']==co and r['query']=='x' and r['edited_variable']=='x' and r['pair_direction']==0)
                        samples[f'{c}:historical={ho}:current={co}']=render(row)
            print(json.dumps({'protocol':VERSION,'stage':a.stage,'histories':len(histories),'rows':len(rows),'sample_prompts':samples},indent=2)); return
        write_new(a.output,{'protocol':VERSION,'stage':a.stage,'seed':histories[0]['seed'],'histories':histories,'rows':rows,'dataset_sha256':digest(rows)})
    else:
        if not a.dataset or not a.scores: p.error('--dataset and --scores required')
        write_new(a.output,sealed(reanalyze(a.dataset,a.scores)))

if __name__=='__main__': main()
