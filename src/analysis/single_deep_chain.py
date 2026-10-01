"""Single-chain competence, paired relevance, and stable-binding controls."""
import csv
import math
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from src.data.single_deep_chain import audit
from src.data.io import sha256_file
from src.analysis.version_chain import identity_transfer, bootstrap, summarize

MIN_ACCURACY = 0.98


def validate_scores(dataset,scores):
    audit(dataset)
    expected = {r['example_id']:r for r in dataset}
    if len(scores)!=len(expected) or {s['example_id'] for s in scores}!=set(expected):
        raise ValueError('scores do not exactly cover all matched cells')
    for s in scores:
        if any(s.get(k)!=v for k,v in expected[s['example_id']].items()):
            raise ValueError('score metadata differs from dataset')
        if set(s['candidate_logits'])!=set(s['candidate_values']) or not all(math.isfinite(float(v)) for v in s['candidate_logits'].values()):
            raise ValueError('invalid raw candidate logits')
        if s['full_vocab_next_token_accuracy'] not in (0,1) or s['accuracy'] not in (0,1) or s['full_vocab_rank']<1 or s['candidate_rank']<1:
            raise ValueError('invalid competence diagnostics')


def relevance(dataset,scores):
    validate_scores(dataset,scores)
    pairs, cells = defaultdict(dict),defaultdict(dict)
    effects, rows, controls = [],[],[]
    for s in scores:
        pairs[s['pair_id']][s['pair_direction']] = s
    for pair in pairs.values():
        b,e = pair[0],pair[1]
        E = identity_transfer(b,e)
        key = (b['history_id'],b['depth'],b['focal_variable'],b['edit_role'],b['version_index'],b['orientation'])
        cells[key][b['query_role']] = E
        effects.append({k:b[k] for k in ('pair_id','history_id','depth','focal_variable','edit_role','version_index','age_from_current','query_role')}
                       | {'identity_transfer_E':E})
    for (hid,d,f,role,i,o),queries in cells.items():
        R = queries['focal']-queries['distractor']
        row = {'history_id':hid,'depth':d,'axis':f,'orientation':o,'version_index':i,
               'age_from_current':d-i,'R':R}
        if role=='stable_control':
            controls.append({'history_id':hid,'depth':d,'focal_variable':f,'orientation':o,
                             'R_stable':-R})
        else:
            rows.append(row)
            # Each history has one focal chain: pooled estimates balance x/z across histories.
            rows.append({**row,'axis':'pooled'})
    return effects,rows,controls


def competence(dataset,scores):
    validate_scores(dataset,scores)
    baseline = {}
    for s in scores:
        if s['pair_direction']==0:
            key = (s['history_id'],s['query_role'])
            if key in baseline and any(baseline[key][k]!=s[k] for k in
                ('candidate_logits','full_vocab_rank','candidate_rank','accuracy','full_vocab_next_token_accuracy')):
                raise ValueError('repeated baseline scores disagree')
            baseline[key] = s
    cells = defaultdict(list)
    for s in baseline.values():
        cells[(s['depth'],s['focal_variable'],s['orientation'],s['query_role'],'baseline',None,None)].append(s)
    for s in scores:
        if s['pair_direction']:
            cells[(s['depth'],s['focal_variable'],s['orientation'],s['query_role'],'edited',s['edit_role'],s['version_index'])].append(s)
    details = []
    for (d,f,o,q,member,role,index),ss in sorted(cells.items(),key=str):
        details.append({'depth':d,'focal_variable':f,'orientation':o,'query_role':q,
            'member':member,'edit_role':role,'version_index':index,'n':len(ss),
            'full_vocab_accuracy':float(np.mean([s['full_vocab_next_token_accuracy'] for s in ss])),
            'candidate_accuracy':float(np.mean([s['accuracy'] for s in ss])),
            'full_vocab_rank_mean':float(np.mean([s['full_vocab_rank'] for s in ss])),
            'full_vocab_rank_max':max(s['full_vocab_rank'] for s in ss),
            'candidate_rank_max':max(s['candidate_rank'] for s in ss)})
    summaries, qualified = [],[]
    for d in sorted({r['depth'] for r in dataset}):
        dc = [c for c in details if c['depth']==d]
        passed = len(dc)==8*(d+3) and all(c['full_vocab_accuracy']>=MIN_ACCURACY and c['candidate_accuracy']>=MIN_ACCURACY for c in dc)
        if passed:
            qualified.append(d)
        summary = {'depth':d,'competence_passed':passed}
        for role in ('focal','distractor'):
            for member in ('baseline','edited'):
                ss = ([s for s in baseline.values() if s['depth']==d and s['query_role']==role] if member=='baseline' else
                      [s for s in scores if s['depth']==d and s['query_role']==role and s['pair_direction']==1])
                prefix = f'{member}_{role}'
                summary.update({prefix+'_n':len(ss),prefix+'_full_vocab_accuracy':float(np.mean([s['full_vocab_next_token_accuracy'] for s in ss])),
                    prefix+'_candidate_accuracy':float(np.mean([s['accuracy'] for s in ss])),
                    prefix+'_rank_mean':float(np.mean([s['full_vocab_rank'] for s in ss])),
                    prefix+'_rank_max':max(s['full_vocab_rank'] for s in ss)})
                if role=='focal':
                    previous,oldest = [],[]
                    for s in ss:
                        chain = s['versions'][s['focal_variable']]; l = s['candidate_logits']
                        previous.append(l[chain[-1]]-l[chain[-2]])
                        oldest.append(l[chain[-1]]-l[chain[0]])
                    summary[prefix+'_current_minus_previous_mean'] = float(np.mean(previous))
                    summary[prefix+'_current_minus_oldest_mean'] = float(np.mean(oldest))
        summaries.append(summary)
    return {'minimum_cell_accuracy':MIN_ACCURACY,'qualified_depths':qualified,'depths':summaries,'cells':details,
        'rule':'each depth independently: baseline and every edit/query/focal/orientation cell must have full-vocabulary and candidate accuracy >= 0.98'}


def control_summary(controls,draws=2000,seed=0):
    groups = defaultdict(list)
    for r in controls:
        groups[(r['depth'],r['focal_variable'])].append(r['R_stable'])
        groups[(r['depth'],'pooled')].append(r['R_stable'])
    return [{'depth':d,'axis':axis,**bootstrap(values,draws,seed)} for (d,axis),values in sorted(groups.items())]


def depth_one_sanity(directory):
    directory = Path(directory)
    paths = [directory/name for name in ('analysis.json','R_summary.csv','paired_contrasts.csv')]
    if not all(p.is_file() for p in paths):
        return {'available':False,'scope':'sanity only; never depth eligibility or evidence about multiple obsolete versions'}
    doc = json.loads(paths[0].read_text())
    if doc['stage']!='full':
        raise ValueError('depth-one sanity must use the existing full analysis')
    with paths[1].open() as f:
        profile = [r for r in csv.DictReader(f) if int(r['depth'])==1 and r['axis']=='symmetric']
    with paths[2].open() as f:
        gap = [r for r in csv.DictReader(f) if int(r['depth'])==1 and r['axis']=='symmetric' and r['contrast']=='current_minus_previous']
    if {int(r['version_index']) for r in profile}!={0,1} or len(gap)!=1:
        raise ValueError('existing depth-one sanity summaries are incomplete')
    return {'available':True,'profile':profile,'current_minus_obsolete':gap[0],
        'expected_direction':float(gap[0]['mean'])>0,'gap_ci_excludes_zero':float(gap[0]['ci_low'])>0,
        'artifact_hashes':{p.name:sha256_file(p) for p in paths},
        'scope':'sanity only; does not answer the multiple-obsolete question'}
