#!/usr/bin/env python3
"""Descriptive version relevance with history-bootstrap CIs and depth competence."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data.io import read_jsonl, sha256_file
from src.data.version_chain import template_hash
from src.analysis.version_chain import competence, relevance, summarize
from src.utils import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--scores',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--bootstrap-draws',type=int,default=2000)
    p.add_argument('--bootstrap-seed',type=int,default=20261030)
    a = p.parse_args()
    out = Path(a.output_dir)
    if out.exists():
        raise FileExistsError('preserve analyses; choose a fresh output directory')
    ds, ss = read_jsonl(a.dataset),read_jsonl(a.scores)
    prov_path = a.scores+'.provenance.json'
    prov = json.loads(Path(prov_path).read_text())
    if prov['dataset_sha256'] != sha256_file(a.dataset) or prov['template_sha256'] != template_hash():
        raise ValueError('score provenance differs from dataset/template')
    effects, rows = relevance(ds,ss)
    summary, contrasts = summarize(rows,a.bootstrap_draws,a.bootstrap_seed)
    comp = competence(ds,ss)
    out.mkdir(parents=True)
    for name,data in [('edit_effects',effects),('R_by_history',rows),('R_summary',summary),
                      ('paired_contrasts',contrasts),('competence_cells',comp['cells']),('competence_by_depth',comp['depths'])]:
        pd.DataFrame(data).to_csv(out/(name+'.csv'),index=False)
    fig, ax = plt.subplots(figsize=(7,4))
    for depth in sorted({r['depth'] for r in summary}):
        rs = sorted([r for r in summary if r['axis']=='symmetric' and r['depth']==depth],key=lambda r:r['age_from_current'])
        y = [r['mean'] for r in rs]
        ax.errorbar([r['age_from_current'] for r in rs],y,
                    yerr=[[r['mean']-min(r['ci_low'],r['mean']) for r in rs],
                          [max(r['ci_high'],r['mean'])-r['mean'] for r in rs]],marker='o',label=f'depth {depth}')
    ax.axhline(0,color='gray',linewidth=.5)
    ax.set(xlabel='Age from current (0 = current)',ylabel='Query-specific causal relevance R')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'R_by_age.png',dpi=160); plt.close(fig)
    doc = {'stage':ds[0]['stage'],'analysis_label':'exploratory pilot' if ds[0]['stage']=='pilot' else 'primary full chain experiment',
        'dataset_path':str(Path(a.dataset).resolve()),'scores_path':str(Path(a.scores).resolve()),
        'dataset_sha256':sha256_file(a.dataset),'scores_sha256':sha256_file(a.scores),
        'scores_provenance_sha256':sha256_file(prov_path),'template_sha256':template_hash(),
        'config_sha256':prov['config_sha256'],'token_map_sha256':prov['token_map_sha256'],
        'model_revision':prov['model_revision'],'tokenizer_revision':prov['tokenizer_revision'],
        'competence':comp,'bootstrap_unit':'history_id','bootstrap_draws':a.bootstrap_draws,
        'bootstrap_seed':a.bootstrap_seed,
        'analysis_code_sha256':hashlib.sha256(b''.join(Path(p).read_bytes() for p in
            (__file__,'src/analysis/version_chain.py','src/analysis/metrics.py'))).hexdigest(),
        'definitions':{
            'E':'(replacement_logit_edit - source_logit_edit) - (replacement_logit_base - source_logit_base)',
            'R_i':'E(edit variable v version i | query v) - E(same edit | query other variable)',
            'symmetric':'within-history mean of x and z R_i','age_from_current':'depth - version_index'},
        'hypotheses':{'discrete':'sharp current gap; obsolete versions roughly similar',
            'recency':'positive adjacent newer-minus-older relevance contrasts',
            'hybrid':'sharp current gap plus residual obsolete relevance gradient'},
        'caveats':['No clean irrelevant-style baseline was added; other-variable query is the relevance subtraction, not an unassigned mention control.',
            'Pointwise bootstrap intervals are descriptive; overlapping intervals do not establish equivalence.',
            'Histories are independent across depths; depth contrasts are not paired across depths.',
            'All audited trials retained; depth eligibility uses competence only, including edited prompts.']}
    save_json(doc,out/'analysis.json')
    print(json.dumps({'stage':ds[0]['stage'],'qualified_depths':comp['qualified_depths'],'competence':comp['depths']}))


if __name__ == '__main__':
    main()
