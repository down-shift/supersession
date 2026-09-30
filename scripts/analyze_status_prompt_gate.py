#!/usr/bin/env python3
"""Task competence only. Selection is explicit; frozen approval is cellwise perfect."""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.data.io import read_jsonl, sha256_file
from src.data.status_prompt_gate import VARIANTS, competence_summary, checked_artifact
from src.utils import provenance, save_json


def analyze(dataset_path, scores_path, output_dir, select_variant=None):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is nonempty; choose a fresh output directory')
    dataset, scores = read_jsonl(dataset_path), read_jsonl(scores_path)
    summary = competence_summary(dataset, scores)
    if summary['stage'] == 'development':
        if select_variant is not None and select_variant not in {r['prompt_variant'] for r in dataset}:
            raise ValueError('development requires an explicit --select-variant; selection uses competence only')
    elif select_variant is not None:
        raise ValueError('frozen gate cannot reselect a prompt variant')
    else:
        select_variant = dataset[0]['prompt_variant']
    prov_path = str(scores_path)+'.provenance.json'
    # Mandatory saved model provenance; no scoring/inference is performed here.
    scoring = json.loads(Path(prov_path).read_text())
    if scoring.get('dataset_sha256') != sha256_file(dataset_path) or scoring.get('causal_effects_computed') is not False:
        raise ValueError('scoring provenance does not describe these competence-only inputs')
    summary.update(dataset_path=str(Path(dataset_path).resolve()), scores_path=str(Path(scores_path).resolve()),
                   dataset_sha256=sha256_file(dataset_path), scores_sha256=sha256_file(scores_path),
                   selected_variant=select_variant, scoring_provenance_sha256=sha256_file(prov_path),
                   generation_provenance_sha256=sha256_file(str(dataset_path)+'.provenance.json'),
                   provenance=provenance({},dataset_path))
    # Development selection is manual; report all wording cells, not causal effects.
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summary['cells']).to_csv(out/'competence_cells.csv',index=False)
    artifact = out/(('selection.json' if select_variant else 'development_report.json')
                    if summary['stage']=='development' else 'frozen_gate.json')
    save_json(summary,artifact)
    if summary['stage']=='frozen_gate' and summary['gate_pass']:
        checked_artifact(artifact,'frozen_gate')
    print(pd.DataFrame(summary['cells']).to_string(index=False))
    if summary['stage']=='frozen_gate':
        print('Frozen competence gate PASSED' if summary['gate_pass'] else
              'PROMPT-DESIGN FAILURE: at least one status/query/update-order cell has an error')
        if not summary['gate_pass']:
            return False
    elif select_variant:
        checked_artifact(artifact,'development')
        print(f'Explicit competence-only wording selection: {select_variant}; frozen gate still required')
    else:
        print('Development report only; explicitly select wording before generating a frozen gate')
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',required=True); p.add_argument('--behavior',required=True)
    p.add_argument('--output-dir',required=True); p.add_argument('--select-variant',choices=VARIANTS)
    a = p.parse_args()
    if not analyze(a.dataset,a.behavior,a.output_dir,a.select_variant):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
