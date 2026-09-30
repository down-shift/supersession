#!/usr/bin/env python3
"""Generate disposable prompt development, fresh frozen gate, or gated final data."""
import argparse
import json
from pathlib import Path
from src.data.io import read_jsonl, write_jsonl, sha256_file
from src.data.status_prompt_gate import (VARIANTS, DEFAULT_SEEDS, generate_prompt_stage,
                                       checked_artifact, template_hash)
from src.utils import provenance, save_json

# Behavioral artifacts only; never read a mechanistic partition.
LEGACY_INPUTS = ('outputs/supersession/status.jsonl', 'outputs/supersession/status_2x2.jsonl',
                 'outputs/supersession/status_2x2_competence_gate.jsonl')


def generate_stage(stage, values_path, token_path, output, n=None, seed=None,
                   selection_path=None, frozen_gate_path=None, exclude_paths=()):
    out = Path(output)
    if out.exists() or Path(str(out)+'.provenance.json').exists():
        raise FileExistsError(f'{out} exists; choose a fresh output path')
    n = n if n is not None else (96 if stage == 'confirmatory' else 24)
    seed = seed if seed is not None else DEFAULT_SEEDS[stage]
    variants = VARIANTS
    binding = {}
    excluded_paths = [p for p in LEGACY_INPUTS if Path(p).exists()] + list(exclude_paths)
    if stage == 'frozen_gate':
        if not selection_path:
            raise ValueError('frozen gate requires an explicit development --selection artifact')
        selected = checked_artifact(selection_path, 'development')
        variants = (selected['selected_variant'],)
        excluded_paths.append(selected['dataset_path'])
        binding = {'selection_path':str(Path(selection_path).resolve()),
                   'selection_sha256':sha256_file(selection_path)}
    elif stage == 'confirmatory':
        if not frozen_gate_path:
            raise ValueError('confirmatory generation requires an explicit passing --frozen-gate artifact')
        gate = checked_artifact(frozen_gate_path, 'frozen_gate')
        variants = (gate['selected_variant'],)
        generation = json.loads(Path(gate['dataset_path']+'.provenance.json').read_text())
        selection = checked_artifact(generation['selection_path'], 'development')
        excluded_paths += [gate['dataset_path'], selection['dataset_path']]
        binding = {'frozen_gate_path':str(Path(frozen_gate_path).resolve()),
                   'frozen_gate_sha256':sha256_file(frozen_gate_path)}
    elif stage != 'development':
        raise ValueError('unsupported prompt stage')
    if stage != 'development':
        artifact = selected if stage == 'frozen_gate' else gate
        scoring = json.loads(Path(artifact['scores_path']+'.provenance.json').read_text())
        if sha256_file(token_path) != scoring['token_map_sha256']:
            raise ValueError('token map differs from selected/gated scoring')
    proposals = json.loads(Path(values_path).read_text())
    token_doc = json.loads(Path(token_path).read_text())
    values = [v for v in proposals if v in token_doc['token_ids']]
    excluded_paths = sorted(set(str(Path(p).resolve()) for p in excluded_paths))
    excluded = [r for p in excluded_paths for r in read_jsonl(p)]
    rows = generate_prompt_stage(stage, n, values, seed, variants, excluded, frozen_gate_path)
    write_jsonl(rows, out)
    save_json({**provenance({'seed':seed},out), 'stage':stage, 'n_histories':n,
               'n_records':len(rows), 'prompt_variants':list(variants), 'template_sha256':template_hash(),
               'values_sha256':sha256_file(values_path), 'token_map_sha256':sha256_file(token_path),
               'selected_values':values, 'rejected_proposals':[v for v in proposals if v not in values],
               'excluded_datasets':[{'path':p,'sha256':sha256_file(p)} for p in excluded_paths],
               'update_order':'both xz and zx within every history/status/query/variant', **binding},
              str(out)+'.provenance.json')
    print(f'wrote {len(rows)} {stage} records for {n} fresh histories to {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage',choices=('development','frozen_gate','confirmatory'),required=True)
    p.add_argument('--values',required=True); p.add_argument('--token-ids',required=True)
    p.add_argument('--output',required=True); p.add_argument('--n',type=int); p.add_argument('--seed',type=int)
    p.add_argument('--selection'); p.add_argument('--frozen-gate')
    p.add_argument('--exclude-dataset',action='append',default=[],help='additional old/development histories to exclude')
    a = p.parse_args()
    generate_stage(a.stage,a.values,a.token_ids,a.output,a.n,a.seed,a.selection,a.frozen_gate,a.exclude_dataset)


if __name__ == '__main__':
    main()
