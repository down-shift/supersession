"""Hash-bound stage lineage, frozen gate recomputation, and run authorization."""
import json
import logging
from pathlib import Path

from src.cross_model.protocol import (COUNTS, CONTRACT, check_manifest, disjoint, evaluate,
                                      manifest, read_sealed, sealed, validate_dataset, write_new)
from src.data.io import read_jsonl, sha256_file
from src.cross_model.progress import progress

logger = logging.getLogger(__name__)


def local_artifact_path(path):
    """Resolve a saved absolute path after a run directory was moved between hosts."""
    saved = Path(path)
    if saved.exists():
        return saved
    parts = saved.parts
    if 'outputs' in parts:
        local = Path(*parts[parts.index('outputs'):])
        if local.exists():
            local = local.resolve()
            logger.info("Resolved relocated artifact path %s -> %s", saved, local)
            return local
    raise FileNotFoundError(f"artifact path is unavailable locally: {saved}")


def dataset_info(path, config, config_path, candidate_path, stage=None):
    logger.info("Validating dataset lineage and hashes: %s", path)
    path = local_artifact_path(path)
    info = read_sealed(str(path)+'.provenance.json')
    rows = read_jsonl(path)
    if stage and info['stage'] != stage: raise ValueError('wrong dataset stage')
    validate_dataset(rows, info['stage'])
    for prior in progress(info.get('prior_datasets', []), desc='Checking prior dataset hashes', unit='dataset', leave=False):
        try:
            prior_path = local_artifact_path(prior['path'])
        except FileNotFoundError:
            logger.warning("Prior history file is unavailable after relocation; recorded hash retained: %s", prior['path'])
            continue
        if sha256_file(prior_path) != prior['sha256']:
            raise ValueError('prior dataset hash changed')
    prior_paths = []
    for prior in info.get('prior_datasets', []):
        try: prior_paths.append(str(local_artifact_path(prior['path'])))
        except FileNotFoundError: pass
    if sorted(disjoint(rows, prior_paths)) != info['history_signatures']:
        raise ValueError('dataset history lineage mismatch')
    # Candidate maps and rendered histories are independent of model compute
    # dtype. This permits a numerical-stability retry while score manifests
    # remain exact about the dtype that produced likelihoods.
    check_manifest(info['provenance'], config, config_path, candidate_path, path,
                   allow_dtype_change=True)
    return rows, info


def score_info(path, dataset, config, config_path, candidate_path, stage):
    logger.info("Validating score artifact against dataset: %s", path)
    dataset = local_artifact_path(dataset)
    rows, info = dataset_info(dataset, config, config_path, candidate_path, stage)
    path = local_artifact_path(path)
    sidecar = read_sealed(str(path)+'.provenance.json')
    check_manifest(sidecar['provenance'], config, config_path, candidate_path, dataset)
    if sidecar['scores_sha256'] != sha256_file(path) or sidecar['stage'] != stage:
        raise ValueError('score hash or stage mismatch')
    scores = read_jsonl(path)
    return rows, scores, info, sidecar


def gate_report(config, config_path, candidate_path, dataset, scores, development_report):
    logger.info("Recomputing frozen competence gate from saved score rows")
    dev = read_sealed(development_report)
    if dev['stage'] != 'development': raise ValueError('gate requires development report')
    dev_dataset = local_artifact_path(dev['dataset_path'])
    dev_scores = local_artifact_path(dev['scores_path'])
    dr, ds, _, _ = score_info(dev_scores, dev_dataset, config, config_path, candidate_path, 'development')
    if dev['scores_sha256'] != sha256_file(dev_scores) or dev['dataset_sha256'] != sha256_file(dev_dataset) or evaluate(dr, ds, gate=False) != dev['evaluation']: raise ValueError('development report differs from underlying scores')
    rows, scored, info, side = score_info(scores, dataset, config, config_path, candidate_path, 'frozen_gate')
    if info.get('development_report_sha256') != sha256_file(development_report):
        raise ValueError('gate dataset has wrong development lineage')
    disjoint(rows, [str(local_artifact_path(dev['dataset_path']))])
    return {'stage': 'frozen_gate', 'evaluation': evaluate(rows, scored),
            'dataset_path': str(Path(dataset).resolve()), 'scores_path': str(Path(scores).resolve()),
            'dataset_sha256': sha256_file(dataset), 'scores_sha256': sha256_file(scores),
            'dataset_provenance_sha256': sha256_file(str(dataset)+'.provenance.json'),
            'score_provenance_sha256': sha256_file(str(scores)+'.provenance.json'),
            'development_report_path': str(Path(development_report).resolve()),
            'development_report_sha256': sha256_file(development_report),
            'contract': CONTRACT, 'provenance': manifest(config, config_path, candidate_path, dataset),
            'stop_reason': None if evaluate(rows, scored)['pass'] else 'frozen semantic competence gate failed'}


def verify_gate(path, config, config_path, candidate_path):
    gate = read_sealed(path)
    if gate.get('contract') != CONTRACT or gate.get('stage') != 'frozen_gate':
        raise ValueError('gate contract mismatch')
    development_path = local_artifact_path(gate['development_report_path'])
    if gate['development_report_sha256'] != sha256_file(development_path):
        raise ValueError('development report changed')
    dataset_path = local_artifact_path(gate['dataset_path'])
    scores_path = local_artifact_path(gate['scores_path'])
    recomputed = gate_report(config, config_path, candidate_path, dataset_path,
                             scores_path, development_path)
    check_manifest(gate['provenance'], config, config_path, candidate_path, dataset_path)
    recomputed['provenance'] = gate['provenance']
    comparable_gate = {**gate, 'dataset_path':str(dataset_path.resolve()),
                       'scores_path':str(scores_path.resolve()),
                       'development_report_path':str(development_path.resolve())}
    if comparable_gate != recomputed or not recomputed['evaluation']['pass']:
        raise ValueError('failed or inconsistent frozen gate')
    return gate


def verify_confirmation(dataset, config, config_path, candidate_path):
    rows, info = dataset_info(dataset, config, config_path, candidate_path, 'confirmatory')
    gate = verify_gate(info['gate_path'], config, config_path, candidate_path)
    if sha256_file(info['gate_path']) != info['gate_sha256']:
        raise ValueError('confirmatory gate binding mismatch')
    dev = read_sealed(local_artifact_path(gate['development_report_path']))
    disjoint(rows, [str(local_artifact_path(gate['dataset_path'])),
                    str(local_artifact_path(dev['dataset_path']))])
    report = read_sealed(info['preflight_path'])
    if report['gate_sha256'] != info['gate_sha256'] or sha256_file(info['preflight_path']) != info['preflight_sha256']:
        raise ValueError('preflight report changed or belongs to another gate')
    check_manifest(report['provenance'], config, config_path, candidate_path,
                   local_artifact_path(gate['dataset_path']))
    return rows, info
