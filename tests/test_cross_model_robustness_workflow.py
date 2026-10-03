"""Synthetic-score stage-lineage tests; these never load pretrained weights."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import robustness_v2 as cli
from src.cross_model import robustness_protocol as design
from src.cross_model import robustness_v2 as data
from src.cross_model.protocol import digest, read_sealed, sealed, write_new
from src.cross_model.workflow import dataset_info, gate_report, verify_gate
from src.data.io import sha256_file, write_jsonl
from src.utils import load_config


@pytest.fixture
def lineage(tmp_path, monkeypatch):
    # Geometry itself is exercised with exact toy offsets in the tokenizer tests;
    # this fixture tests sealed stage bindings and recomputed eligibility.
    monkeypatch.setattr(design, 'verify_dataset_info', lambda rows, info: None)
    config_path = 'configs/cross_model_relational_v2/qwen3_8b.yaml'
    config = load_config(config_path)
    candidate = tmp_path/'candidates.json'
    write_new(candidate, sealed({'fixture': 'synthetic candidate map; no weights or inference'}))

    def stage_bundle(stage, prior=(), development_report=None, fail=False):
        _, rows = data.generate(stage, prior)
        dataset, scores_path = tmp_path/f'{stage}.jsonl', tmp_path/f'{stage}_scores.jsonl'
        write_jsonl(rows, dataset)
        info = {'stage': stage, 'history_signatures': data.disjoint(rows, prior),
                'prior_datasets': [{'path': str(p), 'sha256': sha256_file(p)} for p in prior],
                'provenance': design.manifest(config, config_path, candidate, dataset)}
        if development_report:
            info['development_report_sha256'] = sha256_file(development_report)
        write_new(str(dataset)+'.provenance.json', sealed(info))
        scores = []
        for r in rows:
            bad = fail and r['condition'] == 'entity_mention'
            masses = {v: -1. if bad else (-.1 if v == r['answer'] else -10.) for v in data.VALUES}
            scores.append({**copy.deepcopy(r), 'prompt': data.render(r, None, False),
                           'score_kind': 'competence_only', 'semantic_log_mass': masses,
                           'semantic_rank': 8 if bad else 1, 'semantic_accuracy': 0 if bad else 1})
        write_jsonl(scores, scores_path)
        write_new(str(scores_path)+'.provenance.json', sealed({'stage': stage,
            'scores_sha256': sha256_file(scores_path),
            'provenance': design.manifest(config, config_path, candidate, dataset)}))
        return dataset, scores_path, rows, scores
    # A failing descriptive development result cannot veto a fresh gate.
    dd, ds, dr, scored = stage_bundle('development', fail=True)
    dev_report = tmp_path/'development_report.json'
    write_new(dev_report, sealed({'stage': 'development', 'dataset_path': str(dd), 'scores_path': str(ds),
        'dataset_sha256': sha256_file(dd), 'scores_sha256': sha256_file(ds),
        'evaluation': design.evaluate(dr, scored, gate=False)}))
    return SimpleNamespace(config=config, config_path=config_path, candidate=candidate,
                           dev_dataset=dd, dev_report=dev_report, make_stage=stage_bundle, root=tmp_path)


def test_frozen_gate_recomputes_after_descriptive_development_failure(lineage):
    x = lineage
    dataset, scores, _, _ = x.make_stage('frozen_gate', [x.dev_dataset], x.dev_report)
    report = gate_report(x.config, x.config_path, x.candidate, dataset, scores, x.dev_report, design=design)
    assert report['evaluation']['pass'] and report['evaluation']['expected_cell_count'] == 384
    path = x.root/'gate_report.json'
    write_new(path, sealed(report))
    assert verify_gate(path, x.config, x.config_path, x.candidate, design=design)['evaluation']['pass']


def test_failed_gate_is_preserved_and_blocks_confirmation_before_tokenizer_loading(lineage, monkeypatch):
    x = lineage
    dataset, scores, _, _ = x.make_stage('frozen_gate', [x.dev_dataset], x.dev_report, fail=True)
    report = gate_report(x.config, x.config_path, x.candidate, dataset, scores, x.dev_report, design=design)
    path = x.root/'failed_gate.json'
    write_new(path, sealed(report))
    assert not report['evaluation']['pass']
    args = SimpleNamespace(stage='confirmatory', output=str(x.root/'confirmatory.jsonl'),
            config=x.config_path, candidates=str(x.candidate), gate=str(path), preflight=str(x.root/'preflight.json'),
            prior_dataset=[], development_report=None, local_files_only=True)
    monkeypatch.setattr(cli, 'prior_paths', lambda a: [])
    monkeypatch.setattr(cli, 'tokenizer', lambda *a: pytest.fail('tokenizer/model preparation must not follow a failed gate'))
    with pytest.raises(ValueError, match='failed or inconsistent'):
        cli.generate_stage(args, x.config, {'freeze_path': 'unused', 'freeze_sha256': 'unused'})
    assert not Path(args.output).exists()
    assert read_sealed(path)['evaluation']['pass'] is False


def test_forged_pass_flag_and_changed_original_score_hash_rejected(lineage):
    x = lineage
    dataset, scores, _, _ = x.make_stage('frozen_gate', [x.dev_dataset], x.dev_report, fail=True)
    report = gate_report(x.config, x.config_path, x.candidate, dataset, scores, x.dev_report, design=design)
    report['evaluation']['pass'] = True
    path = x.root/'forged_gate.json'
    write_new(path, sealed(report))
    with pytest.raises(ValueError, match='failed or inconsistent'):
        verify_gate(path, x.config, x.config_path, x.candidate, design=design)
    with scores.open('a') as handle:
        handle.write('{}\n')
    with pytest.raises(ValueError, match='score hash'):
        verify_gate(path, x.config, x.config_path, x.candidate, design=design)


def test_missing_recorded_prior_dataset_is_rejected_for_new_protocol(lineage):
    x = lineage
    dataset, _, _, _ = x.make_stage('frozen_gate', [x.dev_dataset], x.dev_report)
    x.dev_dataset.unlink()
    with pytest.raises(ValueError, match='requires every recorded prior'):
        dataset_info(dataset, x.config, x.config_path, x.candidate, design=design)


def test_token_geometry_tamper_rejected_before_scoring(tmp_path, monkeypatch):
    geometry = tmp_path/'geometry.json'
    write_new(geometry, sealed({'by_prompt_sha256': {}, 'example_to_prompt_sha256': {}}))
    frozen = tmp_path/'freeze.json'
    write_new(frozen, sealed({}))
    monkeypatch.setattr(design, 'verify_freeze', lambda path: {})
    info = {'freeze_path': str(frozen), 'freeze_sha256': sha256_file(frozen),
            'geometry_path': str(geometry), 'geometry_sha256': sha256_file(geometry)}
    geometry.write_text('{}')
    with pytest.raises(ValueError, match='token geometry changed'):
        design.verify_dataset_info([], info)
