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
    config_path = 'configs/cross_model_relational_v2_geometryfix_exclusions/qwen3_8b.yaml'
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


def test_same_prompt_opposite_orientation_maps_semantic_slots_by_physical_span():
    canonical = {
        'initial_x': {'char_start': 10, 'char_end': 15, 'text': 'coral'},
        'initial_z': {'char_start': 20, 'char_end': 25, 'text': 'pearl'},
        'queried_entity': {'char_start': 0, 'char_end': 4, 'text': 'Nora'},
    }
    reversed_labels = {
        'initial_x': {'char_start': 20, 'char_end': 25, 'text': 'pearl'},
        'initial_z': {'char_start': 10, 'char_end': 15, 'text': 'coral'},
        'queried_entity': {'char_start': 0, 'char_end': 4, 'text': 'Nora'},
    }
    assert cli.map_span_fields(reversed_labels, canonical) == {
        'initial_x': 'initial_z', 'initial_z': 'initial_x', 'queried_entity': 'queried_entity'}


class CharacterOffsetTokenizer:
    chat_template = 'fixture-template'

    class Backend:
        @staticmethod
        def to_str():
            return 'character-offset-fixture'

    backend_tokenizer = Backend()

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {'input_ids': [ord(ch) for ch in text]}
        if return_offsets_mapping:
            result['offset_mapping'] = [(i, i + 1) for i in range(len(text))]
        return result

    @staticmethod
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return '<user>' + messages[0]['content'] + '<assistant>'


def _opposite_orientation_members():
    history = data._history_definitions('validation')[0]
    other = copy.deepcopy(history)
    def swap_suffix(key):
        if key.endswith('_x'):
            return key[:-1] + 'z'
        if key.endswith('_z'):
            return key[:-1] + 'x'
        return key
    other['history_id'] += ':opposite'
    other['variables'] = history['variables'][::-1]
    other['orientation'] = 1 - history['orientation']
    other['allocation_cell']['orientation'] = other['orientation']
    other['matching_values'] = {swap_suffix(k): v for k, v in history['matching_values'].items()}
    other['replacement_values'] = {swap_suffix(k): v for k, v in history['replacement_values'].items()}
    first = data._member(history, 'superseded', 0, 0, 'x', 'x', 0)
    second = data._member(other, 'superseded', 1, 1, 'z', 'z', 0)
    return first, second


def test_opposite_orientation_identical_prompts_record_and_validate_geometry(monkeypatch):
    row, opposite = _opposite_orientation_members()
    tok = CharacterOffsetTokenizer()
    assert row['orientation'] != opposite['orientation']
    assert data.render(row, tok, True) == data.render(opposite, tok, True)
    monkeypatch.setattr(cli, 'validate_tokens', lambda *a, **k: {})
    geometry = cli.token_audit(tok, [row, opposite])['token_span_geometry']
    assert len(geometry['by_prompt_sha256']) == 1
    assert geometry['example_to_span_key'][row['example_id']]['initial_x'] == 'initial_x'
    assert geometry['example_to_span_key'][opposite['example_id']]['initial_x'] == 'initial_z'
    design.verify_geometry_rows([row, opposite], geometry)


def test_geometry_validation_rejects_incorrect_semantic_text(monkeypatch):
    row, _ = _opposite_orientation_members()
    tok = CharacterOffsetTokenizer()
    monkeypatch.setattr(cli, 'validate_tokens', lambda *a, **k: {})
    geometry = cli.token_audit(tok, [row])['token_span_geometry']
    wrong = copy.deepcopy(row)
    wrong['semantic_values']['initial_x'] = 'WRONG'
    with pytest.raises(ValueError, match='geometry span texts differ'):
        design.verify_geometry_rows([wrong], geometry)


def test_same_prompt_span_mapping_rejects_nonidentical_geometry():
    span = {'char_start': 10, 'char_end': 15, 'text': 'coral'}
    with pytest.raises(ValueError, match='inconsistent semantic span geometry'):
        cli.map_span_fields({'initial_x': span}, {'initial_x': {**span, 'char_start': 11}})


def test_failed_gemma_histories_are_recorded_in_new_exclusion_revision():
    import json

    ledger = json.loads(data.EXCLUSION_LEDGER.read_text())
    record = ledger['resolved_confirmation_exclusion']
    assert not ledger['confirmation_migration_hold']
    assert ledger['design_revision'] == data.DESIGN_REVISION
    assert record['model'] == 'gemma3_4b'
    assert record['stage'] == 'confirmatory'
    assert record['histories'] == 96
    assert record['members'] == 18_432
    assert record['score_file_present'] is False
    assert record['score_stage_status'] == 'geometry span texts differ from semantic values/entities'
    assert record['physical_signature_count'] == 96
    assert len(record['physical_signatures']) == len(set(record['physical_signatures'])) == 96
