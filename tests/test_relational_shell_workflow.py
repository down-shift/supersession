"""Exercise real Bash control flow with uv replaced; never load model weights."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REV = 'shell_test'


@pytest.fixture
def shell_workspace(tmp_path):
    (tmp_path / 'scripts').mkdir()
    for name in ('prepare_and_run_relational_runtimefix.sh', 'run_relational_confirmations_overnight.sh'):
        shutil.copyfile(ROOT / 'scripts' / name, tmp_path / 'scripts' / name)
    helper = tmp_path / 'src/cross_model'
    helper.mkdir(parents=True)
    shutil.copyfile(ROOT / 'src/cross_model/runtime_provenance.py', helper / 'runtime_provenance.py')
    source = tmp_path / 'outputs/cross_model_relational_v2/factorial_relation_counterbalanced_20261003'
    expected = {'python': '3.12.13', 'packages': {'torch': 'saved-version'}}
    for model in ('qwen3_8b', 'gemma3_4b'):
        run = source / model
        run.mkdir(parents=True)
        for name in ('candidates.json', 'development.jsonl', 'gate.jsonl',
                     'development_scores.jsonl', 'gate_scores.jsonl',
                     'development.jsonl.provenance.json', 'gate.jsonl.provenance.json',
                     'development_scores.jsonl.provenance.json', 'gate_scores.jsonl.provenance.json'):
            (run / name).write_text(json.dumps({'provenance': expected}))
        v1 = tmp_path / f'outputs/{model}_review2'
        v1.mkdir(parents=True)
        (v1 / 'confirmatory_scores.jsonl.provenance.json').write_text('{}')
        config = tmp_path / 'configs/cross_model_relational_v2_geometryfix_exclusions'
        config.mkdir(parents=True, exist_ok=True)
        (config / f'{model}.yaml').write_text('fixture')
    for model in ('qwen3_8b', 'gemma3_4b', 'phi4_mini'):
        v1 = tmp_path / f'outputs/{model}_review2'
        v1.mkdir(parents=True, exist_ok=True)
        for artifact in ('confirmatory_scores.jsonl', 'confirmatory_scores.jsonl.provenance.json', 'candidates.json'):
            (v1 / artifact).write_text('{}')
        (config / f'{model}.yaml').write_text('fixture')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    uv = bin_dir / 'uv'
    uv.write_text(f'#!{sys.executable}\n' + r'''
import json, os, pathlib, sys
args = sys.argv[1:]
body = sys.stdin.read() if args[-1:] == ['-'] else ''
def option(name):
    return args[args.index(name) + 1] if name in args else None
config = option('--config') or os.environ.get('MODEL_CONFIG', '')
model = pathlib.Path(config).stem if config else None
command = args[args.index('scripts.robustness_v2') + 1] if 'scripts.robustness_v2' in args else None
checkpoint = 'from src.cross_model.runtime import load_pinned_model' in body
runtime = 'runtime_mismatches' in body
record = {'args': args, 'command': command, 'model': model, 'checkpoint': checkpoint,
          'runtime': runtime, 'env': os.environ.get('UV_PROJECT_ENVIRONMENT'),
          'python': option('--python')}
with open('calls.jsonl', 'a') as log:
    log.write(json.dumps(record) + '\n')
if runtime and os.environ.get('FAIL_RUNTIME') == model:
    sys.exit(31)
if 'validate_saved_score_runtime' in body:
    sys.path.insert(0, os.getcwd())
    from src.cross_model.runtime_provenance import validate_saved_score_runtime
    validate_saved_score_runtime(os.environ['SCORES'],
                                 {'python': '3.12.13', 'packages': {'torch': 'saved-version'}})
if checkpoint and os.environ.get('FAIL_CHECKPOINT') == model:
    sys.exit(32)
if command == 'score' and os.environ.get('FAIL_SCORE') == model:
    sys.exit(33)
if 'report = read_sealed' in body:
    assert os.environ.get('SCORES'), 'existing-analysis validation requires SCORES'
if command:
    output = pathlib.Path(option('--output'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{}')
    if command == 'migrate':
        pathlib.Path(str(output) + '.scores.jsonl').write_text('')
        pathlib.Path(str(output) + '.scores.jsonl.provenance.json').write_text(json.dumps({
            'provenance': {'python': '3.13.5'},
            'original_inference_provenance': {'python': '3.12.13', 'packages': {}}}))
''')
    uv.chmod(0o755)
    env = {**os.environ, 'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
           'RELATIONAL_REVISION': REV}
    return tmp_path, env


def launch(workspace, script='prepare_and_run_relational_runtimefix.sh', **settings):
    root, env = workspace
    result = subprocess.run(['bash', str(root / 'scripts' / script)], cwd=root,
                            env={**env, **settings}, capture_output=True, text=True, timeout=30)
    path = root / 'calls.jsonl'
    calls = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return result, calls


def test_preparation_hands_off_to_both_confirmations_with_original_runtime(shell_workspace):
    result, calls = launch(shell_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    audit = next(c for c in calls if c['command'] == 'audit')
    assert '--design-only' in audit['args']
    scores = [c for c in calls if c['command'] == 'score']
    assert [c['model'] for c in scores] == ['qwen3_8b', 'gemma3_4b']
    for model in ('qwen3_8b', 'gemma3_4b'):
        model_calls = [c for c in calls if c['model'] == model and c['python'] and c['args'][0] == 'run']
        assert all(c['python'] == '3.12.13' for c in model_calls)
        assert len({c['env'] for c in model_calls}) == 1
    checkpoints = [i for i, c in enumerate(calls) if c['checkpoint']]
    assert len(checkpoints) == 2
    assert max(checkpoints) < next(i for i, c in enumerate(calls) if c['command'] == 'score')


def test_qwen_score_failure_stops_its_analysis_but_runs_gemma(shell_workspace):
    result, calls = launch(shell_workspace, FAIL_SCORE='qwen3_8b')
    assert result.returncode != 0
    assert [c['model'] for c in calls if c['command'] == 'score'] == ['qwen3_8b', 'gemma3_4b']
    analyses = [c['model'] for c in calls if c['command'] == 'analyze' and '--stage' in c['args']
                and c['args'][c['args'].index('--stage') + 1] == 'confirmatory']
    assert analyses == ['gemma3_4b']


def test_gemma_checkpoint_failure_blocks_both_scoring_runs(shell_workspace):
    result, calls = launch(shell_workspace, FAIL_CHECKPOINT='gemma3_4b')
    assert result.returncode != 0
    assert not any(c['command'] == 'score' for c in calls)


def test_preparation_runtime_failure_blocks_migration_and_scoring(shell_workspace):
    result, calls = launch(shell_workspace, FAIL_RUNTIME='qwen3_8b')
    assert result.returncode != 0
    assert not any(c['command'] in ('migrate', 'score') for c in calls)


def test_existing_output_directory_is_preserved_without_running_uv(shell_workspace):
    root, _ = shell_workspace
    base = root / f'outputs/cross_model_relational_v2/{REV}'
    base.mkdir()
    marker = base / 'preserve.txt'
    marker.write_text('original')
    result, calls = launch(shell_workspace)
    assert result.returncode != 0 and not calls
    assert marker.read_text() == 'original'


def runner_inputs(workspace, state):
    root, _ = workspace
    base = root / f'outputs/cross_model_relational_v2/{REV}'
    for model in ('qwen3_8b', 'gemma3_4b'):
        run = base / model
        migrated = run / 'migrated'
        migrated.mkdir(parents=True)
        for path in (run / 'candidates.json', run / 'confirmatory.jsonl',
                     migrated / 'gate_report.json', migrated / 'preflight.json'):
            path.write_text('{}')
        runtime = {'python': '3.12.13', 'packages': {'torch': 'saved-version'}}
        (migrated / 'gate.jsonl.scores.jsonl.provenance.json').write_text(json.dumps({'provenance': runtime}))
        (run / 'confirmatory_scores.jsonl').write_text('preserve')
        if state == 'complete':
            (run / 'confirmatory_scores.jsonl.provenance.json').write_text(json.dumps({'provenance': runtime}))
            (run / 'confirmatory_analysis.json').write_text('{}')
        else:
            if state == 'mismatch' and model == 'gemma3_4b':
                runtime = {**runtime, 'python': '3.13.5'}
            (run / 'confirmatory_scores.jsonl.run.json').write_text(json.dumps({'config': runtime}))
    return base


def test_runner_reuses_bundles_and_resumes_only_matching_checkpoints(shell_workspace):
    runner_inputs(shell_workspace, 'resume')
    result, calls = launch(shell_workspace, 'run_relational_confirmations_overnight.sh')
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(c['command'] == 'generate' for c in calls)
    scores = [c for c in calls if c['command'] == 'score']
    assert len(scores) == 2 and all('--resume' in c['args'] for c in scores)


def test_runner_validates_completed_outputs_without_rescoring(shell_workspace):
    base = runner_inputs(shell_workspace, 'complete')
    result, calls = launch(shell_workspace, 'run_relational_confirmations_overnight.sh')
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(c['command'] in ('generate', 'score', 'analyze') for c in calls)
    assert (base / 'qwen3_8b/confirmatory_scores.jsonl').read_text() == 'preserve'


def test_mismatched_gemma_checkpoint_blocks_both_runs_before_scoring(shell_workspace):
    base = runner_inputs(shell_workspace, 'mismatch')
    result, calls = launch(shell_workspace, 'run_relational_confirmations_overnight.sh')
    assert result.returncode != 0
    assert not any(c['command'] == 'score' for c in calls)
    assert (base / 'gemma3_4b/confirmatory_scores.jsonl').read_text() == 'preserve'
