"""CPU-only contract tests; no real model or tokenizer downloads."""
import copy
import json
import math
from pathlib import Path
import pytest
from src.data import stale_decision_v1 as data
from src.experiments import stale_decision_v1 as exp
from src.analysis import stale_decision_v1 as analysis


class Tokenizer:
    eos_token_id = 0
    chat_template = 'test'
    special_tokens_map = {'eos_token': '<EOS>'}
    backend_tokenizer = None

    def __init__(self):
        self.backend_tokenizer = self

    def to_str(self):
        return 'character-tokenizer-test-only'

    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text], 'offset_mapping': [(i, i+1) for i in range(len(text))]}

    def apply_chat_template(self, messages, **kwargs):
        return '[USER]\n' + messages[0]['content'] + '\n[ASSISTANT]'


@pytest.fixture(scope='module')
def rows():
    return data.generate()


@pytest.fixture
def scores(rows):
    result = []
    for r in rows:
        lp = {a: -3. for a in r['actions']}
        if r['family'] == 'superseded' and r['member'] == 0:
            lp[r['wrong_action']] = -2.
        text = r['answer']
        result.append(dict(example_id=r['example_id'], row_hash=data.digest(r), model_id='test',
                           action_logp=lp, easy_target={v: -3. for v in r['policy']},
                           easy_other={v: -3. for v in r['policy']}, raw_text=text, easy_raw_text=r['correct_state'],
                           terminated=True, easy_terminated=True,
                           generated=exp.classify(text, r['actions'], r['answer']),
                           easy_generated=exp.classify(r['correct_state'], list(r['policy']), r['correct_state']),
                           canonical_valid_mass=sum(math.exp(v) for v in lp.values()), prompt_tokens=100, history_position=10))
    return result


def test_determinism_balance_and_controls(rows):
    assert rows == data.generate()
    assert len(rows) == 2112
    assert data.validate(rows)
    for task in ('access', 'routing'):
        control = [r for r in rows if r['task'] == task and r['family'] == 'current_only']
        counts = {a: sum(r['answer'] == a for r in control) for a in control[0]['actions']}
        assert len(set(counts.values())) == 1
        for state in data.VOCABS['natural']:
            subset = [r for r in control if r['correct_state'] == state]
            assert len({r['answer'] for r in subset}) == 2
            assert len(set(sum(r['answer'] == a for r in subset) for a in subset[0]['actions'])) == 1
    assert {r['order_stratum'] for r in rows} == {'aligned', 'reversed'}
    for r in rows:
        if r['family'] != 'current_only':
            a, b = r['edit_span']
            assert r['prompt'][a:b] == r['old_values'][r['member']]


@pytest.mark.parametrize('mutation', ['ground_truth', 'swapped', 'policy', 'prompt', 'collision', 'current', 'missing', 'order', 'template'])
def test_invalid_injected(rows, mutation):
    rr = copy.deepcopy(rows)
    if mutation == 'ground_truth': rr[0]['answer'] = rr[0]['wrong_action']
    elif mutation == 'swapped': rr[0]['old_values'].reverse()
    elif mutation == 'policy': rr[0]['policy'][rr[0]['currents'][0]] = rr[0]['wrong_action']
    elif mutation == 'prompt': rr[1]['prompt'] += 'extra'
    elif mutation == 'collision': rr[0]['actions'][1] = rr[0]['actions'][0]
    elif mutation == 'current': rr[1]['currents'][0] = rr[1]['old_values'][0]
    elif mutation == 'missing': rr.pop()
    elif mutation == 'order': rr[0]['order_stratum'] = 'bad'
    elif mutation == 'template': rr[0]['template'] = 'heldout'
    with pytest.raises(ValueError): data.validate(rr)


def test_splits_and_orientation_leakage(rows):
    gate = data.generate('frozen_gate', 2); confirmation = data.generate('confirmation', 8)
    data.verify_no_overlap(rows, gate, confirmation)
    assert {r['template'] for r in confirmation} == {'heldout'}
    with pytest.raises(ValueError, match='leakage'): data.verify_no_overlap(rows, copy.deepcopy(rows))
    r = copy.deepcopy(rows[0]); before = data.signature(r)
    r['entities'].reverse(); r['currents'].reverse(); r['old_values'].reverse()
    assert data.signature(r) == before


def test_token_audit_all_occurrences(rows):
    audit = exp.token_audit(rows, Tokenizer())
    assert len(audit['prompts']) == len(rows)
    assert len(audit['pairs']) == 192 * 5 * 3
    for r in rows:
        assert audit['prompts'][r['example_id']]['downstream']['events'][r['answer']][0]['ids'][-1] == 0


def test_candidate_collisions_and_prefix():
    tok = Tokenizer()
    with pytest.raises(ValueError): exp.action_events(tok, 'Answer:', ['X', 'X'])
    # EOS makes lexical-prefix labels disjoint full events.
    assert exp.action_events(tok, 'Answer:', ['X', 'XY'])
    class Collision(Tokenizer):
        def __call__(self, text, **kwargs):
            return super().__call__(text.replace('Y', 'X'), **kwargs)
    with pytest.raises(ValueError): exp.action_events(Collision(), 'Answer:', ['X', 'Y'])
    class Retokenize(Tokenizer):
        def __call__(self, text, **kwargs):
            return super().__call__('changed' if ' X' in text else text, **kwargs)
    with pytest.raises(ValueError): exp.action_events(Retokenize(), 'Answer:', ['X', 'Y'])


def test_strict_vs_relaxed():
    assert exp.classify(' **APPROVE** ', ['APPROVE', 'DENY'], 'APPROVE') == {'classification': 'invalid', 'relaxed_action': 'APPROVE'}
    assert exp.classify('APPROVE because...', ['APPROVE', 'DENY'], 'APPROVE')['classification'] == 'invalid'
    assert exp.classify('DENY', ['APPROVE', 'DENY'], 'APPROVE')['classification'] == 'wrong'
    assert exp.classify('APPROVE', ['APPROVE', 'DENY'], 'APPROVE', False)['classification'] == 'invalid'


def test_scores_missing_nonfinite_swapped_models(rows, scores):
    exp.checked_scores(rows, scores)
    with pytest.raises(ValueError): exp.checked_scores(rows, scores[:-1])
    for field in ('action_logp', 'easy_target'):
        ss = copy.deepcopy(scores); ss[0][field][next(iter(ss[0][field]))] = float('nan')
        with pytest.raises(ValueError): exp.checked_scores(rows, ss)
    ss = copy.deepcopy(scores); ss[0]['generated']['classification'] = 'wrong'
    with pytest.raises(ValueError): exp.checked_scores(rows, ss)
    ss = copy.deepcopy(scores); ss[0]['model_id'] = 'other'
    with pytest.raises(ValueError): exp.checked_scores(rows, ss)


def test_signed_effects_bootstrap_and_gate(rows, scores):
    effects = analysis.cells(rows, scores)
    assert all(e['D'] == 1. for e in effects if e['family'] == 'superseded')
    assert all(e['D'] == 0. for e in effects if e['family'] != 'superseded')
    assert analysis.bootstrap([1, 2, 3]) == analysis.bootstrap([1, 2, 3])
    with pytest.raises(ValueError): analysis.bootstrap([float('inf')])
    report = analysis.analyze(rows, scores)
    assert report['primary']['access']['Delta_semantic']['mean'] == 1
    assert report['H3_secondary']['available'] is False  # Perfect generation supplies no failures.
    assert report['gates']['any_score_pass']
    assert not any(c['behavior_eligible'] for c in report['gates']['cells'].values())
    for s in scores:
        r = next(r for r in rows if r['example_id'] == s['example_id'])
        s['raw_text'] = 'nonsense'; s['generated'] = exp.classify('nonsense', r['actions'], r['answer'])
    assert not analysis.gates(rows, scores)['any_score_pass']


def test_grouped_predictive_validity_runs(rows, scores):
    for i, (r, s) in enumerate(zip(rows, scores)):
        if i % 3 == 0:
            s['raw_text'] = r['wrong_action']; s['generated'] = exp.classify(s['raw_text'], r['actions'], r['answer'])
    result = analysis.prediction(rows, scores, analysis.cells(rows, scores))
    assert result['available']
    assert math.isfinite(result['plus_R_log_loss'])


def test_interrupted_resume_and_manifest(rows, tmp_path, monkeypatch):
    rr = rows[:11]; config = json.loads(Path('configs/stale_decision_v1/protocol.json').read_text())
    monkeypatch.setattr(exp, 'runtime_info', lambda m: {'fake': True})
    calls = [0]
    def scorer(model, tok, prompt, events):
        calls[0] += 1
        if calls[0] == 5: raise RuntimeError('interrupted')
        return ({k: -3. for k in events}, None, None, None)
    monkeypatch.setattr(exp, 'score_prompt', scorer)
    monkeypatch.setattr(exp, 'generate_response', lambda *a: ('invalid diagnostic', [1, 0], True))
    output = tmp_path/'scores.jsonl'
    with pytest.raises(RuntimeError): exp.run(rr, config, output, object(), Tokenizer())
    assert len(data.read_rows(output)) == 0
    assert len(data.read_rows(str(output)+'.easy.jsonl')) == 2
    with output.open('ab') as f: f.write(b'{"truncated":')
    monkeypatch.setattr(exp, 'score_prompt', lambda m,t,p,e: ({k: -3. for k in e}, None, None, None))
    exp.run(rr, config, output, object(), Tokenizer(), resume=True)
    assert len(data.read_rows(output)) == 11
    assert len(data.read_rows(str(output)+'.easy.jsonl')) == 11
    assert list(tmp_path.glob('*.interrupted.*.bin'))
    with pytest.raises(FileExistsError): exp.run(rr, config, output, object(), Tokenizer())
    changed = copy.deepcopy(config); changed['model']['revision'] = 'a'*40
    with pytest.raises(ValueError, match='manifest'): exp.run(rr, changed, output, object(), Tokenizer(), resume=True)
    ss = data.read_rows(output); ss[0]['row_hash'] = 'bad'
    output.write_text(''.join(json.dumps(s)+'\n' for s in ss))
    with pytest.raises(ValueError): exp.run(rr, config, output, object(), Tokenizer(), resume=True)


def test_immutable_revision_and_exclusive_creation(tmp_path):
    config = json.loads(Path('configs/stale_decision_v1/protocol.json').read_text())
    exp.check_config(config)
    config['model']['revision'] = 'main'
    with pytest.raises(ValueError): exp.check_config(config)
    path = tmp_path/'artifact.json'; data.write_new(path, {'a': 1})
    with pytest.raises(FileExistsError): data.write_new(path, {'a': 2})


def test_full_answer_score_includes_eos_likelihood():
    torch = pytest.importorskip('torch')
    class TinyTok(Tokenizer):
        eos_token_id = 3
        def __call__(self, text, **kwargs):
            return {'input_ids': [0] if text == 'P' else [0, 1 if text.endswith('A') else 2]}
    class TinyModel:
        def get_input_embeddings(self):
            return torch.nn.Embedding(4, 1)
        def __call__(self, input_ids, **kwargs):
            class Output: pass
            out = Output(); out.logits = torch.zeros((1, input_ids.shape[1], 4)); return out
    tok = TinyTok(); events = exp.action_events(tok, 'P', ['A', 'B'])
    masses = exp.score_prompt(TinyModel(), tok, 'P', events)[0]
    assert masses['A'] == pytest.approx(-2 * math.log(4))
    assert sum(math.exp(v) for v in masses.values()) == pytest.approx(1/8)
