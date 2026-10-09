"""stale_decision_v2: stale_decision_v1b's scoring and checks on the v2 data (docs/stale_decision_v2.md).
v1b's docstring follows; everything it says about v1b applies to v2.

stale_decision_v1b: documented pre-inference correction of stale_decision_v1 (docs/stale_decision_v1b.md).

Unchanged from v1: the data generator, factorial, seeds, splits, prompts, easy-retrieval events, strict
parser, checkpoint contracts, gates and the D / Delta / R_easy_local contrasts. Changed:

* action events are the label exactly as the model generates it at the start of the assistant turn
  (no leading space) followed by the designated EOS; v1 scored a space-prefixed label that the chat
  template makes essentially unreachable;
* a historical-retrieval query ("what was the INITIAL value?") is asked in the superseded construction;
* an optional KV-cache scorer, used only after `verify_cached_scorer` reproduces the reference scorer.

No v1 file is modified.
"""
import hashlib
import json
import math
import re
from pathlib import Path

from src.cross_model.scoring import score_prompt
from src.cross_model.tokens import continuations, encode, token_span
from src.data.stale_decision_v2 import build_prompt, digest, read_rows, validate, write_new
from src.experiments import stale_decision_v1 as v1

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'stale_decision_v2'
HISTORICAL_FAMILY = 'superseded'
INTROS = {'development': 'Use only CURRENT assignments. Historical records do not set current values.',
          'heldout': 'Read the dated record. Only the final assignment governs the decision.'}
HISTORICAL_INTRO = 'The record below lists initial and final assignments.'


def code_hash():
    files = ['src/data/stale_decision_v2.py', 'src/experiments/stale_decision_v1.py',
             'src/analysis/stale_decision_v1.py', 'src/experiments/stale_decision_v2.py',
             'src/analysis/stale_decision_v2.py', 'scripts/stale_decision_v2.py',
             'src/cross_model/scoring.py', 'src/cross_model/tokens.py', 'docs/stale_decision_v1b.md',
             'docs/stale_decision_v2.md']
    return digest({p: (ROOT / p).read_text() for p in files})


def check_config(config):
    """v1b runtime: unquantized float32 on one GPU (int8 and bfloat16 scoring were shown to depend on
    tokens after the scored position; docs/stale_decision_v1b.md), plus the v1 split, decoding and
    bootstrap settings."""
    if config.get('protocol') != PROTOCOL:
        raise ValueError('wrong protocol')
    if config.get('scorer') not in ('reference', 'kv_cache', 'batched'):
        raise ValueError('scorer must be reference, kv_cache or batched')
    model = config.get('model', {})
    if (model.get('dtype') != 'float32' or model.get('quantization') != 'none' or model.get('device_map') != 0 or
            model.get('require_full_gpu') is not True or model.get('attn_implementation') != 'sdpa' or
            model.get('tokenizer_id') != model.get('id')):
        raise ValueError('v1b runtime is unquantized float32 on a single GPU')
    for key in ('revision', 'tokenizer_revision'):
        if not re.fullmatch('[a-f0-9]{40}', str(model.get(key, ''))):
            raise ValueError('immutable model/tokenizer revisions required')
    if any(config.get(k) != v for k, v in [('development_replicates', 1), ('gate_replicates', 2), ('confirmation_replicates', 4)]):
        raise ValueError('frozen split counts differ')
    if config.get('max_new_tokens') != 16 or config.get('bootstrap_draws') != 2000:
        raise ValueError('frozen decoding/bootstrap settings differ')


def action_events(tokenizer, prompt, actions):
    """Full label as generated at the assistant-turn start (no leading space) plus one EOS."""
    prefix = encode(tokenizer, prompt)
    eos = tokenizer.eos_token_id
    if not prefix or not isinstance(eos, int):
        raise ValueError('nonempty prefix and designated EOS required')
    events, sequences = {}, []
    for action in actions:
        full = encode(tokenizer, prompt + action)
        if full[:len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError('candidate retokenizes prefix')
        ids = full[len(prefix):] + [eos]
        if len(ids) > 16:
            raise ValueError('canonical action cannot finish within fixed generation budget')
        if eos in ids[:-1]:
            raise ValueError('EOS inside candidate')
        if any(ids[:len(s)] == s or s[:len(ids)] == ids for s in sequences):
            raise ValueError('candidate continuation collision/prefix overlap')
        sequences.append(ids)
        events[action] = [{'text': action, 'ids': ids}]
    return events


def historical_prompt(r):
    """Superseded construction only: same record, asking for the initial value; answer = edited old value."""
    if r['family'] != HISTORICAL_FAMILY:
        raise ValueError('historical retrieval is asked only in the superseded construction')
    body, _ = build_prompt(r, 'easy')
    attribute = 'clearance' if r['task'] == 'access' else 'routing state'
    target = r['entities'][0]
    if r['template'] == 'heldout':
        old_q, new_q = f'Report the final {attribute} for {target}.', f'Report the initial {attribute} for {target}.'
    else:
        old_q, new_q = f"What is {target}'s CURRENT {attribute}?", f"What was {target}'s INITIAL {attribute}?"
    intro = INTROS[r['template']]
    if body.count(old_q) != 1 or not body.startswith(intro + '\n'):
        raise ValueError('historical query rewrite failed')
    return HISTORICAL_INTRO + body[len(intro):].replace(old_q, new_q), r['old_values'][r['member']]


def score_cached(model, tokenizer, prompt, events):
    """Same quantity as score_prompt (summed token log probabilities, logsumexp over surfaces), computing
    the prompt once and reusing its KV cache for every multi-token event."""
    import torch
    prefix = encode(tokenizer, prompt)
    device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([prefix], device=device), use_cache=True)
        cache = out.past_key_values
        first = out.logits[0, -1].float().log_softmax(-1)
        if not torch.isfinite(first).all():
            raise ValueError('nonfinite logits')
        masses, surfaces = {}, {}
        for value, members in events.items():
            likelihoods = []
            for event in members:
                ids = event['ids']
                lp = float(first[ids[0]])
                if len(ids) > 1:
                    o = model(input_ids=torch.tensor([ids[:-1]], device=device), past_key_values=cache, use_cache=True)
                    rest = o.logits[0].float().log_softmax(-1)
                    if not torch.isfinite(rest).all():
                        raise ValueError('nonfinite logits')
                    lp += float(rest[torch.arange(len(ids) - 1, device=rest.device),
                                     torch.tensor(ids[1:], device=rest.device)].sum())
                    cache.crop(len(prefix))
                likelihoods.append(lp)
            surfaces[value] = [{'text': e['text'], 'ids': e['ids'], 'log_probability': lp}
                               for e, lp in zip(members, likelihoods)]
            top = max(likelihoods)
            masses[value] = top + math.log(sum(math.exp(x - top) for x in likelihoods))
    return masses, surfaces


def score_batched(model, tokenizer, prompt, events):
    """Same quantity as score_prompt, computing every multi-token event in one right-padded batch.
    Causal attention with right padding leaves the real positions unchanged; float32 makes the result
    independent of the batch layout (checked by verify-scorer)."""
    import torch
    prefix = encode(tokenizer, prompt)
    device = model.get_input_embeddings().weight.device
    flat = [(value, i, e) for value, members in events.items() for i, e in enumerate(members)]
    seqs = [prefix + e['ids'][:-1] for _, _, e in flat]
    width = max(len(x) for x in seqs)
    pad = tokenizer.eos_token_id
    ids = torch.tensor([x + [pad] * (width - len(x)) for x in seqs], device=device)
    mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in seqs], device=device)
    keep = width - (len(prefix) - 1)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits.float()
    if not torch.isfinite(logits).all():
        raise ValueError('nonfinite logits')
    lps = logits.log_softmax(-1)
    likelihood = {}
    for row, (value, i, e) in enumerate(flat):
        steps = torch.arange(len(e['ids']), device=lps.device)
        likelihood[value, i] = float(lps[row, steps, torch.tensor(e['ids'], device=lps.device)].sum())
    masses, surfaces = {}, {}
    for value, members in events.items():
        lp = [likelihood[value, i] for i in range(len(members))]
        surfaces[value] = [{'text': e['text'], 'ids': e['ids'], 'log_probability': x} for e, x in zip(members, lp)]
        top = max(lp)
        masses[value] = top + math.log(sum(math.exp(x - top) for x in lp))
    return masses, surfaces


def scorer(config):
    if config['scorer'] == 'batched':
        return score_batched
    if config['scorer'] == 'kv_cache':
        return score_cached
    return lambda model, tok, prompt, events: score_prompt(model, tok, prompt, events)


def verify_cached_scorer(model, tok, rows, n=12, tolerance=0.02, fast_scorer=None):
    """Compare a fast scorer (KV-cache by default) with the reference scorer on identical prompts and events."""
    fast_scorer = fast_scorer or score_cached
    report = []
    for r in rows[:n]:
        for name, (prompt, events) in {
                'downstream': (v1.rendered(tok, build_prompt(r)[0]), None),
                'easy_target': (v1.rendered(tok, build_prompt(r, 'easy')[0]), None)}.items():
            ev = action_events(tok, prompt, r['actions']) if name == 'downstream' else continuations(tok, prompt, values=list(r['policy']))
            ref = score_prompt(model, tok, prompt, ev)[0]
            fast = fast_scorer(model, tok, prompt, ev)[0]
            report.append({'example_id': r['example_id'], 'query': name,
                           'max_abs_diff': max(abs(ref[k] - fast[k]) for k in ref)})
    worst = max(x['max_abs_diff'] for x in report)
    return {'comparisons': report, 'max_abs_diff': worst, 'tolerance': tolerance, 'passed': worst <= tolerance}


def token_audit(rows, tok):
    """v1 geometry audit of easy and edited spans, plus the v1b action events and historical prompts."""
    validate(rows)
    result = {'dataset_hash': digest(rows), 'tokenizer_hash': v1.tokenizer_fingerprint(tok),
              'score_event': 'full action label as generated at assistant-turn start (no leading space) + designated EOS',
              'prompts': {}, 'pairs': []}
    pair_groups = {}
    for r in rows:
        variants = {}
        queries = [('downstream', 'downstream', False), ('easy_target', 'easy', False), ('easy_other', 'easy', True)]
        for name, query, other in queries:
            body, char_span = build_prompt(r, query, other)
            prompt = v1.rendered(tok, body)
            events = action_events(tok, prompt, r['actions']) if query == 'downstream' else continuations(tok, prompt, values=list(r['policy']))
            offsets = None
            if char_span:
                if prompt.count(body) != 1:
                    raise ValueError('chat template must preserve body exactly once')
                shift = prompt.index(body)
                offsets = token_span(tok, prompt, shift + char_span[0], shift + char_span[1])
            variants[name] = {'prompt_hash': digest(prompt), 'n_tokens': len(encode(tok, prompt)),
                              'edit_token_span': offsets, 'events': events}
        if r['family'] == HISTORICAL_FAMILY:
            prompt = v1.rendered(tok, historical_prompt(r)[0])
            variants['historical'] = {'prompt_hash': digest(prompt), 'n_tokens': len(encode(tok, prompt)),
                                      'events': continuations(tok, prompt, values=list(r['policy']))}
        result['prompts'][r['example_id']] = variants
        pair_groups.setdefault(r['pair_id'], {})[r['member']] = r
    for pid, members in pair_groups.items():
        if len(members) == 1:
            continue
        a, b = members[0], members[1]
        for name, query, other in [('downstream', 'downstream', False), ('easy_target', 'easy', False), ('easy_other', 'easy', True)]:
            ai = encode(tok, v1.rendered(tok, build_prompt(a, query, other)[0]))
            bi = encode(tok, v1.rendered(tok, build_prompt(b, query, other)[0]))
            aa = result['prompts'][a['example_id']][name]['edit_token_span']
            bb = result['prompts'][b['example_id']][name]['edit_token_span']
            if ai[:aa[0]] != bi[:bb[0]] or ai[aa[-1]+1:] != bi[bb[-1]+1:] or ai[aa[0]:aa[-1]+1] == bi[bb[0]:bb[-1]+1]:
                raise ValueError('token edit alters surrounding context or has no change')
            result['pairs'].append({'pair_id': pid, 'query': name, 'length_delta': len(bi) - len(ai)})
    result['token_counts_by_family'] = {}
    for family in sorted({r['family'] for r in rows}):
        counts = [result['prompts'][r['example_id']]['downstream']['n_tokens'] for r in rows if r['family'] == family]
        result['token_counts_by_family'][family] = {'min': min(counts), 'max': max(counts), 'mean': sum(counts) / len(counts)}
    return result


ROUNDING_TOLERANCE = 1e-5  # float32 log_softmax can round a log probability of ~0 to +1e-7


def validate_easy(rows, scores, require_checksums=False):
    """v1.validate_easy with the documented float32 rounding tolerance on log probabilities and masses."""
    expected = {r['example_id']: r for r in rows}
    cache = {}
    for s in scores:
        v1.verify_record_checksum(s, required=require_checksums)
        eid = s['example_id']
        if eid in cache or eid not in expected or s['row_hash'] != digest(expected[eid]):
            raise ValueError('invalid easy checkpoint metadata')
        r = expected[eid]
        v1._check_generation_tokens(s, 'easy')
        for field in ('easy_target', 'easy_other'):
            if set(s[field]) != set(r['policy']) or not all(
                    math.isfinite(v) and v <= ROUNDING_TOLERANCE for v in s[field].values()):
                raise ValueError('invalid easy checkpoint scores')
            if sum(math.exp(v) for v in s[field].values()) > 1 + ROUNDING_TOLERANCE:
                raise ValueError('easy candidate probability mass exceeds one')
        if s['easy_generated'] != v1.classify(s['easy_raw_text'], list(r['policy']), r['correct_state'], s['easy_terminated']):
            raise ValueError('invalid easy checkpoint generation')
        cache[eid] = s
    return cache


def _check_scores(row, score, expected_model_id=None):
    """v1.check_score_record with the documented float32 rounding tolerance."""
    if expected_model_id is not None and score.get('model_id') != expected_model_id:
        raise ValueError('score model ID differs from frozen configuration')
    for field, labels in [('action_logp', row['actions']), ('easy_target', row['policy']), ('easy_other', row['policy'])]:
        values = score.get(field)
        if not isinstance(values, dict) or set(values) != set(labels) or not all(
                isinstance(v, (int, float)) and math.isfinite(v) and v <= ROUNDING_TOLERANCE for v in values.values()):
            raise ValueError('missing/nonfinite/invalid scores')
        if sum(math.exp(v) for v in values.values()) > 1 + ROUNDING_TOLERANCE:
            raise ValueError('disjoint event mass exceeds one')
    v1._check_generation_tokens(score, '')
    v1._check_generation_tokens(score, 'easy')
    if score.get('generated') != v1.classify(score.get('raw_text'), row['actions'], row['answer'], score['terminated']):
        raise ValueError('generation parse mismatch')
    if score.get('easy_generated') != v1.classify(score.get('easy_raw_text'), list(row['policy']), row['correct_state'],
                                                  score['easy_terminated']):
        raise ValueError('easy generation parse mismatch')
    mass = sum(math.exp(v) for v in score['action_logp'].values())
    if abs(score.get('canonical_valid_mass', float('nan')) - mass) > 1e-9:
        raise ValueError('invalid canonical probability mass')
    if not all(isinstance(score.get(k), (int, float)) and math.isfinite(score[k]) for k in ('prompt_tokens', 'history_position')):
        raise ValueError('nonfinite predictors')


def check_score_record(row, score, expected_model_id=None):
    _check_scores(row, score, expected_model_id)
    hist = score.get('historical')
    if row['family'] == HISTORICAL_FAMILY:
        if not isinstance(hist, dict) or set(hist.get('scores', {})) != set(row['policy']):
            raise ValueError('missing historical retrieval scores')
        if not all(math.isfinite(v) and v <= ROUNDING_TOLERANCE for v in hist['scores'].values()):
            raise ValueError('invalid historical scores')
        answer = row['old_values'][row['member']]
        if hist.get('generated') != v1.classify(hist.get('raw_text'), list(row['policy']), answer, hist.get('terminated')):
            raise ValueError('historical generation parse mismatch')
    elif hist is not None:
        raise ValueError('historical retrieval only in the superseded construction')


def checked_scores(rows, scores, complete=True, expected_model_id=None, require_checksums=False):
    validate(rows)
    expected = {r['example_id']: r for r in rows}
    seen, models = set(), set()
    for s in scores:
        v1.verify_record_checksum(s, required=require_checksums)
        eid = s['example_id']
        if eid in seen or eid not in expected or s.get('row_hash') != digest(expected[eid]):
            raise ValueError('duplicate/unknown/stale score')
        seen.add(eid)
        models.add(s['model_id'])
        check_score_record(expected[eid], s, expected_model_id)
    if len(models) > 1:
        raise ValueError('analyze models separately')
    if complete and seen != set(expected):
        raise ValueError('missing score records')
    return seen


def run(rows, config, output, model, tok, resume=False, limit=None):
    """Same order and checkpoint contract as v1.run: every easy query first, then downstream."""
    check_config(config)
    score = scorer(config)
    audit = token_audit(rows, tok)
    manifest = {'protocol': PROTOCOL, 'dataset_hash': digest(rows), 'config_hash': digest(config),
                'code_hash': code_hash(), 'tokenizer_hash': audit['tokenizer_hash'], 'audit_hash': digest(audit),
                'config': config, 'runtime': v1.runtime_info(model), 'limit': limit}
    output = Path(output)
    sidecar = str(output) + '.run.json'
    if resume:
        if json.loads(Path(sidecar).read_text()) != manifest:
            raise ValueError('resume manifest mismatch')
        saved = v1.recover_jsonl(output)
        done = checked_scores(rows, saved, complete=False, expected_model_id=config['model']['id'], require_checksums=True)
    else:
        for suffix in ('', '.run.json', '.audit.json', '.easy.jsonl', '.complete.json'):
            if Path(str(output) + suffix).exists():
                raise FileExistsError(str(output) + suffix)
        write_new(sidecar, manifest)
        write_new(str(output) + '.audit.json', audit)
        output.touch(exist_ok=False)
        done = set()
    selected = rows if limit is None else rows[:limit]
    easy_path = Path(str(output) + '.easy.jsonl')
    easy_cache = validate_easy(rows, v1.recover_jsonl(easy_path) if resume else [], require_checksums=resume)
    for r in selected:
        eid = r['example_id']
        if eid in easy_cache:
            continue
        values = {}
        for name, other in [('easy_target', False), ('easy_other', True)]:
            p = v1.rendered(tok, build_prompt(r, 'easy', other)[0])
            values[name] = score(model, tok, p, continuations(tok, p, values=list(r['policy'])))[0]
            if name == 'easy_target':
                easy_raw, easy_ids, easy_terminated = v1.generate_response(model, tok, p, config['max_new_tokens'])
        saved_easy = dict(example_id=eid, row_hash=digest(r), **values, easy_raw_text=easy_raw,
                          easy_token_ids=easy_ids, easy_terminated=easy_terminated, eos_token_id=tok.eos_token_id,
                          easy_generated=v1.classify(easy_raw, list(r['policy']), r['correct_state'], easy_terminated))
        validate_easy(rows, [saved_easy])
        v1.append_record(easy_path, saved_easy)
        easy_cache[eid] = saved_easy
    for r in selected:
        if r['example_id'] in done:
            continue
        easy = easy_cache[r['example_id']]
        p = v1.rendered(tok, build_prompt(r)[0])
        action_logp = score(model, tok, p, action_events(tok, p, r['actions']))[0]
        raw, raw_ids, terminated = v1.generate_response(model, tok, p, config['max_new_tokens'])
        hist = None
        if r['family'] == HISTORICAL_FAMILY:
            hp_body, answer = historical_prompt(r)
            hp = v1.rendered(tok, hp_body)
            h_raw, h_ids, h_term = v1.generate_response(model, tok, hp, config['max_new_tokens'])
            hist = {'scores': score(model, tok, hp, continuations(tok, hp, values=list(r['policy'])))[0],
                    'raw_text': h_raw, 'token_ids': h_ids, 'terminated': h_term, 'answer': answer,
                    'generated': v1.classify(h_raw, list(r['policy']), answer, h_term)}
        prompt_audit = audit['prompts'][r['example_id']]['downstream']
        s = dict(example_id=r['example_id'], row_hash=digest(r), model_id=config['model']['id'],
                 easy_target=easy['easy_target'], easy_other=easy['easy_other'], action_logp=action_logp,
                 raw_text=raw, generated_token_ids=raw_ids, terminated=terminated,
                 easy_raw_text=easy['easy_raw_text'], easy_token_ids=easy['easy_token_ids'],
                 easy_terminated=easy['easy_terminated'], eos_token_id=tok.eos_token_id,
                 generated=v1.classify(raw, r['actions'], r['answer'], terminated),
                 easy_generated=easy['easy_generated'], historical=hist,
                 canonical_valid_mass=sum(math.exp(v) for v in action_logp.values()),
                 prompt_tokens=prompt_audit['n_tokens'], history_position=(prompt_audit['edit_token_span'] or [-1])[0])
        check_score_record(r, s, config['model']['id'])
        v1.append_record(output, s)
    if limit is not None:
        return
    final = read_rows(output)
    checked_scores(rows, final, expected_model_id=config['model']['id'], require_checksums=True)
    write_new(str(output) + '.complete.json', {
        'status': 'complete', 'records': len(final),
        'scores_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'easy_sha256': hashlib.sha256(easy_path.read_bytes()).hexdigest(),
        'run_manifest_sha256': hashlib.sha256(Path(sidecar).read_bytes()).hexdigest()})


def verify_complete_run(output, rows, config):
    output = Path(output)
    completion = json.loads(Path(str(output) + '.complete.json').read_text())
    manifest_bytes = Path(str(output) + '.run.json').read_bytes()
    expected = {'status': 'complete', 'records': len(rows),
                'scores_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'easy_sha256': hashlib.sha256(Path(str(output) + '.easy.jsonl').read_bytes()).hexdigest(),
                'run_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()}
    if completion != expected:
        raise ValueError('completed score artifacts differ from their saved content hashes')
    manifest = json.loads(manifest_bytes)
    if (manifest.get('dataset_hash') != digest(rows) or manifest.get('config_hash') != digest(config) or
            manifest.get('code_hash') != code_hash()):
        raise ValueError('completed run provenance mismatch')
    scores = read_rows(output)
    checked_scores(rows, scores, expected_model_id=config['model']['id'], require_checksums=True)
    return manifest, scores
