"""stale_decision_think: decisions with reasoning enabled, generation only (docs/stale_decision_think.md).

The sealed confirmation rows (stale_decision_v2, seed 82003, held-out template) are rendered with the chat
template's thinking switch on. Each decision prompt is answered by greedy decoding until the model's end-of-turn
token or `max_new_tokens`; the final answer is the text after the last closing think marker and is parsed with
the strict v1 parser. A generation that is truncated, or whose trace never closes, is invalid. No action event is
teacher-forced after a generated trace; the outcome is the generated decision only.
"""
import hashlib
import json
import os
import re
from pathlib import Path

from src.data.stale_decision_v2 import build_prompt, digest, generate, read_rows, validate, write_new
from src.experiments import stale_decision_v1 as v1

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'stale_decision_think'
MODELS = {'Qwen/Qwen3-8B': 'b968826d9c46dd6066d109eabc6255188de91218',
          'Qwen/Qwen3-14B': '40c069824f4251a91eefaf281ebe4c544efd3e18',
          'mistralai/Ministral-3-14B-Reasoning-2512': '51f9210f3cd20f3452a80d5819d15dc61cc50630'}
FAMILIES = ('superseded', 'updated_other', 'entity_mention', 'unassigned', 'current_only')
TASK = 'routing'


def code_hash():
    files = ['src/experiments/stale_decision_think.py', 'scripts/stale_decision_think.py', 'docs/stale_decision_think.md',
             'src/data/stale_decision_v2.py', 'src/experiments/stale_decision_v1.py']
    return digest({p: (ROOT / p).read_text() for p in files})


def check_config(config):
    m = config.get('model', {})
    if config.get('protocol') != PROTOCOL or MODELS.get(m.get('id')) != m.get('revision'):
        raise ValueError('model and revision must come from the frozen thinking list')
    if (m.get('dtype') != 'float32' or m.get('attn_implementation') != 'sdpa' or
            m.get('load_class') not in ('causal_lm', 'image_text_to_text') or not isinstance(m.get('tokenizer_kwargs'), dict) or
            not isinstance(m.get('end_token'), str) or not isinstance(m.get('think_close'), str)):
        raise ValueError('thinking runtime: float32, sdpa, load class, tokenizer options, end token, think marker')
    if config.get('max_new_tokens') != 4096 or config.get('bootstrap_draws') != 2000:
        raise ValueError('frozen decoding/bootstrap settings differ')
    if config.get('histories') not in ('all', 'first_half'):
        raise ValueError('histories must be all or first_half')


def items(config):
    """Confirmation rows for the routing task and the decision families, in dataset order. first_half keeps the
    first 192 routing histories in dataset order (fixed before any thinking-mode data)."""
    rows = generate('confirmation', 4)
    validate(rows, expected_histories=768)
    keep = [r for r in rows if r['task'] == TASK and r['family'] in FAMILIES]
    if config['histories'] == 'first_half':
        hids = list(dict.fromkeys(r['history_id'] for r in keep))[:192]
        keep = [r for r in keep if r['history_id'] in set(hids)]
    return keep


def load(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    m = config['model']
    cls = AutoModelForCausalLM if m['load_class'] == 'causal_lm' else AutoModelForImageTextToText
    model = cls.from_pretrained(m['id'], revision=m['revision'], torch_dtype=torch.float32,
                                attn_implementation=m['attn_implementation'], device_map=0).eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('model is not entirely float32')
    tok = AutoTokenizer.from_pretrained(m['id'], revision=m['revision'], padding_side='left', **m['tokenizer_kwargs'])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def render(tok, r):
    return tok.apply_chat_template([{'role': 'user', 'content': build_prompt(r, 'downstream')[0]}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)


def parse(text, config, r, terminated):
    """Final answer = text after the last closing think marker, with the end token removed."""
    m = config['model']
    closed = m['think_close'] in text
    final = text.rsplit(m['think_close'], 1)[1] if closed else ''
    final = final.replace(m['end_token'], '')
    out = v1.classify(final, r['actions'], r['answer'], terminated and closed)
    return {'closed': closed, 'final_text': final, 'generated': out}


def generate_batch(model, tok, config, batch):
    import torch
    m = config['model']
    end_id = tok.convert_tokens_to_ids(m['end_token'])
    if end_id == tok.unk_token_id:
        raise ValueError('end token missing from the tokenizer')
    texts = [render(tok, r) for r in batch]
    enc = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        out = model.generate(**enc, do_sample=False, num_beams=1, max_new_tokens=config['max_new_tokens'],
                             eos_token_id=end_id, pad_token_id=tok.pad_token_id)
    records = []
    for r, ids in zip(batch, out[:, enc['input_ids'].shape[1]:].tolist()):
        if end_id in ids:
            ids = ids[:ids.index(end_id) + 1]
        terminated = bool(ids) and ids[-1] == end_id
        text = tok.decode(ids, skip_special_tokens=False)
        records.append({'example_id': r['example_id'], 'row_hash': digest(r), 'model_id': m['id'], 'protocol': PROTOCOL,
                        'raw_text': text, 'generated_token_ids': ids, 'n_generated_tokens': len(ids),
                        'terminated': terminated, **parse(text, config, r, terminated)})
    return records


def run(config, output, model, tok, batch_size, limit=None):
    """Resumable: appends one JSON line per item, fsync per batch; skips items already present."""
    rows = items(config)[:limit] if limit else items(config)
    path = Path(output)
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            done.add(rec['example_id'])
    todo = [r for r in rows if r['example_id'] not in done]
    with path.open('a') as f:
        for i in range(0, len(todo), batch_size):
            for rec in generate_batch(model, tok, config, todo[i:i + batch_size]):
                f.write(json.dumps(rec, sort_keys=True) + '\n')
            f.flush()
            os.fsync(f.fileno())
    return rows


def analyze(config, records):
    from src.analysis.stale_decision_v1 import bootstrap
    rows = {r['example_id']: r for r in items(config)}
    byid = {x['example_id']: x for x in records}
    if set(byid) != set(rows) or any(byid[k]['row_hash'] != digest(rows[k]) for k in rows):
        raise ValueError('records do not match the frozen item set')
    hist = {}
    for k, r in rows.items():
        hist.setdefault(r['history_id'], {})[(r['family'], r['member'])] = byid[k]
    out = {'n_histories': len(hist), 'items': len(rows)}
    for f in FAMILIES[:-1]:
        sw = [int(h[(f, 0)]['generated']['classification'] == 'wrong') - int(h[(f, 1)]['generated']['classification'] == 'wrong')
              for h in hist.values()]
        out[f'switch_{f}'] = bootstrap(sw)
        out[f'switch_counts_{f}'] = {'to_wrong': sw.count(1), 'to_correct': sw.count(-1)}
    out['current_only_accuracy'] = bootstrap([int(h[('current_only', 0)]['generated']['classification'] == 'correct')
                                              for h in hist.values()])
    rec = list(byid.values())
    out['invalid_rate'] = sum(x['generated']['classification'] == 'invalid' for x in rec) / len(rec)
    out['truncated_rate'] = sum(not x['terminated'] for x in rec) / len(rec)
    out['unclosed_rate'] = sum(not x['closed'] for x in rec) / len(rec)
    lengths = sorted(x['n_generated_tokens'] for x in rec)
    out['generated_tokens'] = {'median': lengths[len(lengths) // 2], 'p95': lengths[int(0.95 * (len(lengths) - 1))],
                               'max': lengths[-1]}
    return out
