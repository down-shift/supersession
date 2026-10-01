"""Tokenizer-only surface event and exact semantic edit validation."""
import re
from collections import defaultdict

from src.cross_model.protocol import VALUES, digest
from src.data.supersession_behavior import render_behavior_example


def surfaces(value):
    return [' ' + value, ' ' + value.title(), value, value.title()]


def encode(tokenizer, text):
    return list(tokenizer(text, add_special_tokens=False)['input_ids'])


def continuations(tokenizer, prompt):
    prefix = encode(tokenizer, prompt)
    events, owners = {}, {}
    for value in VALUES:
        events[value] = []
        seen = set()
        for text in surfaces(value):
            full = encode(tokenizer, prompt + text)
            if full[:len(prefix)] != prefix or len(full) <= len(prefix):
                if text.startswith(' '):
                    raise ValueError(f'required spaced continuation retokenizes prefix: {text!r}')
                continue  # Unspaced variants are eligible only if prefix-preserving.
            sequence = tuple(full[len(prefix):])
            if sequence in owners and owners[sequence] != value:
                raise ValueError('cross-semantic token collision')
            owners[sequence] = value
            if sequence not in seen:
                events[value].append({'text': text, 'ids': list(sequence), 'surface_strings': [text]})
                seen.add(sequence)
            else:
                next(e for e in events[value] if tuple(e['ids']) == sequence)['surface_strings'].append(text)
    sequences = list(owners)
    for a in sequences:
        for b in sequences:
            if len(a) < len(b) and b[:len(a)] == a:
                raise ValueError('surface token events are not prefix-free; probability sum would double count')
    return events


def surface_geometry_audit(events):
    """Visible tokenizer geometry, not a balance-based vocabulary selector."""
    if set(events) != set(VALUES): raise ValueError('surface audit requires all fixed values')
    table = {}
    for value, members in events.items():
        if not members: raise ValueError('empty semantic surface class')
        table[value] = {'distinct_event_count': len(members),
                        'events': [{'token_length': len(e['ids']), 'token_ids': e['ids'],
                                    'surface_strings': e['surface_strings']} for e in members]}
    return {'by_value': table,
            'equal_event_counts': len({len(m) for m in events.values()}) == 1,
            'equal_token_length_profiles': len({tuple(sorted(len(e['ids']) for e in m)) for m in events.values()}) == 1,
            'policy': 'Mandatory disclosure before logits; unequal event counts/lengths are not exclusions'}


def token_span(tokenizer, prompt, start, end):
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    positions = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if a < end and b > start]
    if not positions or positions != list(range(positions[0], positions[-1]+1)):
        raise ValueError('semantic span must cover contiguous tokens')
    return positions


def semantic_positions(row, tokenizer):
    prompt = render_behavior_example(row, tokenizer, True)
    # Each semantic value occurs once in this renderer; never locate by token ID.
    sites = {'final_preanswer': [len(encode(tokenizer, prompt))-1]}
    for field, value in row['semantic_values'].items():
        hits = list(re.finditer(r'(?<!\w)' + re.escape(value) + r'(?!\w)', prompt))
        if len(hits) != 1:
            raise ValueError(f'ambiguous value span: {field}')
        h = hits[0]; sites[field] = token_span(tokenizer, prompt, h.start(), h.end())
    entity = row['variables'][('x', 'z').index(row['query'])]
    question = prompt.rfind('What is ' + entity)
    if question < 0: raise ValueError('queried entity location missing')
    start = question + len('What is ')
    sites['queried_entity'] = token_span(tokenizer, prompt, start, start+len(entity))
    return prompt, sites


def audit_pairs(rows, tokenizer):
    pairs = defaultdict(dict)
    for row in rows: pairs[row['pair_id']][row['pair_direction']] = row
    audits = []
    for pid, members in sorted(pairs.items()):
        if set(members) != {0, 1}: raise ValueError('incomplete edit pair')
        base, edit = members[0], members[1]
        bp, bs = semantic_positions(base, tokenizer); ep, es = semantic_positions(edit, tokenizer)
        bi, ei = encode(tokenizer, bp), encode(tokenizer, ep)
        field = base['edited_field']; a, b = bs[field], es[field]
        # Even variable-length edits must leave all outside-span tokens identical.
        if bi[:a[0]] != ei[:b[0]] or bi[a[-1]+1:] != ei[b[-1]+1:]:
            raise ValueError(f'{pid}: token edit changes context outside value span')
        if bi[a[0]:a[-1]+1] == ei[b[0]:b[-1]+1]:
            raise ValueError('counterfactual has no token change')
        aligned = len(bi) == len(ei) and a == b
        audits.append({'pair_id': pid, 'base_span': a, 'edit_span': b,
                       'source_value': base['source_value'], 'replacement_value': base['replacement_value'],
                       'base_span_ids': bi[a[0]:a[-1]+1], 'edit_span_ids': ei[b[0]:b[-1]+1],
                       'base_prompt_sha256': digest(bp), 'edit_prompt_sha256': digest(ep),
                       'length_delta': len(ei)-len(bi), 'mechanism_aligned': aligned})
    return {'status': 'passed', 'pairs': audits, 'pairs_checked': len(audits),
            'all_mechanism_aligned': all(a['mechanism_aligned'] for a in audits)}


def audit_all_substitutions(tokenizer, rows):
    """Every candidate in every represented semantic slot, using tokenizer evidence only."""
    representations = defaultdict(lambda: defaultdict(set))
    lengths = []
    prompts_seen = set()
    for row in rows:
        if row['pair_direction'] != 0: continue
        prompt = render_behavior_example(row, tokenizer, True)
        if prompt in prompts_seen: continue
        prompts_seen.add(prompt)
        original_ids = encode(tokenizer, prompt)
        for field, original in row['semantic_values'].items():
            hits = list(re.finditer(r'(?<!\w)' + re.escape(original) + r'(?!\w)', prompt))
            if len(hits) != 1: raise ValueError('ambiguous substitution source')
            start, end = hits[0].span()
            old_span = token_span(tokenizer, prompt, start, end)
            for value in VALUES:
                edited = prompt[:start]+value+prompt[end:]
                new_span = token_span(tokenizer, edited, start, start+len(value))
                edited_ids = encode(tokenizer, edited)
                if (original_ids[:old_span[0]] != edited_ids[:new_span[0]] or
                    original_ids[old_span[-1]+1:] != edited_ids[new_span[-1]+1:]):
                    raise ValueError('substitution changes tokens outside semantic span')
                ids = tuple(edited_ids[new_span[0]:new_span[-1]+1])
                representations[row['condition']+'|'+field][value].add(ids)
                lengths.append(len(new_span)-len(old_span))
    return {'all_substitutions_equal_length': all(x == 0 for x in lengths),
            'substitutions_checked': len(lengths),
            'slot_representations': {k:{v:[list(ids) for ids in sorted(seqs)] for v,seqs in values.items()}
                                    for k,values in representations.items()}}


def validate(tokenizer, rows):
    prompts = sorted({render_behavior_example(r, tokenizer, True) for r in rows})
    maps = {digest(p): continuations(tokenizer, p) for p in prompts}
    signatures = {digest(m) for m in maps.values()}
    if len(signatures) != 1:
        raise ValueError('candidate sequences vary across exact rendered contexts')
    events = next(iter(maps.values()))
    canonical = {}
    for value in VALUES:
        ids = encode(tokenizer, prompts[0]+' '+value)[len(encode(tokenizer, prompts[0])):]
        if len(ids) == 1: canonical[value] = ids[0]
    if len(set(canonical.values())) != len(canonical): raise ValueError('canonical token collision')
    return {'events': events, 'surface_geometry_audit': surface_geometry_audit(events), 'surface_rule': 'four proposals; unspaced prefix-changing forms excluded tokenizer-only',
            'excluded_surfaces': {v: [s for s in surfaces(v) if encode(tokenizer, prompts[0]+s)[:len(encode(tokenizer, prompts[0]))] != encode(tokenizer, prompts[0])] for v in VALUES},
            'canonical_one_token_ids': canonical,
            'canonical_raw_R_defined': set(canonical) == set(VALUES),
            'chat_template_sha256': digest(getattr(tokenizer, 'chat_template', None)),
            'tokenizer_sha256': digest(tokenizer.backend_tokenizer.to_str()),
            'prompt_examples': [prompts[0]], 'unique_prefixes_checked': len(prompts),
            'edit_audit': audit_pairs(rows, tokenizer),
            'exhaustive_slot_audit': audit_all_substitutions(tokenizer, rows)}


def check_tokenizer(tokenizer, candidate):
    if candidate['chat_template_sha256'] != digest(getattr(tokenizer, 'chat_template', None)):
        raise ValueError('chat template hash mismatch')
    if candidate['tokenizer_sha256'] != digest(tokenizer.backend_tokenizer.to_str()):
        raise ValueError('tokenizer hash mismatch')
