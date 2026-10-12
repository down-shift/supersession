"""stale_decision_v2s: static other-attribute construction on the v2 confirmation routing histories
(docs/stale_decision_v2_balanced.md, part B). No model imports.

The 384 routing histories of the sealed v2 confirmation split (seed 82003, held-out template) are regenerated
with the sealed v2 generator. For each history, a fifth construction is added: `static_other`. It is the
`updated_other` record (the obsolete line assigns the edited word to the queried entity's archive tag) with the
queried entity's later archive-tag update replaced by a filler line with the same number of words and no state
word, so the archive tag is never updated. Everything else, including both members, is the v2 record.

A reproducibility subset rescores the `updated_other` pairs of the first CHECK_HISTORIES routing histories, which
must reproduce the sealed scores of the same model.
"""
from collections import defaultdict
from functools import lru_cache

from src.data import stale_decision_v2 as v2
from src.data.stale_decision_v2 import digest, read_rows, write_new  # noqa: F401 (re-exported for the CLI)

SCHEMA = 'stale_decision_v2s'
SOURCE_SPLIT, SOURCE_REPLICATES = 'confirmation', 4
FAMILIES = ('static_other', 'updated_other')
FILLER = 'Later, {entity} had a routine audit.'  # six words, as 'Later, {entity} had archive tag {state}.'
CHECK_HISTORIES = 48
N_HISTORIES = 384


def archive_update(r):
    return f'Later, {r["entities"][0]} had archive tag {r["other_current"]}.'


def build_prompt(r, query='downstream', other=False):
    if r['family'] != 'static_other':
        return v2.build_prompt(r, query, other)
    body, span = v2.build_prompt(dict(r, family='updated_other'), query, other)
    line = archive_update(r)
    filler = FILLER.format(entity=r['entities'][0])
    if body.count(line) != 1 or filler in body:
        raise ValueError('archive-tag update of the queried entity must occur exactly once')
    at = body.index(line)
    body = body.replace(line, filler)
    if span is not None and at < span[0]:
        span = [span[0] + len(filler) - len(line), span[1] + len(filler) - len(line)]
    return body, span


@lru_cache(maxsize=1)
def _sources():
    rows = v2.generate(SOURCE_SPLIT, SOURCE_REPLICATES)
    v2.validate(rows, expected_histories=len(v2.GRID) * SOURCE_REPLICATES)
    return tuple(r for r in rows if r['task'] == 'routing')


def sources():
    """Sealed v2 confirmation rows of the routing task (regenerated deterministically and validated by v2)."""
    return [dict(r) for r in _sources()]


def generate():
    src = [r for r in sources() if r['family'] == 'updated_other']
    hids = list(dict.fromkeys(r['history_id'] for r in src))
    check = set(hids[:CHECK_HISTORIES])
    rows = []
    for r in src:
        for family in FAMILIES:
            if family == 'updated_other' and r['history_id'] not in check:
                continue
            x = dict(r, schema=SCHEMA, family=family, source_example_id=r['example_id'],
                     pair_id=f"{r['history_id']}:{family}", example_id=f"{r['history_id']}:{family}:{r['member']}")
            x['prompt'], x['edit_span'] = build_prompt(x)
            rows.append(x)
    validate(rows)
    return rows


def validate(rows, **_):
    if not rows or any(r['schema'] != SCHEMA or r['family'] not in FAMILIES or r['task'] != 'routing' or
                       r['split'] != SOURCE_SPLIT or r['template'] != 'heldout' for r in rows):
        raise ValueError('rows are not v2s static-other routing rows')
    src = {r['example_id']: r for r in sources()}
    ids, pairs, by_family = set(), defaultdict(dict), defaultdict(set)
    for r in rows:
        if r['example_id'] in ids:
            raise ValueError('duplicate ID')
        ids.add(r['example_id'])
        s = src[r['source_example_id']]
        if s['family'] != 'updated_other' or s['history_id'] != r['history_id'] or s['member'] != r['member']:
            raise ValueError('source row mismatch')
        changing = {'schema', 'family', 'source_example_id', 'pair_id', 'example_id', 'prompt', 'edit_span'}
        if {k: v for k, v in r.items() if k not in changing} != {k: v for k, v in s.items() if k not in changing - {'source_example_id'}}:
            raise ValueError('history metadata differs from the sealed source')
        if (r['prompt'], r['edit_span']) != build_prompt(r):
            raise ValueError('prompt/span mismatch')
        if r['family'] == 'updated_other' and (r['prompt'], r['edit_span']) != (s['prompt'], s['edit_span']):
            raise ValueError('reproducibility rows must equal the sealed prompts')
        if r['family'] == 'static_other' and (archive_update(r) in r['prompt'] or
                                              FILLER.format(entity=r['entities'][0]) not in r['prompt']):
            raise ValueError('static construction still updates the archive tag')
        if r['pair_id'] != f"{r['history_id']}:{r['family']}" or r['example_id'] != f"{r['pair_id']}:{r['member']}":
            raise ValueError('pair/example identifier mismatch')
        pairs[r['pair_id']][r['member']] = r
        by_family[r['family']].add(r['history_id'])
    if len(by_family['static_other']) != N_HISTORIES or len(by_family['updated_other']) != CHECK_HISTORIES:
        raise ValueError('static-other rows must cover all routing histories')
    for members in pairs.values():
        if set(members) != {0, 1}:
            raise ValueError('missing pair')
        a, b = members[0], members[1]
        sa, sb = a['edit_span'], b['edit_span']
        if a['prompt'][:sa[0]] != b['prompt'][:sb[0]] or a['prompt'][sa[1]:] != b['prompt'][sb[1]:]:
            raise ValueError('edit changed outside intended occurrence')
        if (a['answer'], a['correct_state']) != (b['answer'], b['correct_state']):
            raise ValueError('paired current state/action changed')
    return True
