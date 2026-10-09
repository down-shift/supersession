"""stale_decision_v2 construction: v1's generator with three changes (docs/stale_decision_v2.md):
no trailing "Answer:" line in the user message, new seeds, new entity-name pools. No model imports."""
import hashlib
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

SCHEMA = 'stale_decision_v2'
SEEDS = {'development': 82001, 'frozen_gate': 82002, 'confirmation': 82003}
FAMILIES = ('superseded', 'updated_other', 'entity_mention', 'unassigned', 'current_only', 'live')
GRID = tuple(itertools.product(('access', 'routing'), ('sequential', 'interleaved', 'competing'),
                               ('natural', 'opaque'), (0, 1), (0, 1), (0, 1), (0, 1)))
VOCABS = {'natural': ('active', 'pending', 'restricted', 'suspended'),
          'opaque': ('S1', 'S2', 'S3', 'S4')}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def signature(r):
    """Includes both edit alternatives; invariant to entity orientation and template."""
    assignments = sorted(zip(r['entities'], r['currents']))
    return digest([r['task'], assignments, sorted(r['policy'].items()), sorted(r['old_values']),
                   r['difficulty'], r['vocabulary']])


def build_prompt(r, query='downstream', other=False):
    entities, currents = r['entities'], list(r['currents'])
    if r['family'] == 'live':
        currents[0] = r['old_values'][r['member']]
    old = r['old_values'][r['member']] if r['family'] != 'current_only' else None
    attribute = 'clearance' if r['task'] == 'access' else 'routing state'
    historical = []
    for idx in r['historical_order']:
        e = entities[idx]
        value = old if idx == 0 and r['family'] != 'live' else r['other_old']
        if r['family'] in ('superseded', 'live'):
            line = f'Initially, {e} had {attribute} {value}.'
        elif r['family'] == 'updated_other':
            line = f'Initially, {e} had archive tag {value}.'
        elif r['family'] == 'entity_mention':
            line = f'Initially, {e} mentioned the word {value}.'
        else:
            line = f'Initially, the unassigned word was {value}.'
        historical.append(line)
    updates = []
    for idx in r['current_order']:
        e = entities[idx]
        # Equal block size in A-D: the other-tag update is supplied everywhere.
        updates.extend([f'Later, {e} had archive tag {r["other_current"]}.',
                        f'Finally, {e} had {attribute} {currents[idx]}.'])
    if r['family'] == 'current_only':
        lines = updates
    elif r['difficulty'] == 'sequential':
        lines = historical + updates
    else:
        lines = [historical[0]] + updates[:2] + [historical[1]] + updates[2:]
    if r['difficulty'] == 'competing' and r['family'] != 'current_only':
        # Relevant competing references, identical in both members; never the edit site.
        lines.insert(-2, f'An archived discussion of {entities[0]} quoted {r["other_old"]}.')
        lines.insert(-2, f'An archived discussion of {entities[1]} quoted {r["other_current"]}.')
    target = entities[int(other)]
    if r['template'] == 'heldout':
        intro = 'Read the dated record. Only the final assignment governs the decision.'
        direct = f'Report the final {attribute} for {target}.'
        question = f'Apply the supplied rule to the final assignment of {target}; give its action.'
    else:
        intro = 'Use only CURRENT assignments. Historical records do not set current values.'
        direct = f'What is {target}\'s CURRENT {attribute}?'
        question = f'What action does the CURRENT {attribute} of {target} require?'
    policy = ', '.join(f'{v} -> {r["policy"][v]}' for v in r['policy_order'])
    if r['task'] == 'routing':
        # Mechanically checked second lookup, with no state-dependent shortcuts.
        keys = {v: f'B{i}' for i, v in enumerate(r['policy_order'])}
        policy = 'state to bin: ' + ', '.join(f'{v} -> {keys[v]}' for v in r['policy_order'])
        policy += '; bin to action: ' + ', '.join(f'{keys[v]} -> {r["policy"][v]}' for v in r['policy_order'])
    suffix = direct if query == 'easy' else question
    labels = list(VOCABS[r['vocabulary']]) if query == 'easy' else r['actions']
    body = '\n'.join([intro, *lines, 'Policy: ' + policy, suffix,
                      'Respond with exactly one label (' + ', '.join(labels) + '), with no other text.'])
    # Offset is constructed from the intended sentence, never global value search.
    if old is not None:
        edit_line = historical[r['historical_order'].index(0)]
        if r['family'] == 'live':
            edit_line = f'Finally, {entities[0]} had {attribute} {currents[0]}.'
        if r['family'] == 'live':
            line_start = body.index(edit_line)
        else:
            slot = r['historical_order'].index(0)
            line_index = slot if r['difficulty'] == 'sequential' else (0 if slot == 0 else 3)
            line_start = len(intro) + 1 + sum(len(line) + 1 for line in lines[:line_index])
        start = line_start + edit_line.rindex(old)
        span = [start, start + len(old)]
    else:
        span = None
    return body, span


def generate(split='development', replicates=1):
    if split not in SEEDS or not isinstance(replicates, int) or replicates < 1:
        raise ValueError('invalid split/replicates')
    rng = random.Random(SEEDS[split])
    rows = []
    for rep in range(replicates):
        for i, (task, difficulty, vocabulary, action_bit, ho, co, entity_bit) in enumerate(GRID):
            if i % 16 == 0:
                base_states = list(VOCABS[vocabulary]); rng.shuffle(base_states)
            rotation = (2 * action_bit + 2 * ho + co) % 4
            states = base_states[rotation:] + base_states[:rotation]
            actions = ['APPROVE', 'DENY'] if task == 'access' else ['EAST', 'WEST']
            policy = dict(zip(states, [actions[0], actions[0], actions[1], actions[1]]))
            current = states[2 * action_bit]
            old_correct = states[2 * action_bit + 1]
            old_wrong = states[2 * (1 - action_bit)]
            pool = {'development': 'Dvb', 'frozen_gate': 'Gtb', 'confirmation': 'Evb'}[split]
            entities = [f'{pool}Unit{rep * len(GRID) + i:04d}A', f'{pool}Unit{rep * len(GRID) + i:04d}B']
            if entity_bit:
                entities.reverse()
            hid = f'{SCHEMA}:{split}:{rep}:{i}'
            base = dict(schema=SCHEMA, split=split, seed=SEEDS[split], history_id=hid,
                        task=task, difficulty=difficulty, vocabulary=vocabulary, entities=entities,
                        entity_pair_id=digest(sorted(entities)), policy=policy, policy_id=digest(policy),
                        policy_order=list(base_states), actions=actions, currents=[current, states[(2 * action_bit + 2) % 4]],
                        old_values=[old_wrong, old_correct], other_old=old_correct, other_current=current,
                        historical_order=[ho, 1-ho], current_order=[co, 1-co],
                        order_stratum='aligned' if ho == co else 'reversed',
                        template='heldout' if split == 'confirmation' else 'development',
                        template_id='heldout' if split == 'confirmation' else 'development', vocabulary_id=vocabulary)
            base['history_signature'] = signature(base)
            for family in FAMILIES:
                for member in range(1 if family == 'current_only' else 2):
                    r = dict(base, family=family, member=member, pair_id=f'{hid}:{family}',
                             example_id=f'{hid}:{family}:{member}')
                    actual_current = r['old_values'][member] if family == 'live' else current
                    r['correct_state'] = actual_current
                    r['answer'] = policy[actual_current]
                    r['wrong_action'] = next(a for a in actions if a != r['answer'])
                    r['prompt'], r['edit_span'] = build_prompt(r)
                    rows.append(r)
    validate(rows)
    return rows


def validate(rows, *, expected_histories=None):
    if not rows:
        raise ValueError('empty dataset')
    ids, histories, pairs = set(), defaultdict(list), defaultdict(dict)
    if len({r['split'] for r in rows}) != 1:
        raise ValueError('mixed dataset splits')
    for r in rows:
        if r['schema'] != SCHEMA or r['split'] not in SEEDS or r['seed'] != SEEDS[r['split']]:
            raise ValueError('protocol/seed mismatch')
        if r['example_id'] in ids or r['family'] not in FAMILIES:
            raise ValueError('duplicate ID or invalid family')
        ids.add(r['example_id']); histories[r['history_id']].append(r)
        if r['member'] in pairs[r['pair_id']]:
            raise ValueError('duplicate member')
        pairs[r['pair_id']][r['member']] = r
        if (r['task'] not in ('access', 'routing') or
                r['difficulty'] not in ('sequential', 'interleaved', 'competing') or
                r['vocabulary'] not in VOCABS):
            raise ValueError('condition is outside the frozen task grid')
        states = set(VOCABS[r['vocabulary']])
        if (len(r['currents']) != 2 or len(r['old_values']) != 2 or
                len(r['policy_order']) != len(states) or set(r['policy_order']) != states or
                len(set(r['policy_order'])) != len(states)):
            raise ValueError('policy order and assignment vectors must cover the frozen states')
        if r['actions'] != (['APPROVE', 'DENY'] if r['task'] == 'access' else ['EAST', 'WEST']):
            raise ValueError('action candidates differ from the frozen task vocabulary')
        if r['vocabulary_id'] != r['vocabulary']:
            raise ValueError('vocabulary group identifier mismatch')
        if r['policy_id'] != digest(r['policy']) or r['entity_pair_id'] != digest(sorted(r['entities'])):
            raise ValueError('policy/entity group identifier mismatch')
        expected_template = 'heldout' if r['split'] == 'confirmation' else 'development'
        if r['template'] != expected_template or r['template_id'] != expected_template:
            raise ValueError('template holdout violation')
        if not set(r['currents'] + r['old_values'] + [r['other_old'], r['other_current']]) <= states:
            raise ValueError('unknown state label')
        if r['order_stratum'] != ('aligned' if r['historical_order'] == r['current_order'] else 'reversed'):
            raise ValueError('order stratum mismatch')
        if any(old == r['other_current'] for old in r['old_values']):
            raise ValueError('other attribute not superseded')
        if r['other_old'] != r['old_values'][1] or r['other_current'] != r['currents'][0]:
            raise ValueError('matched unrelated-attribute assignments changed')
        if set(r['policy']) != states or len(set(r['actions'])) != 2 or set(r['policy'].values()) != set(r['actions']):
            raise ValueError('policy/candidate collision')
        if any(list(r['policy'].values()).count(a) != 2 for a in r['actions']):
            raise ValueError('policy must be two-to-two')
        if len(set(r['entities'])) != 2 or any(sorted(r[k]) != [0, 1] for k in ('historical_order', 'current_order')):
            raise ValueError('entity/order collision')
        if len(set([r['currents'][0], *r['old_values']])) != 3:
            raise ValueError('historical alternatives must both be obsolete')
        if r['policy'][r['old_values'][0]] == r['policy'][r['currents'][0]] or r['policy'][r['old_values'][1]] != r['policy'][r['currents'][0]]:
            raise ValueError('swapped historical action labels')
        current = r['old_values'][r['member']] if r['family'] == 'live' else r['currents'][0]
        if r['correct_state'] != current or r['answer'] != r['policy'][current] or r['wrong_action'] != next(a for a in r['actions'] if a != r['answer']):
            raise ValueError('incorrect ground truth')
        if (r['prompt'], r['edit_span']) != build_prompt(r) or r['history_signature'] != signature(r):
            raise ValueError('prompt/span/signature mismatch')
        if r['pair_id'] != f"{r['history_id']}:{r['family']}" or r['example_id'] != f"{r['pair_id']}:{r['member']}":
            raise ValueError('pair/example identifier mismatch')
    seen_histories = set()
    for hid, records in histories.items():
        expected_id_prefix = f"{SCHEMA}:{records[0]['split']}:"
        if not hid.startswith(expected_id_prefix):
            raise ValueError('history identifier/split mismatch')
        sig = records[0]['history_signature']
        if sig in seen_histories:
            raise ValueError('duplicate concrete history')
        seen_histories.add(sig)
        if {(r['family'], r['member']) for r in records} != {(f, m) for f in FAMILIES for m in range(1 if f == 'current_only' else 2)}:
            raise ValueError('missing history cells')
        reference = records[0]
        changing = {'family', 'member', 'pair_id', 'example_id', 'correct_state', 'answer', 'wrong_action', 'prompt', 'edit_span'}
        for r in records:
            if {k:v for k,v in r.items() if k not in changing} != {k:v for k,v in reference.items() if k not in changing}:
                raise ValueError('history metadata changed')
    if expected_histories is not None:
        if expected_histories < 1 or len(histories) != expected_histories:
            raise ValueError('dataset does not contain the frozen number of histories')
        split = rows[0]['split']
        expected_ids = {f'{SCHEMA}:{split}:{rep}:{i}'
                        for rep in range(expected_histories // len(GRID))
                        for i in range(len(GRID))}
        if expected_histories % len(GRID) or set(histories) != expected_ids:
            raise ValueError('dataset does not cover the complete frozen factorial split')
    for members in pairs.values():
        a = members[0]
        if a['family'] == 'current_only':
            continue
        if set(members) != {0, 1}:
            raise ValueError('missing pair')
        b = members[1]; sa, sb = a['edit_span'], b['edit_span']
        if a['prompt'][:sa[0]] != b['prompt'][:sb[0]] or a['prompt'][sa[1]:] != b['prompt'][sb[1]:]:
            raise ValueError('edit changed outside intended occurrence')
        if a['family'] != 'live' and (a['answer'], a['correct_state']) != (b['answer'], b['correct_state']):
            raise ValueError('paired current state/action changed')
    return True


def verify_no_overlap(*datasets):
    seen = set()
    for rows in datasets:
        validate(rows)
        signatures = {signature(r) for r in rows}
        if seen & signatures:
            raise ValueError('history leakage including paired edits/orientations')
        seen |= signatures
