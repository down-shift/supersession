from src.cross_model.robustness_v2 import generate, validate, render, CONDITIONS, COUNTS


def test_full_condition_and_independent_order_crossing():
    histories, rows = generate('development')
    assert len(histories) == COUNTS['development']
    validate(histories, rows, 'development')
    for h in histories:
        cells = {(r['condition'], r['historical_entity_order'], r['current_entity_order'], r['query'])
                 for r in rows if r['history_id'] == h['history_id']}
        assert cells == {(c,ho,co,q) for c in CONDITIONS for ho in (0,1) for co in (0,1) for q in ('x','z')}


def test_determinism_fresh_stage_ids_and_pair_matching():
    h1,r1=generate('frozen_gate'); h2,r2=generate('frozen_gate')
    assert h1 == h2 and r1 == r2
    assert not ({h['history_id'] for h in h1} & {h['history_id'] for h in generate('development')[0]})
    for row in r1:
        if row['condition'] in ('entity_mention','other_attribute'):
            prompt=render(row)
            if row['condition']=='entity_mention':
                assert 'unrelated note about' in prompt
            else:
                assert 'badge' in prompt or 'tag' in prompt
    assert validate(h1,r1,'frozen_gate')
