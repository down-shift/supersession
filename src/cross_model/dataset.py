"""Shared semantic histories, unchanged nora_v1 rendering."""
from src.cross_model.protocol import SEEDS, VALUES, VERSION
from src.data.supersession_behavior import generate_behavior_pairs


def generate(stage, n):
    rows = generate_behavior_pairs('controls_counterbalanced', n, VALUES, SEEDS[stage])
    names = ('Nora', 'Liam', 'Ava', 'Omar', 'Mila', 'Eli', 'Iris', 'Noah', 'Zoe', 'Theo', 'Maya', 'Leo')
    attrs = ('badge', 'color', 'code', 'label')
    for r in rows:
        i = r['history_index']
        r.update(prompt_family='natural_entity_attribute_v1', prompt_variant='nora_v1',
                 entities=[names[(2*i) % 12], names[(2*i+1) % 12]], attribute=attrs[i % 4],
                 cross_model_protocol=VERSION, cross_model_stage=stage)
        r['variables'] = r['entities'] if r['orientation'] == 0 else r['entities'][::-1]
    return rows
