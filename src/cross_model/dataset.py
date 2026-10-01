"""Shared semantic histories, unchanged nora_v1 rendering."""
import logging

from src.cross_model.protocol import SEEDS, VALUES, VERSION
from src.data.supersession_behavior import generate_behavior_pairs
from src.cross_model.progress import progress

logger = logging.getLogger(__name__)


def generate(stage, n):
    logger.info("Constructing %d %s histories (seed=%s)", n, stage, SEEDS[stage])
    rows = generate_behavior_pairs('controls_counterbalanced', n, VALUES, SEEDS[stage])
    names = ('Nora', 'Liam', 'Ava', 'Omar', 'Mila', 'Eli', 'Iris', 'Noah', 'Zoe', 'Theo', 'Maya', 'Leo')
    attrs = ('badge', 'color', 'code', 'label')
    for r in progress(rows, desc=f'Preparing {stage} histories', unit='history'):
        i = r['history_index']
        r.update(prompt_family='natural_entity_attribute_v1', prompt_variant='nora_v1',
                 entities=[names[(2*i) % 12], names[(2*i+1) % 12]], attribute=attrs[i % 4],
                 cross_model_protocol=VERSION, cross_model_stage=stage)
        r['variables'] = r['entities'] if r['orientation'] == 0 else r['entities'][::-1]
    return rows
