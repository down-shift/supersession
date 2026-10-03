import pytest

from scripts.export_cross_model_summaries import (
    span_gap,
    span_to_position_distance,
    v1_attribute_frame_summaries,
)
from src.cross_model.robustness_analysis import summary


def test_token_distance_helpers_respect_half_open_spans():
    assert span_gap(2, 4, 7, 9) == 3
    assert span_gap(2, 4, 4, 5) == 0  # adjacent
    assert span_gap(2, 5, 3, 7) == 0  # overlap
    assert span_to_position_distance(2, 5, 0) == 2
    assert span_to_position_distance(2, 5, 3) == 0
    assert span_to_position_distance(2, 5, 8) == 4
    for args in ((-1, 2, 3, 4), (3, 2, 4, 5)):
        with pytest.raises(ValueError):
            span_gap(*args)
    for args in ((2, 2, 3), (2, 4, -1)):
        with pytest.raises(ValueError):
            span_to_position_distance(*args)


def test_v1_attribute_frame_summaries_reuse_history_rows_exactly():
    values = [1.0, 2.0, -1.0, 4.0]
    live = [0.5, 1.5, 1.0, 2.0]
    rows = [
        {'history_id': f'h{i}',
         'R_superseded_minus_R_irrelevant_counterbalanced': value,
         'R_live_minus_R_superseded': live_value}
        for i, (value, live_value) in enumerate(zip(values, live))
    ]
    dataset_rows = [
        {'history_id': f'h{i}', 'attribute': 'badge' if i < 2 else 'code',
         'prompt_variant': 'frame_a', 'prompt_family': 'family_a'}
        for i in range(4)
    ]
    primary = 'R_superseded_minus_R_irrelevant_counterbalanced'
    secondary = 'R_live_minus_R_superseded'
    report = {'history_rows': rows,
              'statistics': {'results': {
                  primary: summary(values), secondary: summary(live)}}}
    result = v1_attribute_frame_summaries('model', report, dataset_rows)
    assert len(result) == 3  # two attribute slices and one frame marginal
    by_attr = {row['target_attribute']: row for row in result
               if row['summary_level'] == 'attribute × frame'}
    assert by_attr['badge']['n_histories'] == 2
    assert by_attr['code']['n_histories'] == 2
    marginal = next(row for row in result if row['summary_level'] == 'frame marginal')
    assert marginal['primary'] == report['statistics']['results'][primary]
    assert marginal['live_minus_superseded'] == report['statistics']['results'][secondary]


def test_v1_attribute_frame_summaries_reject_metadata_mismatch():
    row = {'history_id': 'h1', 'R_superseded_minus_R_irrelevant_counterbalanced': 1.0,
           'R_live_minus_R_superseded': 0.0}
    report = {'history_rows': [row], 'statistics': {'results': {
        'R_superseded_minus_R_irrelevant_counterbalanced': summary([1.0]),
        'R_live_minus_R_superseded': summary([0.0]),
    }}}
    data = [{'history_id': 'h1', 'attribute': 'badge', 'prompt_variant': 'a', 'prompt_family': 'f'},
            {'history_id': 'h1', 'attribute': 'code', 'prompt_variant': 'a', 'prompt_family': 'f'}]
    with pytest.raises(ValueError, match='changes within a history'):
        v1_attribute_frame_summaries('model', report, data)
