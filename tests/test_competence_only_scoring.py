import json

from scripts import run_supersession_behavior as scoring


def test_competence_only_scoring_skips_matched_effect(tmp_path, monkeypatch):
    rows = [
        {'example_id': 'pair:0', 'pair_id': 'pair', 'pair_direction': 0},
        {'example_id': 'pair:1', 'pair_id': 'pair', 'pair_direction': 1},
    ]

    def score_example(model, tokenizer, row, token_ids, **kwargs):
        return {**row, 'prompt': f"prompt-{row['pair_direction']}"}

    def forbidden_effect(*args):
        raise AssertionError('causal effect must not be computed in competence-only mode')

    monkeypatch.setattr(scoring, 'score_example', score_example)
    monkeypatch.setattr(scoring, 'matched_edit_effect', forbidden_effect)
    output = tmp_path / 'scores.jsonl'

    scoring.score_dataset(None, None, rows, {}, output, set(), competence_only=True)

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 2
    assert all('matched_edit_effect' not in row for row in records)
