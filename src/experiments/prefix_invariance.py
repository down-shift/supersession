"""Shared equal-shape batched causal-prefix invariance audit."""
from __future__ import annotations

import torch

from src.data.generate import render_example
from src.experiments.patching import capture_run


def audit_prefix_invariance(model, tokenizer, examples_by_query, queries,
                            chat=True, behavior_records=None, tolerance=1e-4):
    """Audit old-value residual equality with all query variants in one batch.

    ``examples_by_query`` maps query IDs to frozen pair records. Returns a JSON
    serializable report and raises on any prompt, token-prefix, or residual error.
    """
    encodings, positions, prompts = {}, {}, {}
    for query in queries:
        ex = examples_by_query[query]
        prompt = render_example(ex, tokenizer, chat=chat)
        prompts[query] = prompt
        if behavior_records is not None:
            prior = behavior_records.get(ex['example_id'])
            if prior is None or prior.get('prompt') != prompt:
                raise ValueError(f"frozen rendered prompt mismatch for {ex['example_id']}")
        enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        try:
            start = prompt.index(ex['old_x'])
        except ValueError as exc:
            raise ValueError('token IDs through old-x value differ across query variants') from exc
        positions[query] = [i for i, (lo, hi) in enumerate(enc['offset_mapping'])
                            if lo < start + len(ex['old_x']) and hi > start]
        if not positions[query]:
            raise ValueError(f"old-x token span not found for {ex['history_id']}/{query}")
        encodings[query] = list(enc['input_ids'])
    refpos = positions[queries[0]]
    if any(positions[q] != refpos for q in queries[1:]):
        raise ValueError('historical-value token positions differ across query variants')
    end = max(refpos) + 1
    prefix = encodings[queries[0]][:end]
    if any(encodings[q][:end] != prefix for q in queries[1:]):
        raise ValueError('token IDs through old-x value differ across query variants')
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError('tokenizer requires pad or EOS token for batched prefix audit')
    width = max(map(len, encodings.values()))
    ids = torch.full((len(queries), width), int(pad), dtype=torch.long)
    mask = torch.zeros((len(queries), width), dtype=torch.long)
    for i, q in enumerate(queries):
        ids[i, :len(encodings[q])] = torch.tensor(encodings[q], dtype=torch.long)
        mask[i, :len(encodings[q])] = 1
    device = next(model.parameters()).device
    _, captures = capture_run(model, {'input_ids': ids.to(device), 'attention_mask': mask.to(device)})
    per_layer = []
    for layer, activations in sorted(captures.items()):
        ref = activations[0, refpos, :]
        diffs = {q: float((activations[i, positions[q], :] - ref).abs().max().item())
                 for i, q in enumerate(queries[1:], 1)}
        maximum = max(diffs.values())
        per_layer.append({'layer': int(layer), 'max_abs_difference': maximum,
                          'max_by_query': diffs, 'passed': maximum <= tolerance})
    result = {'queries': list(queries), 'tolerance': tolerance,
              'execution': 'one right-padded equal-shape batch',
              'verified_identical_token_prefix_through_old_x': True,
              'per_layer': per_layer, 'passed': all(x['passed'] for x in per_layer)}
    if not result['passed']:
        raise ValueError(f'prefix-invariance residual difference exceeded {tolerance}')
    return result
