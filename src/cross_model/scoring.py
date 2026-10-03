"""Bounded surface-class continuation mass; prefix probabilities without termination."""
import math

from src.cross_model.protocol import VALUES
from src.cross_model.protocol import digest
from src.cross_model.tokens import continuations, encode
from src.data.supersession_behavior import render_behavior_example


def logsumexp(values):
    if not values or any(math.isnan(v) or v == math.inf for v in values):
        raise ValueError('likelihood class is empty or contains NaN/+infinity')
    # A -infinity log probability is a valid zero-mass event. Ignore it when
    # another surface in the same semantic class has nonzero probability.
    values = [v for v in values if v != -math.inf]
    if not values:
        raise ValueError('all surface events in likelihood class have zero probability (-infinity)')
    maximum = max(values)
    return maximum + math.log(sum(math.exp(v-maximum) for v in values))


def score_prompt(model, tokenizer, prompt, events, hook_factory=None, *, values=None,
                 progress_callback=None):
    """Sum log p(token | prompt, previous candidate tokens), without length normalization.

    hook_factory is entered for every forward, including teacher-forced suffixes.
    Only prompt positions are patched; no candidate token is ever patched.
    values restricts evaluated classes without renormalizing their probabilities.
    Token sequences are answer prefixes, not exact terminated answers.
    """
    import torch
    from contextlib import nullcontext
    if values is not None:
        values = tuple(values)
        if not values or len(set(values)) != len(values) or any(v not in events for v in values):
            raise ValueError('requested scoring values must be distinct members of the event map')
        events = {v: events[v] for v in values}
    prefix = encode(tokenizer, prompt)
    device = model.get_input_embeddings().weight.device
    context = hook_factory or nullcontext
    def forward(ids):
        with context(), torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0].float()
        if not torch.isfinite(logits).all():
            bad = int((~torch.isfinite(logits)).sum().item())
            raise ValueError(
                f'model produced {bad} nonfinite logits for prompt_sha256={digest(prompt)} '
                f'input_length={len(ids)}; check model compute dtype/quantization')
        if progress_callback is not None:
            progress_callback()
        return logits
    initial = forward(prefix)[-1]
    initial_lp = initial.log_softmax(-1)
    surface_likelihoods, masses = {}, {}
    for value, members in events.items():
        likelihoods = []
        for event in members:
            ids = event['ids']
            if len(ids) == 1:
                likelihood = float(initial_lp[ids[0]])
            else:
                logits = forward(prefix + ids[:-1])
                positions = logits[len(prefix)-1:len(prefix)+len(ids)-1].log_softmax(-1)
                likelihood = float(positions[torch.arange(len(ids), device=positions.device),
                                             torch.tensor(ids, device=positions.device)].sum())
            if math.isnan(likelihood) or likelihood == math.inf:
                raise ValueError(
                    f'invalid continuation likelihood prompt_sha256={digest(prompt)} '
                    f'value={value!r} surface={event["text"]!r} token_ids={ids!r} '
                    f'log_probability={likelihood!r}')
            likelihoods.append(likelihood)
        surface_likelihoods[value] = [{'text': e['text'], 'ids': e['ids'], 'log_probability': lp}
                                      for e, lp in zip(members, likelihoods)]
        try:
            masses[value] = logsumexp(likelihoods)
        except ValueError as error:
            event_summary = [{'text': e['text'], 'ids': e['ids'], 'log_probability': lp}
                             for e, lp in zip(members, likelihoods)]
            raise ValueError(
                f'{error}; prompt_sha256={digest(prompt)} value={value!r} '
                f'events={event_summary!r}') from error
    greedy = int(initial.argmax())
    return masses, surface_likelihoods, initial, greedy


def score_row(model, tokenizer, row, candidate, competence_only=True, progress_callback=None,
              *, renderer=render_behavior_example):
    prompt = renderer(row, tokenizer, True)
    events = continuations(tokenizer, prompt)
    if events != candidate['events']: raise ValueError('candidate map mismatch at actual scoring prefix')
    masses, surfaces, logits, greedy = score_prompt(model, tokenizer, prompt, events,
                                                     progress_callback=progress_callback)
    ids = candidate['canonical_one_token_ids']; target_id = ids.get(row['answer'])
    canonical_first = encode(tokenizer, prompt+' '+row['answer'])[len(encode(tokenizer, prompt))]
    rank = 1 + sum(v >= masses[row['answer']] for k, v in masses.items() if k != row['answer'])
    return {**row, 'prompt': prompt, 'score_kind': 'competence_only' if competence_only else 'confirmatory',
            'semantic_log_mass': masses, 'surface_likelihoods': surfaces,
            'semantic_rank': rank, 'semantic_accuracy': int(rank == 1),
            'greedy_token_id': greedy, 'generated_first_token': tokenizer.decode([greedy]),
            'full_vocab_next_token_accuracy': int(target_id == greedy) if target_id is not None else 0,
            'exact_token_diagnostic_defined': target_id is not None,
            'canonical_first_token_accuracy': int(canonical_first == greedy),
            'full_vocab_rank': int((logits > logits[canonical_first]).sum())+1,
            'canonical_candidate_logits': {v: float(logits[i]) for v, i in ids.items()},
            'full_vocab_exact_token_note': 'undefined multi-token targets are marked; never treated as gate failures'}
