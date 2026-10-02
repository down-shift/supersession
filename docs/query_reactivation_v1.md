# Query-dependent reactivation of superseded bindings

## Scientific question

After a successful update `x = a; later x = b`, does the same obsolete contextual binding `a` have different causal relevance when the question asks for the current value versus the original value? The established four-query result is that obsolete bindings retain measurable causal relevance after successful updating. This experiment asks whether that relevance changes with query relevance.

The established current-query result is the central finding: superseded bindings remain causally relevant during current-state retrieval. A historical query is a secondary addressability positive control. In this design, editing the old binding also changes the correct answer to that historical query, so a larger historical effect is expected when the question directly requests that value. Treat the contrast as evidence that the old value remains functionally retrievable, not as the paper's main novelty or a surprising mechanism by itself. Query-conditioned state retrieval also has mechanistic precedent, including [Prakash et al.](https://arxiv.org/abs/2505.14685).

The recommended study order is **cross-model confirmation → query-reactivation as secondary evidence → downstream-transfer experiment → writing**. A downstream-transfer test should hold the correct/current answer fixed while asking whether stale information changes another computation; that is the higher-priority test of functional consequences. This protocol does not implement that experiment. For context, see [When Context Changes](https://arxiv.org/abs/2609.38866).

Do not infer erasure, suppression, a validity representation, or a dedicated circuit.

## Estimands

For each history, use the existing matched `identity_transfer` estimand: the baseline-to-edited change in replacement-minus-source candidate log odds. Positive values mean the edit causally transfers preference toward the replacement.

For current and historical queries of both variables:

```text
R_old_current = 0.5 * ((E(old_x,current_x)-E(old_x,current_z))
                     + (E(old_z,current_z)-E(old_z,current_x)))
R_old_historical = 0.5 * ((E(old_x,initial_x)-E(old_x,initial_z))
                         + (E(old_z,initial_z)-E(old_z,initial_x)))
Delta_reactivate = R_old_historical - R_old_current
```

This orientation uses replacement-minus-source log odds, so the positively oriented query is the query for the edited binding’s answer. The primary unit is the history; the x/z values are averaged within history before inference. The confirmation’s primary contrast is the paired history-level `Delta_reactivate`. Incorrect task trials remain in the primary analysis. Competence is reported separately; correct-only summaries, if added later, are sensitivity analyses only.

The existing four-query analysis also reports current-binding query selectivity as a sanity control. The fresh confirmatory dataset edits obsolete x/z bindings only, so that control is not estimable there.

## Exploratory versus confirmatory

The existing four-query reanalysis is **POST-HOC / EXPLORATORY**. It was not preregistered, does not alter or overwrite the established artifacts, and must not be relabeled confirmatory. Its code and outputs are marked accordingly.

`query_reactivation_v1` is a separate frozen fresh-history confirmation. It uses the existing natural-language `nora_v1` assignment/update semantics, with fixed x/z roles and the query changed between current and original retrieval. The candidate vocabulary, template, stage seeds, sample counts, estimands, bootstrap unit, and competence threshold are protocol constants. Development and frozen-gate scoring expose competence only; no causal effect is calculated or available to the gate. Confirmation requires a passing gate and disjoint concrete histories. Scoring is resumable and writes provenance sidecars; outputs are create-only.

## Dataset and gate protocol

Each history has four unedited questions: current x, original x, current z, original z. Confirmation adds matched baseline/edit pairs for both obsolete bindings under all four queries. Replacement values are shared across queries for a binding and distinct across x and z. Histories are sampled fresh and checked against every supplied prior dataset, including mirrored x/z orientations. Do not tune prompts, values, or thresholds using causal confirmation scores.

The frozen competence gate requires at least 97% full-vocabulary next-token accuracy in each aggregate task: current queries pooled over x/z and orientations, and historical queries pooled the same way. This is frozen at `thresholds.gate_accuracy: 0.97` in `configs/query_reactivation_v1.yaml` and must agree with the scorer. With 24 histories this gives 48 examples per task and tolerates one miss. Query × variable × orientation cells are reported as diagnostics and do not veto the gate. The sole frozen candidate vocabulary is `configs/query_reactivation_values.json`; the generator default, protocol constant, and saved dataset must match it. The candidate-value list and its hash are sealed into the gate and must match at confirmatory generation and scoring. Historical questions explicitly ask what the value was originally; current questions explicitly ask what it is currently. The scorer uses the repository candidate/full-vocabulary scoring path, not string-prefix accuracy.

## Commands

The commands below assume the repository’s validated Qwen token map and pinned config. Run generation and scoring on separate fresh output paths. Inference is intentionally not run as part of implementation.

Post-hoc analysis of the existing local four-query results:

```bash
.venv/bin/python scripts/analyze_query_reactivation.py \
  --behavior outputs/four_query_288/behavior.jsonl \
  --pairs outputs/four_query_288/pair_behavior.jsonl \
  --output-dir outputs/four_query_288/query_reactivation_posthoc_rerun
```

Fresh development and frozen competence gate:

```bash
.venv/bin/python scripts/generate_query_reactivation.py --stage development \
  --values-json configs/query_reactivation_values.json \
  --prior-dataset outputs/four_query_288/behavior_inputs.jsonl \
  --output outputs/query_reactivation_v1/development.jsonl
.venv/bin/python scripts/run_query_reactivation.py \
  --dataset outputs/query_reactivation_v1/development.jsonl \
  --token-ids outputs/four_query_288/token_ids.json --config configs/query_reactivation_v1.yaml \
  --output outputs/query_reactivation_v1/development_scores.jsonl
.venv/bin/python scripts/generate_query_reactivation.py --stage frozen_gate \
  --values-json configs/query_reactivation_values.json \
  --prior-dataset outputs/four_query_288/behavior_inputs.jsonl \
  --prior-dataset outputs/query_reactivation_v1/development.jsonl \
  --output outputs/query_reactivation_v1/frozen_gate.jsonl
.venv/bin/python scripts/run_query_reactivation.py \
  --dataset outputs/query_reactivation_v1/frozen_gate.jsonl \
  --token-ids outputs/four_query_288/token_ids.json --config configs/query_reactivation_v1.yaml \
  --output outputs/query_reactivation_v1/frozen_gate_scores.jsonl
```

The gate artifact is `outputs/query_reactivation_v1/frozen_gate_scores.jsonl.gate.json`. If either aggregate task fails, confirmation is prohibited. If both pass:

```bash
.venv/bin/python scripts/generate_query_reactivation.py --stage confirmatory \
  --values-json configs/query_reactivation_values.json \
  --prior-dataset outputs/four_query_288/behavior_inputs.jsonl \
  --prior-dataset outputs/query_reactivation_v1/development.jsonl \
  --gate outputs/query_reactivation_v1/frozen_gate_scores.jsonl.gate.json \
  --output outputs/query_reactivation_v1/confirmatory.jsonl
.venv/bin/python scripts/run_query_reactivation.py \
  --dataset outputs/query_reactivation_v1/confirmatory.jsonl \
  --token-ids outputs/four_query_288/token_ids.json --config configs/query_reactivation_v1.yaml \
  --gate outputs/query_reactivation_v1/frozen_gate_scores.jsonl.gate.json \
  --output outputs/query_reactivation_v1/confirmatory_scores.jsonl
.venv/bin/python scripts/analyze_query_reactivation.py --stage confirmatory \
  --behavior outputs/query_reactivation_v1/confirmatory_scores.jsonl \
  --pairs outputs/query_reactivation_v1/confirmatory_scores.jsonl \
  --output-dir outputs/query_reactivation_v1/confirmatory_analysis
```

Resumption uses the exact original scoring command plus `--resume`; the checkpoint fingerprint rejects changed datasets, token maps, or config. Preserve failed runs and use new paths for revisions.

## Expected artifacts

- `query_reactivation_per_history.csv`: all primary history-level estimands.
- `query_reactivation_summary.json`: sample sizes, mean, median, 10% trimmed mean, positive fraction, history bootstrap 95% CI, paired sign-flip permutation p-value, and competence cells.
- Stage JSONL datasets, score JSONL checkpoints, `.run.json` and `.provenance.json` sidecars, plus the sealed frozen-gate JSON.

## Interpretation matrix

| Outcome | Interpretation |
|---|---|
| `R_historical > R_current > 0` | Superseded bindings remain causally relevant during current-state retrieval and become much more causally influential when explicitly queried historically. Interpret the historical effect as a positive control for continued addressability; editing the old value also changes the correct historical answer. |
| `R_historical > 0`, `R_current ≈ 0` | The old value remains addressable when requested, while this experiment provides little evidence of causal relevance during current-state retrieval. |
| `R_historical ≈ R_current > 0` | Causal persistence exists, with little evidence of query modulation. |
| `R_historical ≈ 0` | Historical addressability is unsupported, or the historical task/instrument is inadequate. Inspect historical-query competence before interpreting the result. |

All interpretations require adequate current and historical competence. A failed competence gate makes causal interpretation inadequate; it does not license dropping incorrect histories.
