# Analysis amendment: relational v2 interpretation rule

**Recorded:** 2026-10-03 14:18 UTC  
**Applies to:** `factorial_relation_counterbalanced_20261003`  
**Status:** recorded before v2 confirmation results are examined. This is a dated repository analysis amendment, not an external preregistration.

## Confirmatory interpretation

Evaluate each model separately. Do not pool raw nat values across models.

For a model, the full three-control claim is supported only when the lower bound of the 95% history-bootstrap interval is above zero for **each** of these prespecified contrasts:

1. `R_superseded − R_early_unassigned`
2. `R_superseded − R_entity_mention`
3. `R_superseded − R_other_attribute`

The intervals use the frozen v2 history-level bootstrap (2,000 draws, seed 73021). If any contrast does not meet the rule, report the interval and support status for each contrast and narrow the model-specific claim to the contrasts that meet it. Do not describe the composite three-control claim as supported when one or more controls fail the criterion.

A claim that the effect holds under reversed order requires the **reversed-order version of each claimed contrast** to have a 95% history-bootstrap lower bound above zero. Reversed order uses the frozen aggregation of historical/current order cells `(0,1)` and `(1,0)` before computing history-level R and its contrast. Overall contrasts do not substitute for this reversed-order criterion.

Report model-specific estimates and intervals. Do not select wording or claims from pooled raw nats.

## Provenance and freeze handling

This amendment supplements the frozen design; it does not change the contract, renderer, configs, gates, thresholds, datasets, tokenizer artifacts, or protocol freeze. The repository freeze checks the contract, protocol code, configs, and reviewed prompt audit. It does not include this documentation file in its code/config hash. Preserve this amendment as a separate dated record and cite its SHA-256 in any future confirmatory reporting sidecar or analysis packet. Do not rewrite or regenerate the existing protocol freeze to include it.
