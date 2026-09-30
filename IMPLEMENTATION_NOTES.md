# Implementation notes

## Repository and runtime audit

- The target directory contained no source files, project metadata, tests, or Git repository at implementation start. Git commit provenance is therefore recorded as `null` until the directory is placed in a repository.
- The neighboring `../llm-probing` project was inspected for established practices. Its explicit provenance snapshots, JSONL datasets, lazy Hugging Face loading, block-hook convention, and pytest use informed this implementation. Its task-specific activation code was not copied because it captures a different query and model protocol.
- Runtime audit: Python 3.14.4; NumPy 2.4.3; scikit-learn 1.8.0; pandas 3.0.1; matplotlib 3.10.8; PyYAML 6.0.3; pytest 9.0.3. PyTorch, Transformers, datasets, and safetensors are absent. CUDA and MPS checks could not be made through PyTorch because PyTorch is not installed. No model was downloaded or executed here.
- Runtime manager: uv 0.9.4. The project is now locked in `uv.lock` (81 resolved packages), and `.python-version` selects Python 3.13. `uv lock --check` succeeds. Dependencies were not installed into a uv environment during migration; the existing unit test run was under the pre-existing Python 3.14.4 environment.
- No experiment result is reported. Verification here is limited to source compilation and CPU synthetic/unit tests. Model-dependent tokenization, behavioral competence, activation extraction, GPU patching, and empirical plots remain to be run on the work PC.

## Design assumptions

- Four symbolic assignment events use all six legal line orders, with old-before-current precedence for each variable. Variable names are randomly remapped to abstract query/distractor roles; literal `x` is not assigned a semantic privilege.
- A pilot audit found that the earlier generator tied order, query, template, and whitespace to the same example index. The generator now crosses all `6 × 2 × 3 × 2` order/query/template/format cells once per 72 contexts and balances variable-name pairs independently. Earlier pilot results remain exploratory; paired `O_q` versus `O_d` interventions were within context, but the pilot did not independently sample prompt factors. `configs/behavioral_replication.yaml` freezes a 360-context follow-up with a new seed.
- The split group key is `(task family, unordered variable-name pair, exact legal line-order pattern)`. Counterfactual members share the base context split. Dataset values are drawn without replacement within each context.
- Token validation is performed over the exact rendered prompt/chat prefix followed by the proposed answer continuation. It checks both one-token continuation and unchanged prefix tokenization. Accepted value-to-token-ID mapping is written separately.
- Activation storage is query-position only, chunked, and half-precision. The embedding/pre-block-0 vector and each block output are distinct entries. Patching targets block output, never modifies the obsolete token representation retrospectively.
- Fast tokenizer offset mappings are required for the activation patch sweep to prove that token differences are inside the intended assignment-value span. Architectures without discoverable decoder blocks or tokenizer offsets fail explicitly.
- Probe regularization is selected using validation accuracy; train-only scaling and test-only final reporting are used. Probe seeds are configurable in the API. Layer curves are descriptive; no layer-wise significance claims are emitted.

## Known limits / follow-up

- Pilot and primary YAML values are candidate suggestions only. `validate_tokens.py` may reject values depending on the actual tokenizer/chat template; it records exact accepted IDs. The full pipeline validates first, then restricts dataset generation to accepted candidates. If fewer than four distinct stable candidates survive, validation fails and the value list must be expanded or replaced.
- Current split assignment balances by shuffled groups but does not guarantee exact proportions when group sizes differ; the partition remains group-disjoint.
- The current analysis utility summarizes candidate-restricted distribution changes and B/R/M distributions. Cluster-bootstrap summaries exist as a reusable helper; the current simple aggregate script uses example-level intervals for single-trial distributions. The paired difference-in-differences section should be interpreted as context-paired only where matched role-pair records are present.
- Natural-language generation and symbolic generation preserve the same abstract state structure, but the natural-language templates are narrow controlled paraphrases rather than a broad QA benchmark.
- Large run resource use has not been measured. The patch sweep forwards each layer × input position and is intended for a small pilot subset. Activation files are chunked to avoid all-token/all-example caches.
- Before scientific conclusions: run direct and overwrite competence first; examine all-example and correct-trial summaries; verify current-binding patching positive control; retain raw data/provenance; compare obsolete-query with matched obsolete-distractor interventions.

## RTX 5080 inference configuration

- Pilot and primary configs now load Qwen3-8B with bitsandbytes LLM.int8, FP16 for non-quantized modules, `device_map: auto`, and PyTorch SDPA for attention. SDPA is an attention backend, not a weight quantizer; it complements int8 loading.
- The experiment PC must have a CUDA/PyTorch runtime compatible with the installed bitsandbytes wheel. bitsandbytes currently lists SM120 builds for CUDA 12.8+ on Linux x86-64 and Windows x86-64. This host has no PyTorch/CUDA, so warning behavior and speed on the RTX 5080 remain unverified.
- Dtype is explicitly FP16 to avoid the usual BF16-to-FP16 conversion path inside bitsandbytes LLM.int8. Warnings are not suppressed; if any remain, record the exact warning and package/runtime versions.

## Paired behavioral controls/status scoring

- `supersession_behavior_v1` explicitly crosses both variables, both current queries, baseline/edit directions, and all control/status conditions. One `history_id` identifies a matched unit across conditions, including YES/NO. The old history-only controls did not specify z-role values or complete counterfactual pairs; they cannot be upgraded by parsing prompts and must be regenerated in a fresh artifact.
- Generation can restrict proposals using the existing validated token map. The 36-value proposal JSON is not itself a validated vocabulary; the frozen Qwen3 map currently contains 12 values. Exact new prompt/chat continuations and single-token edits are re-audited before any inference.
- Scoring calls the shared behavioral scorer with a metadata-driven renderer and preserves the complete input record. Existing JSONL checkpoint utilities validate dataset/token-map/config fingerprints; the new runner also includes resolved revisions, scoring code, chat template, and runtime versions. Provenance is saved before the first score, and edited records retain per-pair logit components.
- All primary summaries use complete, auditable pairs regardless of correctness and bootstrap matched histories. Raw effects, x/z contrasts, symmetric means, and direct within-history differences are retained. No ratios or automatic competence filtering are used.
- Unassigned mentions are not variable bindings. The irrelevant contrast uses first/second mention slots as explicit analytic x/z labels, with matching source values/replacements across conditions. Controls have unequal event counts/positions; accepted/rejected histories offer the more closely matched lexical status comparison. Obsolete decision diagnostics save both semantic-role margins and fixed-baseline-token margins, with explicit before/after identities and edited-answer logits/ranks when a valid-current edit changes the correct answer.
- No pretrained model was loaded or run locally for this implementation. Verification uses synthetic scored records and deterministic scoring stubs; no behavioral result is claimed.
