# Supersession Without Erasure: cross_model_v1

Status: preregistration-ready implementation, **not a completed replication or externally registered protocol**. No new model competence or confirmatory causal inference has run. Read [the preimplementation audit](cross_model_v1_audit.md) first. Established Qwen experiments, failed Mistral/Phi gates, and stopped version-chain/status branches remain preserved.

## Frozen scientific design

Use the existing `nora_v1` entity–attribute semantics and generator: live, superseded, ordinary irrelevant (diagnostic), and both literal orders of counterbalanced irrelevant. The same semantic histories, replacement identities, seeds, names, attributes, and orientations are used across model families. The assistant suffix remains `Answer:`; chat rendering uses the pinned tokenizer, `add_generation_prompt=True`, `enable_thinking=False`. No stronger classification instruction or candidate list is added to the prompt.

The contract in `src/cross_model/protocol.py` is copied exactly into four new configs. Config mismatches fail; thresholds/seeds/candidates have no CLI override. Stages use 24/24/96 histories and seeds 20261202/20261203/20261204; tokenizer validation uses 20261201. Three focal conditions yield exactly **64 required competence cells** per template: live 16, superseded 16, counterbalanced irrelevant 32. Each cell counts each unique prompt once. Each history has 40 scored members / 20 edit pairs; confirmatory has 3,840 members / 1,920 pairs. No causal analysis is available for development/gate stages.

## Competence decision and alternatives

| Option | Decision | Reason and limitation |
|---|---|---|
| A: lowercase candidate-token ranking | Do not use by itself | Would retain tokenizer/case-dependent incompleteness; a fragment may outrank every validated lowercase token. Ranking remains the decision rule **after** semantic classes are scored. |
| B: predefined surface classes | Use | Lowercase/title-case crossed with zero/one leading ASCII space. Rules fixed before any new logits. Tokenizer-only exclusions and event deduplication are recorded. This is a bounded equivalence class, not arbitrary normalization. |
| C: forced labels/JSON/classification | Decline for v1 | Could improve compliance but adds a stronger classification task and changes retrieval demands. It would require a separate protocol. |
| D: complete sequence likelihood | Use | Score all tokens of each eligible surface, without length normalization. Supports different token counts and avoids treating `Bron` or ` G` as complete semantic answers. Costs more forwards; prioritizes exactness over throughput. |

For prompt token prefix p and token continuation c:

```text
L(c | p) = sum_t log P(c_t | p, c_<t)
S(value | p) = logsumexp_{distinct eligible continuations c for value} L(c | p)
```

Required spaced lowercase/title-case strings must preserve p under joint encoding. Unspaced proposals are included only if joint encoding preserves p. Identical token events are counted once, even if multiple text proposals encode identically. Cross-value token collisions or any prefix-nested token events fail validation, so the class mass is a sum of disjoint continuation events. No length normalization, case correction, probability averaging, max-over-variants, punctuation stripping, or post-result surface additions. Scoring is the complete candidate continuation **without an added EOS/delimiter**; continuation prefixes outside the bounded answer universe remain a limitation. Alternative token sequences decoding to the same word are not exhaustively enumerated.

Per required cell:

- semantic candidate accuracy >= .99;
- mean semantic rank <= 1.01; ties count against rank one;
- mean `S(current)-S(stale) > 0` where stale exists;
- complete expected cells, exact metadata/ID matching, finite scores, fresh stage histories.

The same fixed criteria screen development. Development failure stops before the gate. Full-vocabulary exact next-token accuracy, canonical first-fragment accuracy/rank, raw greedy token, first-token surface parsing, casing-miss diagnostics, and per-surface likelihoods remain reported. **Exact one-token accuracy is undefined for multi-token targets**, recorded with an eligibility count/null summary; first-fragment correctness is not substituted for it. Lowercase single-token raw logits remain saved when available. The gate recomputes semantic accuracy/rank/margin from saved likelihoods rather than trusting stored accuracy or `pass` fields.

## Vocabulary policy

Fixed shared semantic vocabulary:

```text
amber coral jade pearl slate teal violet ivory
```

Shared meanings and identical histories take priority over one-token matching. Model-specific vocabularies would alter lexical distributions; they are not allowed. The single-token intersection across cached Qwen/Mistral/Phi has only seven values and does not justify searching for unusual words. The largest historical-span-length signature group in the existing proposal pool had 18 values; eight familiar colors were chosen within it using tokenizer/format evidence only. [The audit](cross_model_v1_audit.md) preserves the initial incompatible eight-value proposal and the tokenizer-only refinement. No new model logits informed this selection.

Final historical span lengths are Qwen=1, Mistral=2, Phi=1. Exact sequence IDs, casing proposals, exclusions, tokenizer JSON hash, template hash, actual prompt example, every paired edit, and exhaustive candidate substitutions in each represented assignment slot are saved in Stage 0 maps. All values must remain valid; no per-model replacements or truncation. Llama must validate this exact vocabulary; failure is a documented exclusion, not a vocabulary retune. Stage generation repeats audits on the actual dataset. Counterfactual text changes must leave all tokens outside the edited semantic value span identical. The core mechanistic battery additionally requires equal lengths/positions across each donor/recipient pair and patches **all** value-span tokens jointly.

## Model panel fixed on lineage and intervention feasibility

Four planned families, all instruction/chat variants. Model and tokenizer use the same immutable revision in each row; configs also pin tokenizer ID explicitly. Estimates below are engineering expectations for batch-one short histories, not measured RTX 5080 benchmarks. Exact memory/device placement must be recorded on the CUDA host.

| Config / exact model ID | Model and tokenizer revision | Architecture / hidden / layers | Query/KV heads; head dim | Expected working memory, int8 | Execution caveats |
|---|---|---|---|---|---|
| `qwen3_8b`: `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | Qwen3 / 4096 / 36 | 32/8; 128 | roughly 10–13 GiB | Existing loader supported; native code; disable thinking via exact template. |
| `mistral7b`: `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | Mistral / 4096 / 32 | 32/8; 128 | roughly 8–11 GiB | Existing loader supported; native code; `[INST]` template and surface/subword differences; canonical answers now two tokens. |
| `phi4_mini`: `microsoft/Phi-4-mini-instruct` | `cfbefacb99257ffa30c83adab238a50856ac3083` | Phi3 implementation / 3072 / 32 | 24/8; 128 | roughly 5–8 GiB | Pinned remote code and existing compatibility shims; fused qkv; LongRoPE; old run used BF16/no quantization, new panel uses int8/FP16. |
| `llama31_8b`: `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | Llama / 4096 / 32 | 32/8; 128 | roughly 10–13 GiB | Native loader path and adapters; gated access unresolved; header template; pinned tokenizer/config validation still pending. |

Specifications are from [pinned Qwen config](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json), [pinned Mistral config](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/raw/c170c708c41dac9275d15a8fff4eca08d52bab71/config.json), [pinned Phi config](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/cfbefacb99257ffa30c83adab238a50856ac3083/config.json), and [Meta's architecture registry](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py). The pinned Llama file returned HTTP 401 during this audit, so registry specifications are provisional until authenticated Stage 0. Approximate memory estimates follow weight size plus full-precision embeddings/output heads, quantization metadata, logits, and activations; do not assume the same footprint for equal parameter counts.

All new configs request float16, bitsandbytes int8, eager attention, batch-one execution, `device_map=auto`. Actual dispatch is recorded; CPU offload is a technical caveat, not a silent equivalence to fully GPU execution. Do not replace an inaccessible/failed model based on causal effects. If Llama access fails, the preregistered three-family panel remains Qwen/Mistral/Phi. Gemma is deferred because its block normalization/ordering needs a separate adapter audit; v1 rejects it. No random model substitutions.

Residual/block, post-projection attention, and MLP hooks are explicit for Qwen3/Llama/Mistral/Phi3. Native tiny-model tests pass for each family. **Native Phi3 tests do not certify the pinned remote Phi implementation**, and local CPU tests do not certify bitsandbytes/CUDA. Before intervention work, `smoke` and the battery's mandatory runtime smoke check actual module order, dimensions, captures, activation dtypes, and no-op/self-patch logit parity.

## Causal estimands and comparability

Retain the replacement-minus-source and query-specific identity-transfer structure:

```text
E_sequence = [S(replacement)-S(source)]_edited - [S(replacement)-S(source)]_baseline
R_condition = 0.5 * sum_{variable in x,z} [E(associated query)-E(other query)]
primary = R_superseded - R_irrelevant_counterbalanced
secondary = R_live - R_superseded
```

Average counterbalanced irrelevant slot orders **before** relevance contrasts. Ordinary irrelevant remains diagnostic. All histories/trials remain in causal estimates, including any errors during confirmation; do not filter on observed competence or effect sign.

This is an explicit **new sequence/surface-mass estimand**, version `cross_model_sequence_mass_v1`, measured in log probability odds (nats). It is not silently substituted for Qwen's old raw single-token effect. The existing estimator is separately recomputed on canonical raw logits when the **entire shared universe** is one-token validated (currently Qwen/Phi, not Mistral). For single-token singleton classes, sequence log-odds differences reduce exactly to raw logit differences. Multiple surface variants/multiple tokens change the estimand. Qwen must undergo the same new protocol to bridge those measurements; compare neither old 4.52 logits nor its old vocabulary directly to the new Mistral sequence estimate.

Raw logit and log-probability odds magnitudes are not calibrated across architectures; subtracting log-normalizers does not solve temperature/training-scale differences. Report each model's raw effects and sign consistency separately. Secondary dimensionless retention is the **ratio of model means**:

```text
mean(primary) / mean(R_live)
```

Compute a paired history-bootstrap ratio only if the live-effect 95% bootstrap lower bound exceeds **1 nat**, and every denominator among the 2,000 fixed bootstrap draws also exceeds 1. Otherwise report unavailable with a reason. This eligibility rule governs metric interpretation, never model inclusion. Do not average per-history ratios. The ratio need not be in [0,1] and is not an amount of stored information. Do not use z-scores over vocabulary as a new primary estimand, probability effects as temperature-free measures, or ranks as magnitude substitutes.

## Minimal common mechanistic battery

Gate competence and complete confirmatory behavioral scores are required. A weak/null behavioral causal effect **does not** exclude a model from mechanisms.

Use the first **12** confirmatory history IDs in fixed lexicographic order, spanning both orientations. Patch both counterfactual donor directions and average donor-oriented deltas. No outcome-based histories, layers, sites, precision, or heads enter the core battery. Additional superseded **current-value** edit pairs are generated deterministically from those same semantic histories, using their already-fixed replacement identities; no extra prompt tuning.

Semantic positions are found from exact renderer text plus tokenizer offsets, never token-ID search: edited value span; historical value span (only obsolete in superseded); current value span; distractor/current other-variable span; queried entity span; final pre-answer token. Multi-token spans are patched jointly, all tokens, with equal pair alignment. Missing/ambiguous spans or unequal lengths stop the battery without dropping pairs. Donor activations are cast to recipient device/dtype; hidden activations themselves are not quantized by the patch operation.

- **M1/M2**: block residual outputs at **all layers × every represented semantic position**, historical-token and readout localization. Depth is `layer/(num_layers-1)`. Results retain exact layers and normalized coordinates.
- **M3**: fixed early depth <=.25 and late depth >=.75. At each history, average layer effects within windows, then compute `early_historical_minus_readout` and `late_readout_minus_historical` in superseded initial-value cells. Bootstrap histories, not layers. Report both contrasts and full maps; evidence for both preferred sites supports the qualitative depth shift. This is not literal information movement.
- **M4**: attention module output **after output projection, before residual addition**, and MLP output before residual addition, at final pre-answer only, nearest layers to depths .75/.875/1. Within-block order is attention → residual add → MLP → residual add. Component patches are not an additive mediation decomposition.
- **M5 optional**: pre-o_proj query-head results. At the same three late depths, discovery uses the first six histories' stale edits; choose exactly two heads with largest absolute mean query-specific patch relevance (ties: layer/head). Profile on the disjoint final six histories' stale, historical/live, and current bindings. Heldout head profiles are recorded separately; negative effects are not suppression. Full raw head scans are saved, but reserve values never determine selection.

The core battery computes paired within-history query-specific patch relevance and all counterbalanced controls. It tests functional localization under interventions. Matching normalized depth or component class is not neuron/head homology.

## Quantization validity

Old Qwen experiments used int8 with float16 activations. bitsandbytes replaces linear weight computation, including attention/MLP projections, while hook tensors remain floating point; changing weights can still change their content. The runtime smoke records actual dtypes rather than assuming FP16. Eager attention standardizes exposed projection/head paths across the new panel but differs from historical Qwen's requested SDPA, so the new Qwen run is required.

Fixed sensitivity panel: Qwen8 and Phi4-mini, first six confirmatory histories, int8 versus unquantized float16 at the same immutable model revision. Compare complete behavioral history contrasts and stale residual patches at nearest depths .125/.875 × edited-span/final-preanswer, both queries, both edited variables and donor directions. No precision selected on effect strength. Qwen's ~16 GB FP16 weights plus runtime overhead generally exceed comfortable 16 GiB capacity; auto dispatch may offload, and actual placement is saved. Phi's ~7.6 GB FP16 weights are more practical. OOM/unsupported execution produces a technical stop, not evidence of quantization invariance. The sensitivity output is descriptive and does not create a replacement gate or an extra positive replication.

## Stages, restrictions, and provenance

0. Validate pinned tokenizer/config, exact chat/prefix, candidate sequences, disjoint surface events, all candidate substitutions, and paired input alignment. No weights/logits. Select/freeze the common vocabulary before competence. **No additional per-model redevelopment in v1.** The tokenizer-only refinement above is the sole preserved pre-freeze attempt.
1. One fixed 24-history development dataset; inspect only semantic competence, current/stale margins, ranks, surface likelihood/format diagnostics, and full-vocabulary diagnostics. Failure stops the model.
2. One fresh 24-history frozen gate, disjoint from development and available supplied legacy datasets. Fixed 64 cells and thresholds. Its seal binds datasets, scores, their provenance, development report, config, candidate map, code, renderer and immutable revisions. Confirmation recomputes the gate from its underlying files.
3. Produce a **preflight report before confirmatory generation**: model/revision, vocabulary/classes, development metrics, frozen criteria/results, expected cells, estimands, hook paths, and caveats. Confirmation requires its hash and a passing recomputed gate. Generate/score 96 fresh histories, identical across eligible models, and inspect causal results only now.
4. Core mechanistic battery, optional preregistered head battery, fixed quantization sensitivity.

`outputs/cross_model_v1/stage_claims/<model-and-revision-hash>/<stage>.json` permits one dataset per model per stage even across fresh output paths. Preserve claims and failed artifacts. There is no seed retry or threshold override. Scoring/patching alone support explicit `--resume`; resumes require exact fingerprints and saved metadata. Missing legacy Qwen datasets are a recorded audit limitation, not silently certified as disjoint. Shared confirmatory histories **across models** are intentional; within-model stage overlap is forbidden.

Exclude or stop for semantic incompetence, undefined token interventions, unsupported architecture, access/OOM, or failed implementation checks. Never exclude for weak causal effects, an inconvenient mechanism, or failure of the normalized metric's denominator guard. Each protocol/technical failure writes a fresh timestamped stopped artifact; semantic gate reports include an explicit stop reason.

Every artifact records git commit, code hash, immutable model/tokenizer IDs/revisions, full config/hash, renderer hash, candidate-map hash, exact dataset hash, seeds, package/Python versions, quantization, dtype, requested device map, prompt settings; validation maps also contain tokenizer JSON/template hashes and model config. Scoring/mechanism manifests additionally record actual device dispatch, edit audits, and activation dtypes where applicable. New files use exclusive creation; seals detect changes, not malicious adversaries. Raw score sets must match dataset IDs/metadata exactly. Confirmatory scoring requires the same package/Python runtime as gate scoring. Moving files or changing code/config after freezing requires preserving the failed/incompatible branch; paths and code hashes are intentionally strict.

## Statistics and claim levels

Primary unit is history. Preserve all paired edit/query members; average slot order and variable contrasts within history before 2,000 bootstrap draws (seed 73021). Report both estimands' 95% CIs, positive-primary history fraction, literal orientation, semantic-variable contrasts, and descriptive overlapping vocabulary-membership strata. Mechanistic windows and component/head contrasts also aggregate within history. Layer/prompt/edit/member counts are not independent sample sizes.

Do not pool prompt rows across families or fit a population hierarchical model with four purposively chosen families. Report each family separately, alongside descriptive signs, intervals, retention ratios where defined, and exclusions. Shared histories can reduce lexical variation in paired descriptive comparisons; they do not make the selected model families a random population sample.

- **A:** behavioral phenomenon in another family, with semantic competence and the new primary causal contrast.
- **B:** shared qualitative historical-token early / query-state late localization.
- **C:** similar component classes carry effects under comparable interventions.
- **D:** homologous specific circuits/heads — **not established by this battery**.

Different implementations are a scientifically useful result. Controlled English `nora_v1`, closed semantic classes, a small purposefully selected panel, quantization, and only 12 mechanistic histories constrain generalization.

## Exact commands

Run from repository root on the configured CUDA host. Choose each frozen panel slug in turn (`qwen3_8b`, `mistral7b`, `phi4_mini`, `llama31_8b`). All output paths must be new. No expensive commands below were run during implementation.

```bash
MODEL=mistral7b
CONFIG="configs/cross_model_v1/$MODEL.yaml"
RUN="outputs/cross_model_v1/$MODEL"

# Stage 0: no model weights or logits; add --local-files-only for cached tokenizers.
uv run python -m scripts.cross_model validate --config "$CONFIG" --output "$RUN/candidates.json"

# Cheap real loaded-model hook smoke; run once on the CUDA host before mechanisms.
uv run python -m scripts.cross_model smoke --config "$CONFIG" --candidates "$RUN/candidates.json" --output "$RUN/hook_smoke.json"

# Development: fixed 24 histories; scorer never computes E or R.
uv run python -m scripts.cross_model generate --stage development --config "$CONFIG" --candidates "$RUN/candidates.json" --output "$RUN/development.jsonl"
uv run python -m scripts.cross_model score --config "$CONFIG" --candidates "$RUN/candidates.json" --dataset "$RUN/development.jsonl" --output "$RUN/development_scores.jsonl"
uv run python -m scripts.cross_model analyze --stage development --config "$CONFIG" --candidates "$RUN/candidates.json" --dataset "$RUN/development.jsonl" --scores "$RUN/development_scores.jsonl" --output "$RUN/development_report.json"

# Fresh gate; generation stops if fixed development criteria fail.
uv run python -m scripts.cross_model generate --stage frozen_gate --config "$CONFIG" --candidates "$RUN/candidates.json" --development-report "$RUN/development_report.json" --output "$RUN/gate.jsonl"
uv run python -m scripts.cross_model score --config "$CONFIG" --candidates "$RUN/candidates.json" --dataset "$RUN/gate.jsonl" --output "$RUN/gate_scores.jsonl"
uv run python -m scripts.cross_model analyze --stage frozen_gate --config "$CONFIG" --candidates "$RUN/candidates.json" --dataset "$RUN/gate.jsonl" --scores "$RUN/gate_scores.jsonl" --development-report "$RUN/development_report.json" --output "$RUN/gate_report.json"

# Required report to review/share before any confirmatory causal run.
uv run python -m scripts.cross_model preflight --config "$CONFIG" --candidates "$RUN/candidates.json" --gate "$RUN/gate_report.json" --output "$RUN/preflight.json"

# Confirmation: fail-closed gate recomputation, frozen report binding, 96 fresh histories.
uv run python -m scripts.cross_model generate --stage confirmatory --config "$CONFIG" --candidates "$RUN/candidates.json" --gate "$RUN/gate_report.json" --preflight "$RUN/preflight.json" --output "$RUN/confirmatory.jsonl"
uv run python -m scripts.cross_model score --config "$CONFIG" --candidates "$RUN/candidates.json" --dataset "$RUN/confirmatory.jsonl" --output "$RUN/confirmatory_scores.jsonl"
uv run python -m scripts.cross_model analyze --stage confirmatory --config "$CONFIG" --candidates "$RUN/candidates.json" --dataset "$RUN/confirmatory.jsonl" --scores "$RUN/confirmatory_scores.jsonl" --output "$RUN/confirmatory_analysis.json"

# Minimal common M1–M4 battery; JSONL raw grid and .analysis.json history summaries.
uv run python -m scripts.cross_model mechanism --config "$CONFIG" --candidates "$RUN/candidates.json" --dataset "$RUN/confirmatory.jsonl" --scores "$RUN/confirmatory_scores.jsonl" --output "$RUN/mechanism.jsonl"

# Optional M5, with fixed six/six discovery/reserve split.
uv run python -m scripts.cross_model mechanism --heads --config "$CONFIG" --candidates "$RUN/candidates.json" --dataset "$RUN/confirmatory.jsonl" --scores "$RUN/confirmatory_scores.jsonl" --output "$RUN/heads.jsonl"

# Fixed Qwen8/Phi4-mini precision sensitivity only; set MODEL to the relevant slug first.
uv run python -m scripts.cross_model sensitivity --config "$CONFIG" --candidates "$RUN/candidates.json" --dataset "$RUN/confirmatory.jsonl" --scores "$RUN/confirmatory_scores.jsonl" --output "$RUN/precision_sensitivity.json"

# Cheap CPU unit/contract/native tiny-model checks.
uv run pytest tests/test_cross_model_v1.py -q
```

Use `--prior-dataset PATH` on generation to bind additional archived history datasets. Available old Mistral/Phi and Qwen natural-language paths are automatically included. If resuming an interrupted score or mechanism JSONL, rerun its exact command with `--resume`; never regenerate a claimed stage or overwrite its files. Stage 0 maps made on a different environment remain conditional on exact tokenizer/code/config hashes; repeat cheap validation on the execution host into a fresh path before its first stage if needed.
