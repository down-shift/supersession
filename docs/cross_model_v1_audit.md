# Cross-model audit before implementation

Audit base: git `4d1ffac` (clean working tree initially). Recent commits: `9ded9ac` introduced the second-family protocol; `ec6f452`, `ee47cf6` fixed Mistral; `80197f4` added Phi; `bef85f0`, `49821ff`, `34b5fa3`, `4d1ffac` addressed Phi loader compatibility. No established experiment, threshold, score file, or stopped branch was modified.

## Existing implementation

- `src/data/supersession_behavior.py`: explicit paired semantic records; `nora_v1`; current semantic answer derived independently of edits; x/z query contrasts; both irrelevant slot orders. The historical value occurs once in rendered text. All templates append the assistant prefix `Answer:` after the model-specific chat template, with `enable_thinking=False`.
- `src/data/token_validation.py`, `scripts/validate_natural_supersession_vocab.py`: encode **prompt + space + lowercase value** and demand prefix preservation, exactly one extra token, stable token ID, no candidate collisions. Paired input edits must change exactly one token at fixed length. A valid answer continuation does not imply that capitalized forms are one token. Candidate selection preferred the first Qwen map, otherwise the model-specific proposal pool, capped at 12.
- `src/experiments/behavior.py`, `src/analysis/metrics.py`: score final next-token logits, select among lowercase candidate IDs, compute full-vocabulary argmax separately. For a validated one-token candidate, its raw logit is a correct next-token score: subtracting two candidate log probabilities cancels the common log-normalizer. This does **not** score a split capitalized continuation or aggregate equivalent strings.
- `src/analysis/natural_competence.py`: `natural_competence_v2`; unique prompts within each represented cell; 64 required focal cells; exact-token accuracy >=.99, candidate accuracy >=.99, mean candidate rank <=1.01, positive mean current-minus-stale margin. Ordinary irrelevant is diagnostic. Missing IDs/cells fail.
- `scripts/analyze_natural_competence.py`, `scripts/generate_supersession_experiments.py`: score/dataset/config/token-map/renderer hashes, seal, recomputation from underlying scores, full concrete-history signatures, prior-stage overlap checks. The old failed gates remain failed. Some old saved artifacts predate the newer diagnostic fields; their stored cells still expose the failed criteria.
- `src/analysis/supersession_behavior.py`: fixed replacement-minus-source identity transfer, symmetric query relevance, slot-counterbalanced irrelevant, paired history contrasts, 2,000 history-bootstrap draws; all trials retained. No normalization by model scale.
- `src/models/hooks.py`, `src/experiments/patching.py`: block-output residual capture/patch, donor casting to recipient activation dtype; aligned single-position edits; raw patch deltas and guarded recovery ratios.
- `src/experiments/component_patching.py`: deliberately Qwen3-only; attention **module output after o_proj and before residual addition**; MLP before residual addition; attention and MLP interventions are in serial causal order.
- `src/experiments/attention_head_patching.py`: Qwen3-only validation; query-head results at **o_proj input**, not slices of projected hidden dimensions; GQA dimension checks. Negative donor-oriented effects do not identify suppression.
- `src/models/loader.py`: native AutoModel loader for Qwen/Mistral/Llama; Phi pinned remote implementation with typing/tied-weight/LongRoPE buffer shims. Existing Qwen config is float16 + bitsandbytes int8; old Mistral likewise; old Phi was unquantized BF16.

## Failed gates: problem → cause → fix → verification

Problem: exact-token full-vocabulary accuracy diverges from candidate selection.

Cause: the validated lowercase token universe omits capitalized realizations and their subword fragments. Closed-set ranking is often correct even when the unconstrained argmax is outside that universe. Mistral additionally has genuine candidate-ranking failures. A one-token fragment is **not proof of a semantically correct completed answer**.

| Existing run | Unique prompts pooled across conditions | Full-vocabulary exact token | Lowercase candidate accuracy | Full-token misses | Case-only misses after diagnostic whitespace trimming |
|---|---:|---:|---:|---:|---:|
| Mistral development | 554 | 52.53% | 99.28% | 263 | 90 |
| Mistral fresh gate | 568 | 51.06% | 98.59% | 278 | 69 |
| Phi development / saved ropefix rescore | 574 | 85.37% | 100% | 84 | 82 |
| Phi fresh gate | 576 | 78.30% | 100% | 125 | 123 |

These pooled unique-prompt percentages differ from row-weighted and per-condition reports because prompts recur across edit members and conditions. Per-condition Phi figures reproduce the supplied results: development live .6338, superseded .9028, irrelevant/CB .9375; gate .5556/.8125/.8681/.8819. Mistral gate per-condition candidate accuracy is 1.0000/.9931/.9722/.9757 (live/superseded/irrelevant/CB). Thus a semantic-only reanalysis would not automatically rescue the old Mistral gate.

Mistral failed exact-token accuracy in 59/64 stored gate cells and candidate accuracy in 9/64. Phi failed exact-token accuracy in 47/64 and candidate accuracy in 0/64. Both stored `pass` fields are false. The new protocol requires fresh scores and histories; none of these files qualifies a new gate.

Mistral development misses include `Silver` (51), `R` (47), `Ol` (42), `Navy` (39), `Bron` (33), `L` (32), `Whe` (15), and a small number of other candidates or unrelated fragments. Phi development's two non-case-only misses are ` G`; its gate also has two ` G` misses. Stored scores contain only the first greedy token, so longer decoded completions cannot be reconstructed. Capitalization/sentence-initial realization is a plausible explanation, not a validated complete-answer accuracy result.

Exact observed suffixes:

```text
Mistral: ...with no explanation.[/INST]Answer:
Phi:     ...with no explanation.<|end|><|assistant|>Answer:
```

Whitespace is tokenizer-specific (`Navy` versus ` Rust` decoded first tokens). The old validator uses a leading ASCII space even if decoding hides it. No separate dominant punctuation-error cluster appears among these saved misses. Chat-template suffixes and `Answer:` establish different model-specific distributions. The audit does not establish their causal contribution independently. The prefix is deliberately retained in v1 to preserve semantic/prompt continuity.

Fix: a **new** staged, surface-class, complete-sequence protocol with exact-token diagnostics and fresh competence gates. No output correction or old threshold change.

Verification: tokenizer-only prefix preservation, event collisions/prefix-freeness, complete sequence likelihood tests, exact edit audits, per-cell semantic gates, and raw-score-bound seals. New model competence remains unmeasured.

## Existing Qwen natural-language confirmation

`outputs/supersession/nl_confirmatory_analysis/summary.json` and `history_relevance.csv` are present. They record 96 histories, 1,920 matched edit pairs, 3,840 scored members, `R_live=23.0702`, `R_superseded=4.5907`, `R_irrelevant_counterbalanced=.07095`; the paired primary contrast is about 4.52 logits with the reported [4.12,4.94] history-bootstrap CI and 96/96 positive histories. This is the original canonical one-token estimand, not the new surface-mass estimand.

The Qwen natural-language raw development/gate/confirmatory JSONL files named in the protocol are **absent in this checkout**. The saved analysis can be inspected, but the entire original seal/score/dataset lineage and overlap against those missing histories cannot be independently recomputed here. Available historical Mistral/Phi datasets are checked for overlap by the new generator; missing legacy files are not invented.

No Mistral/Phi `R` values or causal analyses were inspected. The stopped version-chain and focal validity/status branches were not reopened.

## Tokenizer-only vocabulary construction before new logits

First exploratory shared proposal: `amber, coral, jade, navy, pearl, rust, olive, silver`. It passed continuation-event checks after the predefined prefix-preservation rule, but mixed one/two-token historical spans in Mistral, making the full-span battery incompatible. This attempt is preserved here; no model forward or causal effect informed the change.

The existing 36-word proposal pool was grouped by historical token-span lengths under cached pinned Qwen/Mistral/Phi tokenizers. Groups included:

- (1,1,1): navy, rust, wheat, bronze, lemon, olive, silver — only seven values.
- (1,2,1): amber, coral, denim, elm, frost, grape, jade, maple, pearl, quartz, slate, teal, violet, zinc, cyan, ivory, plum, walnut — largest group (18).
- (2,2,2): hazel, indigo, ochre, acorn, opal — five values.
- Other groups were smaller or less lexically attractive.

Within the largest group, fixed **ordinary color names** were chosen: `amber, coral, jade, pearl, slate, teal, violet, ivory`. This gives eight shared meanings and equal span length **within each model**, while allowing different lengths across models. Selection used only tokenizer/format feasibility and lexical familiarity. It was finalized before any new competence logits. Llama remains access-dependent and must pass the same validation without per-model vocabulary substitution.

Canonical next-token IDs: Qwen 67605/52003/85015/69623/50580/94607/79736/69816; Phi 111978/79292/87863/100121/86663/177759/117320/131584, in the above order. Mistral's canonical spaced continuations are two tokens; their exact sequences and casing variants are recorded in the new validation maps. Original exact-token accuracy and raw R are undefined for that complete Mistral universe; first-fragment correctness is explicitly diagnostic.

Some unspaced strings retokenize the encoded `Answer:` prefix. The fixed rule excludes those joint-encoding forms using tokenizer evidence, requires spaced lowercase and title-case forms, and deduplicates identical token events. These classes are deliberately bounded; they do not cover every possible semantically equivalent token sequence or formatting choice.

## Validation completed during implementation

- Final cheap repository unit suite after the review corrections: 190 passed, including synthetic score tests, temporary artifact tests and tiny native architecture tests. No pretrained weights or confirmatory effects were used.
- The earlier Stage 0 maps checked 96 unique prefixes, 80 actual edit pairs, and 896 exhaustive candidate/slot substitutions for Qwen, Mistral and Phi. Those maps bind the superseded contract and code hash. They remain preserved but are not valid candidates for this revision.
- Fresh Stage 0 maps for `review_1_before_logits` are written separately under `outputs/cross_model_v1/tokenizer_validation_review1_final_20261001/`. They are no-logit tokenizer/format audits for revision 1; they do not validate revision 2.
- Mistral spaced canonical sequences: amber [1605,1305], coral [2043,1050], jade [1229,1538], pearl [27615,29482], slate [1903,1148], teal [1479,1050], violet [1131,21114], ivory [20668,1463]. All four surface proposals and their distinct events are recorded in its map.
- CLI help and whitespace checks passed. Llama tokenization/access, real pinned remote Phi hooks, GPU/int8 hooks, memory footprints, new semantic development, and new frozen competence gates remain unmeasured.
- Revision 2 adds INFO logs and progress bars and changes the protocol revision/seeds; its candidate maps and gates must be regenerated at fresh paths. Shell syntax, Python compilation, CLI help, and all four config/contract matches passed after the instrumentation edit. The 190-test result above predates this instrumentation revision.

## `cross_model_v1` execution outcomes recorded 2026-10-02

These outcomes postdate the implementation audit above; they do not alter the historical exact-token failures or any stopped branch.

- **Qwen3-8B:** the 96-history confirmatory run had `R_superseded - R_irrelevant_counterbalanced = 3.50` nats (95% history-bootstrap CI `[3.20, 3.80]`), positive in 94/96 histories. This bounded surface-class continuation-mass estimand is distinct from the earlier 4.52-logit result.
- **Phi-4-mini:** passed the frozen semantic competence gate with rank-one accuracy 1.00 in all 64 required cells. No confirmatory causal or mechanistic result is available; the gate pass is not a causal replication.
- **Mistral-7B-Instruct-v0.3:** failed the frozen gate: counterbalanced-irrelevant accuracy was 281/288 (97.57%), below the fixed 99% criterion. Live and superseded accuracy were 100%. Its seven counterbalanced errors were six rank-2 and one rank-3 prompts. The positive mean superseded current-over-stale semantic-mass margin (16.50) is a competence diagnostic, not causal `R`. No confirmatory or mechanistic run was generated. Multi-token candidate continuations make canonical raw-logit `R` undefined for this candidate map.
- **Gemma 3 4B:** gate scoring was still in progress in the last recorded status; no gate decision is recorded here.

The Mistral development and gate reports recompute from their score rows with valid score hashes. Four live diagnostic cells have 11 unique prompts rather than 12 because two histories rendered the same prompt; all 64 required cells are present. The gate failure is the aggregate counterbalanced-irrelevant criterion, not the duplicate prompt. The earlier `FileExistsError` was an output-path collision and is not the gate result. See `outputs/mistral7b_review2/gate_report.json` and its score provenance sidecar.

No cross-model causal generalization is established by the Phi gate or the Mistral failure. Do not inspect or infer unrun causal effects. The intended claim hierarchy and evidence limits are recorded in [paper_claims.md](paper_claims.md).
