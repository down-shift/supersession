# Supersession Without Erasure

Infrastructure for testing how causal decoder-only Transformers make overwritten bindings less accessible or less influential. The old assignment token cannot receive information from later overwrite tokens under a causal mask. This pipeline measures query-position accessibility and downstream effects; it does not assume deletion.

## Environment

The repository uses uv for interpreter selection, dependency resolution, locking, and command execution. `.python-version` selects Python 3.13. Install the CPU/data/test environment with:

```bash
uv sync --extra dev
```

On the experiment machine, install model support as well:

```bash
uv sync --extra model --extra dev
```

Configure model ID, exact optional revision, tokenizer, device map, quantization, dtype, chat setting, candidate vocabulary, and seeds in `configs/pilot.yaml` or `configs/primary.yaml`. No model download is initiated by tests or dataset generation. `src.models.loader` supports standard Hugging Face decoder-only models exposing `layers` and reports unsupported architectures clearly. The configured Qwen3-8B run uses bitsandbytes LLM.int8 with FP16 non-quantized layers and `device_map: auto`. Attention uses PyTorch SDPA (`attn_implementation: sdpa`), which selects an available PyTorch attention kernel; it complements weight quantization rather than replacing it ([Transformers attention backends](https://huggingface.co/docs/transformers/main/attention_interface)). Matching FP16 avoids the common BF16-to-FP16 cast path in bitsandbytes; warnings are not globally suppressed. For RTX 5080 (SM120), bitsandbytes lists CUDA 12.8+ builds for Linux and Windows; install a compatible NVIDIA driver/PyTorch CUDA runtime on the experiment PC ([bitsandbytes install matrix](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/docs/source/installation.mdx)).

## Commands

### Four-query matched pilot (new primary design)

This design uses one fixed history to produce current-x, initial-x, current-z,
and initial-z queries. The pilot config gives 144 histories from the full
`6 orders × 4 variable-name pairs × 2 variable orientations × 3 replicates`
design. It uses exactly 12 tokenizer-validated values. Calibration and
confirmatory histories use disjoint value-pattern namespaces, and neither set
contains duplicate histories. Calibration data must not enter confirmatory
analysis.

```bash
uv run python scripts/validate_tokens.py --config configs/four_query_pilot.yaml --design four-query --output outputs/four_query/token_ids.json
uv run python scripts/generate_four_query.py --config configs/four_query_pilot.yaml --token-ids outputs/four_query/token_ids.json --calibration --kind queries --output outputs/four_query/calibration.jsonl
uv run python scripts/audit_four_query.py outputs/four_query/calibration.jsonl
uv run python scripts/run_four_query_competence.py --config configs/four_query_pilot.yaml --dataset outputs/four_query/calibration.jsonl --token-ids outputs/four_query/token_ids.json --output outputs/four_query/calibration.json
uv run python scripts/generate_four_query.py --config configs/four_query_pilot.yaml --token-ids outputs/four_query/token_ids.json --kind queries --output outputs/four_query/behavior_inputs.jsonl
uv run python scripts/generate_four_query.py --config configs/four_query_pilot.yaml --kind pairs --token-ids outputs/four_query/token_ids.json --output outputs/four_query/pairs.jsonl
uv run python scripts/audit_four_query.py outputs/four_query/behavior_inputs.jsonl --exclude-dataset outputs/four_query/calibration.jsonl
uv run python scripts/audit_four_query.py outputs/four_query/pairs.jsonl --exclude-dataset outputs/four_query/calibration.jsonl
uv run python scripts/run_behavior.py --config configs/four_query_pilot.yaml --dataset outputs/four_query/behavior_inputs.jsonl --token-ids outputs/four_query/token_ids.json --output outputs/four_query/behavior.jsonl
uv run python scripts/run_behavior.py --config configs/four_query_pilot.yaml --dataset outputs/four_query/pairs.jsonl --token-ids outputs/four_query/token_ids.json --output outputs/four_query/pair_behavior.jsonl
uv run python scripts/analyze_four_query.py --behavior outputs/four_query/behavior.jsonl --pairs outputs/four_query/pair_behavior.jsonl --output-dir outputs/four_query/analysis
```

Both scoring commands checkpoint each completed example to JSONL. If interrupted,
rerun the same command with `--resume`; it verifies the dataset, token IDs, and
config against a sidecar manifest and skips completed examples. Calibration
records default to `OUTPUT.records.jsonl`; behavior records use `--output`
directly. Keep the same paths and config when resuming.

`generate_four_query.py --kind histories` writes one record per history;
`queries` expands each into the four fixed query conditions; `pairs` creates
baseline/edit members for each of the four bindings under every query. Run
calibration first and stop if full-vocabulary next-token accuracy misses the
configured threshold. For discovery patching after the behavior result:

On Ubuntu with an NVIDIA GPU, `bash scripts/run_four_query_288_ubuntu.sh`
runs the 288-history behavior sequence end to end. It checks CUDA access,
uses the locked dependencies, stops at a failed calibration gate, and resumes
scoring from matching checkpoints. Set `REVALIDATE_TOKENS=1` to regenerate
the token file.

```bash
uv run python scripts/run_four_query_patching.py --config configs/four_query_pilot.yaml --pairs outputs/four_query/pairs.jsonl --token-ids outputs/four_query/token_ids.json --output outputs/four_query/patching.jsonl --stage discovery
uv run python scripts/analyze_four_query_patching.py --patches outputs/four_query/patching.jsonl --output-dir outputs/four_query/patch_analysis_discovery --stage discovery
```

The default discovery stage uses 24 histories. Its analyzer prespecifies the
selection rule: at `final_preanswer`, select the contiguous three-layer window
with the largest discovery `S_x` (lowest start layer breaks ties). Freeze those
indices, then validate only that region on disjoint histories (up to 96):

```bash
uv run python scripts/run_four_query_patching.py --config configs/four_query_pilot.yaml --pairs outputs/four_query/pairs.jsonl --token-ids outputs/four_query/token_ids.json --output outputs/four_query/patching_heldout.jsonl --stage heldout --layers 12,13,14
uv run python scripts/analyze_four_query_patching.py --patches outputs/four_query/patching_heldout.jsonl --output-dir outputs/four_query/patch_analysis --stage heldout
```

An all-position discovery sweep is optional and limited to 24 histories with
`--all-positions`; heldout patching always requires frozen layer indices.

`configs/four_query_pilot.yaml` uses int8 weights for the inexpensive behavior
pilot. Repeat focal behavioral and patching results with
`configs/four_query_fp16.yaml` before making mechanistic claims.

Only after the paired behavioral effect is established, extract query-state
activations and run history-grouped probes. This evaluates the same four slots
under all four queries; decoding remains descriptive.

```bash
uv run python scripts/run_extraction.py --config configs/four_query_pilot.yaml --dataset outputs/four_query/behavior_inputs.jsonl --token-ids outputs/four_query/token_ids.json --output outputs/four_query/activations
uv run python scripts/run_four_query_probes.py --dataset outputs/four_query/behavior_inputs.jsonl --activations outputs/four_query/activations --token-ids outputs/four_query/token_ids.json --output outputs/four_query/probes.json
```

Run checks:

```bash
uv run pytest -q
```

First validate candidates on the work PC so generated answer values use the actual model tokenizer. The command renders complete prompts; it does not execute the model:

```bash
uv run python scripts/validate_tokens.py --config configs/pilot.yaml --output outputs/pilot/token_ids.json
```

Generate a small pilot dataset, direct-binding control, and explicit counterfactual pairs using only those accepted token values:

```bash
uv run python scripts/generate_dataset.py --config configs/pilot.yaml --token-ids outputs/pilot/token_ids.json --output outputs/pilot/dataset.jsonl
uv run python scripts/generate_direct.py --config configs/pilot.yaml --token-ids outputs/pilot/token_ids.json --output outputs/pilot/direct.jsonl
uv run python scripts/generate_pairs.py --config configs/pilot.yaml --token-ids outputs/pilot/token_ids.json --output outputs/pilot/pairs.jsonl
uv run python scripts/audit_dataset.py outputs/pilot/dataset.jsonl
```

Score direct and overwrite competence:

```bash
uv run python scripts/run_competence.py --config configs/pilot.yaml --direct outputs/pilot/direct.jsonl --overwrite outputs/pilot/dataset.jsonl --token-ids outputs/pilot/token_ids.json
```

Behavioral pilot and input-counterfactual analysis:

```bash
uv run python scripts/run_behavior.py --config configs/pilot.yaml --dataset outputs/pilot/dataset.jsonl --token-ids outputs/pilot/token_ids.json --output outputs/pilot/behavior.jsonl
uv run python scripts/run_behavior.py --config configs/pilot.yaml --dataset outputs/pilot/pairs.jsonl --token-ids outputs/pilot/token_ids.json --output outputs/pilot/pair_behavior.jsonl
uv run python scripts/analyze.py --behavior outputs/pilot/behavior.jsonl --pairs outputs/pilot/pair_behavior.jsonl --output-dir outputs/pilot/analysis
```

Activation extraction, matched linear probes, and residual patching positive control:

```bash
uv run python scripts/run_extraction.py --config configs/pilot.yaml --dataset outputs/pilot/dataset.jsonl --token-ids outputs/pilot/token_ids.json --output outputs/pilot/activations
uv run python scripts/run_probes.py --dataset outputs/pilot/dataset.jsonl --activations outputs/pilot/activations --token-ids outputs/pilot/token_ids.json --output outputs/pilot/probes.json
uv run python scripts/plot_probes.py outputs/pilot/probes.json --output-dir outputs/pilot/analysis
uv run python scripts/run_patching.py --config configs/pilot.yaml --dataset outputs/pilot/dataset.jsonl --token-ids outputs/pilot/token_ids.json --output outputs/pilot/patching.jsonl --n-pairs 2
```

Full primary-model run (includes the preceding stages):

```bash
bash scripts/run_pipeline.sh configs/primary.yaml outputs/primary
```

For a behavioral replication that fully crosses order, query, syntax, and whitespace five times, and does not spend GPU time on probes or patching:

```bash
bash scripts/run_behavioral_replication.sh configs/behavioral_replication.yaml outputs/behavioral_replication
```

Natural-language replication uses the same paired abstract task and tokenizer checks:

```bash
uv run python scripts/validate_tokens.py --config configs/primary.yaml --family natural --output outputs/natural/token_ids.json
uv run python scripts/generate_dataset.py --config configs/primary.yaml --family natural --token-ids outputs/natural/token_ids.json --output outputs/natural/dataset.jsonl
uv run python scripts/generate_direct.py --config configs/primary.yaml --family natural --token-ids outputs/natural/token_ids.json --output outputs/natural/direct.jsonl
uv run python scripts/generate_pairs.py --config configs/primary.yaml --family natural --token-ids outputs/natural/token_ids.json --output outputs/natural/pairs.jsonl
uv run python scripts/run_competence.py --config configs/primary.yaml --direct outputs/natural/direct.jsonl --overwrite outputs/natural/dataset.jsonl --token-ids outputs/natural/token_ids.json --output outputs/natural/competence.json
uv run python scripts/run_behavior.py --config configs/primary.yaml --dataset outputs/natural/dataset.jsonl --token-ids outputs/natural/token_ids.json --output outputs/natural/behavior.jsonl
uv run python scripts/run_behavior.py --config configs/primary.yaml --dataset outputs/natural/pairs.jsonl --token-ids outputs/natural/token_ids.json --output outputs/natural/pair_behavior.jsonl
uv run python scripts/analyze.py --behavior outputs/natural/behavior.jsonl --pairs outputs/natural/pair_behavior.jsonl --output-dir outputs/natural/analysis
```

Then pass the natural dataset and token IDs to the same behavioral, extraction, probe, and patching commands. Run a second model family by creating another YAML config; models are not swept automatically.

## Definitions and interpretation

- Query position is the last model-input token before answer continuation. Activation storage includes the input embedding at that position and each decoder block output at that position. Patching convention is residual stream **leaving** a selected block, at a selected input position.
- `B = logit(O_q) - logit(O_d)` is binding-specific obsolete residue; `R = logit(C_q) - logit(C_d)` is the current-binding control; `M` compares the correct value to all three distractor candidates. Accuracy is argmax over the configured candidate vocabulary. Full candidate probabilities and per-role logits are stored.
- The four-query design fixes history and wording across query variants. Its primary edit score is `[logit(replacement)-logit(source)]_edited - [logit(replacement)-logit(source)]_baseline`; its primary contrast compares that score for current-x versus current-z while holding source, replacement, position, and history fixed. It also records full-vocabulary next-token accuracy, full-vocabulary target rank, candidate rank, and candidate probability.
- Four-query summaries bootstrap and sign-flip at the `history_id` level. Prompt-development and held-out gate histories are generated independently and excluded from confirmatory outputs. `audit_four_query.py` fails on duplicate or overlapping concrete histories, incomplete query sets, or role/edit imbalance.
- Probe scores are linear decoding performance, not mutual information. Low decoding is not evidence of absence. Decoding is not evidence of causal use. Attention weights are not treated as causal evidence.
- Input counterfactual effects are controlled prompt interventions, not proof of an internal causal variable. Patch effects establish effects of the tested residual intervention; they do not identify a complete circuit. A small output effect does not show that obsolete information failed to propagate.
- The main design target is paired obsolete-query versus obsolete-distractor effects under fixed current state, reported both across all trials and alongside aggregate competence. Avoid interpreting a model that fails the configurable direct/overwrite competence gate as showing successful supersession.

## Outputs and provenance

Datasets and per-example behavioral / patch records are JSONL; activations are compressed NPZ chunks. JSON provenance snapshots include timestamp, commit if available, model/tokenizer IDs and resolved revisions, config, seed, data hash, package versions, dtype/device, and chat setting. Behavioral records include exact rendered prompt and query position; token validation output records accepted candidate token IDs and rejects. Do not discard raw records after aggregation.

`run_patching.py` performs both source→target directions for current-query, obsolete-query, and obsolete-distractor value pairs. It batches token positions by layer, saves raw per-layer/per-position logits, and produces the current-binding positive-control heatmap. The optional tokenizer-offset audit requires a fast tokenizer.

## Four-query prompt development, gate, and confirmation

The four-query experiment now separates **prompt development ≠ held-out competence gate ≠ confirmatory experiment**. The previous Qwen3-8B INT8 run failed the pre-set 99% gate (current-x 93.2%, current-z 92.2%, initial-x 58.3%, initial-z 59.4%). Historical-query performance fell as old/current assignments became more separated. No confirmatory result was obtained; this failure is a task-clarity problem, not evidence about mechanistic supersession.

The runner first validates one shared 12-value vocabulary under all three deterministic prompt variants, then scores the same 96 prompt-development histories under each:

- `first_latest` (preferred): plain `x = value` lines; asks which value was assigned first or most recently.
- `initial_update` (fallback): prefixes old lines with `Initial assignment:` and current lines with `Update:`; asks before any updates or after all updates.
- `timestamped` (diagnostic only): prefixes lines with `t0:` / `t1:` and asks at the corresponding time. It is never automatically selected for confirmation.

Selection is fixed in advance: choose `first_latest` if every query has at least 98% development accuracy; otherwise choose `initial_update` if every query meets 98%; otherwise stop. Selection is global across x/z and initial/current. The artifact records query accuracies, dataset/config hashes, revisions, frozen token IDs, and whether confirmatory use is allowed.

A fresh 192-history gate is generated with only that frozen variant. Every query (`current_x`, `initial_x`, `current_z`, `initial_z`) must achieve at least 99% **full-vocabulary next-token accuracy**. Gate failures are saved and halt the runner before confirmatory data is generated. Gate examples cannot be used to retune the prompt; any later prompt revision requires a fresh gate partition/seed. Only a passing gate permits the fresh 288-history confirmatory experiment. Prompt-development and gate data are audited against confirmatory histories and excluded from confirmatory analysis. Initial-value queries serve as retention/task controls; current-value queries must also be nearly perfectly solved before mechanistic work. Timestamped lookup is a more explicit diagnostic task and is not equivalent evidence.

Run the full staged Ubuntu workflow (model inference occurs on that host):

```bash
bash scripts/run_four_query_288_ubuntu.sh
```

The runner does behavior-only scoring and analysis. It does not run probes or patching. The four-query generation path always uses centralized canonical rendering; legacy syntax/template factors remain limited to older experiment paths.
