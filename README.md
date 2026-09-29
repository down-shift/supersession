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
- Probe scores are linear decoding performance, not mutual information. Low decoding is not evidence of absence. Decoding is not evidence of causal use. Attention weights are not treated as causal evidence.
- Input counterfactual effects are controlled prompt interventions, not proof of an internal causal variable. Patch effects establish effects of the tested residual intervention; they do not identify a complete circuit. A small output effect does not show that obsolete information failed to propagate.
- The main design target is paired obsolete-query versus obsolete-distractor effects under fixed current state, reported both across all trials and alongside aggregate competence. Avoid interpreting a model that fails the configurable direct/overwrite competence gate as showing successful supersession.

## Outputs and provenance

Datasets and per-example behavioral / patch records are JSONL; activations are compressed NPZ chunks. JSON provenance snapshots include timestamp, commit if available, model/tokenizer IDs and resolved revisions, config, seed, data hash, package versions, dtype/device, and chat setting. Behavioral records include exact rendered prompt and query position; token validation output records accepted candidate token IDs and rejects. Do not discard raw records after aggregation.

`run_patching.py` performs both source→target directions for current-query, obsolete-query, and obsolete-distractor value pairs. It batches token positions by layer, saves raw per-layer/per-position logits, and produces the current-binding positive-control heatmap. The optional tokenizer-offset audit requires a fast tokenizer.
