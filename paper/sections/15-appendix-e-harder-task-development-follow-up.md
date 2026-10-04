## Appendix E. Harder-task development follow-up

The follow-up protocol and stopping amendment were recorded before the development outcomes were available. Each development level used 12 histories and 1,536 scored members (1,152 unique prompts) per model, with matched histories across levels and models. Qwen's complete-answer accuracy exceeded the prespecified 90% threshold at n=2, n=4, and n=6. The protocol therefore stopped Experiment 2: the tested distractor manipulation did not produce the intended difficulty. No level was selected, no harder-task test was generated, and the nearest-level fallback was not used. Gemma's near-ceiling accuracy at all three levels is not an informative behavioral replication.

| Model | n=2 | n=4 | n=6 |
|---|---:|---:|---:|
| Qwen3-8B | 1,531/1,536 (99.67%) | 1,532/1,536 (99.74%) | 1,533/1,536 (99.80%) |
| Gemma 3 4B | 1,534/1,536 (99.87%) | 1,532/1,536 (99.74%) | 1,532/1,536 (99.74%) |

The development runs used the pinned model revisions and matching per-model runtime fingerprints across levels. The full record includes validation, error distributions, score-based secondary analyses, provenance hashes, and runtimes; see [followup_experiments.md](../docs/followup_experiments.md). Because the stopping rule prevented confirmatory test generation, these results support only the operational conclusion that this distractor manipulation failed to lower Qwen's accuracy as intended.
