## 4. Experimental protocol

### 4.1 Models and eligibility

V1 evaluated Qwen3-8B, Gemma 3 4B, Phi-4-mini, and Mistral-7B-Instruct-v0.3. Qwen, Gemma, and Phi passed the competence gate and completed confirmation. Mistral scored 281/288 on its unassigned-control gate and did not proceed to confirmation. V2 evaluated Qwen, Gemma, and Phi. Qwen and Gemma passed all six condition-level competence gates and completed confirmation; Phi scored 561/576 (97.40%) in the entity-mention condition and did not proceed. Phi's completed v1 estimate is reported separately.

Each gate condition must have at least 99% strict rank-one accuracy, with ties counted as incorrect, and all required diagnostic cells must be present. Gate accuracy counts each distinct rendered prompt once within its condition. The superseded condition must also have a positive mean current-minus-stale score margin. The lowest v2 condition accuracy was 287/288 (99.65%) for Qwen in live and 575/576 (99.83%) for Gemma in entity mention. Both models scored 576/576 in superseded, with current-minus-stale margins of 21.571 and 23.910 nats, respectively. These gates select configurations for confirmation; they do not test the three-control prediction.

**Table 1. Model coverage by protocol.** Gate eligibility is distinct from confirmation performance. A dash denotes no confirmatory estimate.

| Protocol | Model | Gate | Confirmation histories |
|---|---|---|---:|
| v1 | Qwen3-8B | Pass | 96 |
| v1 | Gemma 3 4B | Pass | 96 |
| v1 | Phi-4-mini | Pass | 96 |
| v1 | Mistral-7B-Instruct-v0.3 | Fail: 281/288 in unassigned control | — |
| v2 | Qwen3-8B | Pass | 96 |
| v2 | Gemma 3 4B | Pass | 96 |
| v2 | Phi-4-mini | Fail: 561/576 in entity mention | — |

### 4.2 Histories, inference, and uncertainty

V1 used 24 development histories, 24 gate histories, and 96 confirmation histories per eligible model. Confirmation repeated 12 scheduled combinations of entity pair, attribute, and semantic orientation eight times with different value assignments. V2 used the same stage sizes but crossed six entity pairs, four target attributes (`badge`, `color`, `code`, `label`), and two semantic orientations. Each of the resulting 48 confirmation cells had two histories, with the alternate `team` and `project` relations balanced across the design. Development and gate used fixed 24-history subsets of this allocation. V2 used one set of sentence templates throughout.

Each v2 history contributes $6\times2\times2\times2\times2\times2=192$ scored rows: six conditions, two queries, two edited values, two historical orders, two current orders, and baseline versus edit. This gives 18,432 rows and 9,216 matched edit pairs per model. Each condition contributes 3,072 rows, half baseline and half edited. Some rows share the same rendered prompt, including live's duplicated current-order variants. Confirmation accuracy uses all rows, whereas gate accuracy deduplicates prompts. Estimates and uncertainty are computed over the 96 histories, and the primary analysis retains histories containing task errors.

We aggregate edit and query contrasts within each history before averaging across histories. The reported primary percentile 95% intervals use 2,000 unrestricted history-bootstrap samples with seed 73021: each draw samples 96 histories with replacement from the full confirmation set, without resampling within the 48 entity-pair-by-attribute-by-orientation cells. This follows the dated analysis amendment's history-level bootstrap and targets variation across histories in the balanced confirmation mixture. Appendix B reports a design-preserving sensitivity analysis that resamples two histories within each of the 48 fixed cells. Qwen and Gemma share the same confirmation histories, enabling comparisons under the same assignments and donors. These intervals describe variation across histories under the fixed task design; they do not cover changes in names, vocabulary, sentence templates, numerical precision, or model selection. The exclusion audit and implementation chronology are reported in Appendix C.

Tokenizer audits checked that each baseline/edit pair differed only at the selected value span, apart from the consequences of changing its token length. The reported model configurations used int8 weights; Qwen used float16 computation and Gemma bfloat16. Model checkpoint revisions, inference settings, and the implementation and dataset hashes are reported in Appendix C. Protocol decisions were internally logged rather than externally preregistered.
