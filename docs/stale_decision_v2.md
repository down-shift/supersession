# Manuscript guide

The current manuscript is [When Old Values and Incidental Mentions Change LLM Routing Decisions](../paper_dai/main.tex). Its tables, worked prompt, appendix, and build instructions are in [paper_dai](../paper_dai/README.md). The earlier ACL manuscript and protocol history remain in `paper_v2`.

## Design

Synthetic histories contain two units, updates, and a policy. Routing maps the queried unit's current state to EAST or WEST. Access uses current clearance to choose APPROVE or DENY. Within a pair, one earlier word changes between a word implying the correct action and one implying the wrong action. The current state, policy, question, and correct action remain fixed.

The edited word appears in a former routing state (S), an updated archive tag (O), a mention by the unit (M), or a mention without a named unit (U). Each task uses sequential, interleaved, and competing layouts. The wrong-route word also names the other unit's current state, so edited-word sensitivity and unit confusion are possible explanations.

The score effect `D_c` is the mean paired change in wrong-minus-correct action log probability, in nats. The net change in wrong answers is the wrong-answer fraction with the wrong-route word minus that with the correct-route word. Invalid answers count as accuracy failures and are outside the wrong-answer category in this calculation. Generation is greedy with thinking disabled and a 16-token budget in the main runs. A valid answer must terminate with the end-of-sequence token and match an allowed label after trimming whitespace.

## Results and populations

The [main routing table](../paper_dai/routing.tex) uses confirmation histories from layouts that passed screening for each model. Replacing the correct-route word with the wrong-route word increased wrong routes in S and M for all five models. The paired increase was larger for M in Qwen3-8B and Granite 4.1 8B. The Qwen3-14B and Qwen3.5 9B intervals include zero. Gemma 4 12B is inconclusive because two saved intervals on the same 384 histories disagree on whether zero is included.

The post hoc retrieval analysis retains histories where the model reports the queried current state correctly across all eight prompts, reports the former state correctly in both S prompts, and chooses correctly without earlier lines. Each model uses the same retained histories across conditions. Qwen3-8B retains 335 of 384 histories, with net increases of 0.161 for S and 0.266 for M. Separate state reports do not establish which state the model used when routing.

The original primary score outcomes use 97.5% history-bootstrap intervals adjusted for the two tasks. Generated decisions and mention comparisons are secondary and use 95% intervals without adjustment across models. The protocol was frozen internally before confirmation.

## Limits and follow-ups

A neutral-word baseline is needed to distinguish harm from the wrong-route word from help from the correct-route word. Matched paraphrases would measure switching under equivalent wording changes. Conditions use different sentence templates, so the S–M comparison does not isolate the role of a former assignment. The records are short and synthetic, and effects in deployed systems remain unknown.

The archive-tag follow-up replaces the later tag update with a neutral line of equal word count. Its score effect is larger than O for both Qwen3 models. Thinking-enabled runs use a larger generation budget, so their comparison with the main runs cannot isolate thinking. The identical-current-state follow-up uses new histories and is descriptive.

Saved aggregates were audited, but original raw model outputs and scoring records were unavailable for independent reproduction. The [author evidence audit](../paper_dai/README.md) maps results to the committed data and documents the missing inputs.
