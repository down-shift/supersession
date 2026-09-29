"""Validate answers as single continuation tokens in the exact rendered prompt."""
from tqdm.auto import tqdm

def continuation_token_id(tokenizer, prompt, answer):
    prefix=tokenizer(prompt,add_special_tokens=False)["input_ids"]
    full=tokenizer(prompt+answer,add_special_tokens=False)["input_ids"]
    if full[:len(prefix)] != prefix or len(full)!=len(prefix)+1:
        raise ValueError(f"{answer!r} is not exactly one continuation token for this prompt")
    return int(full[-1])

def validate_candidate_vocabulary(tokenizer, contexts, candidates, chat=True):
    valid={}; rejected={}
    for value in tqdm(candidates,desc="Validating candidate tokens"):
        ids=set(); ok=True; failure=None
        for ex in contexts:
            prompt=__import__("src.data.generate",fromlist=["render_example"]).render_example(ex,tokenizer,chat=chat)
            try: ids.add(continuation_token_id(tokenizer,prompt," "+value))
            except ValueError as exc: ok=False; failure=str(exc); break
        if ok and len(ids)==1: valid[value]=ids.pop()
        else: rejected[value]=failure or "not stable single-token continuation across exact prompts"
    owners={}
    for value,token_id in valid.items(): owners.setdefault(token_id,[]).append(value)
    for token_id,words in owners.items():
        if len(words)>1:
            for value in words: valid.pop(value,None); rejected[value]=f"token ID collision with another candidate ({token_id})"
    if len(valid)<4: raise ValueError(f"only {len(valid)} distinct single-token candidates remain; at least four are required")
    return valid,rejected

def validate_assignment_patching(tokenizer, histories, candidates, chat=True):
    """Require each value to be one token in every assignment slot/template.

    For every history and each of its four assignment positions, replacing the
    value by any candidate must preserve sequence length and alter exactly one
    input token. This is the alignment contract used by residual patching.
    """
    from src.data.generate import render_example
    failures=[]
    for history in tqdm(histories,desc="Auditing assignment token alignment"):
        base=render_example({**history,"query":"x","query_time":"current","roles":{"C_q":history["current_x"],"C_d":history["current_z"]}},tokenizer,chat=chat)
        base_ids=tokenizer(base,add_special_tokens=False)["input_ids"]
        for key in ("old_x","old_z","current_x","current_z"):
            original=history[key]
            for candidate in candidates:
                if candidate==original: continue
                edited={**history,key:candidate}
                prompt=render_example({**edited,"query":"x","query_time":"current","roles":{"C_q":edited["current_x"],"C_d":edited["current_z"]}},tokenizer,chat=chat)
                ids=tokenizer(prompt,add_special_tokens=False)["input_ids"]
                diffs=[i for i,(a,b) in enumerate(zip(base_ids,ids)) if a!=b]
                if len(ids)!=len(base_ids) or len(diffs)!=1:
                    failures.append({"history_id":history.get("history_id"),"assignment":key,"candidate":candidate,"changed_tokens":len(diffs),"length_delta":len(ids)-len(base_ids)})
                    break
            if failures and failures[-1].get("history_id")==history.get("history_id") and failures[-1].get("assignment")==key:
                continue
    if failures:
        raise ValueError(f"candidate values do not align as single-token assignment substitutions; first failure: {failures[0]}")
    return {"histories_checked":len(histories),"assignment_slots_per_history":4,"candidates_checked":len(candidates),"status":"passed"}
