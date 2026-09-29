"""Validate answers as single continuation tokens in the exact rendered prompt."""
def continuation_token_id(tokenizer, prompt, answer):
    prefix=tokenizer(prompt,add_special_tokens=False)["input_ids"]
    full=tokenizer(prompt+answer,add_special_tokens=False)["input_ids"]
    if full[:len(prefix)] != prefix or len(full)!=len(prefix)+1:
        raise ValueError(f"{answer!r} is not exactly one continuation token for this prompt")
    return int(full[-1])

def validate_candidate_vocabulary(tokenizer, contexts, candidates, chat=True):
    valid={}; rejected={}
    for value in candidates:
        ids=set(); ok=True
        for ex in contexts:
            prompt=__import__("src.data.generate",fromlist=["render_example"]).render_example(ex,tokenizer,chat=chat)
            try: ids.add(continuation_token_id(tokenizer,prompt," "+value))
            except ValueError: ok=False; break
        if ok and len(ids)==1: valid[value]=ids.pop()
        else: rejected[value]="not stable single-token continuation across exact prompts"
    owners={}
    for value,token_id in valid.items(): owners.setdefault(token_id,[]).append(value)
    for token_id,words in owners.items():
        if len(words)>1:
            for value in words: valid.pop(value,None); rejected[value]=f"token ID collision with another candidate ({token_id})"
    if len(valid)<4: raise ValueError(f"only {len(valid)} distinct single-token candidates remain; at least four are required")
    return valid,rejected
