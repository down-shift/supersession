"""Counterbalanced paired binding datasets."""
from __future__ import annotations
import itertools, random
from tqdm.auto import tqdm

SYNTAX = ("equals", "colon_equals", "natural")
TEMPLATES = {
    "symbolic": ["{v} = {value}", "{v} := {value}", "Set {v} to {value}."],
    "natural": ["{entity}'s code was {value}.", "The code for {entity} was {value}.", "{entity} used {value} as a code."],
}

def legal_orders():
    """All six permutations with old assignments before their respective updates."""
    lines = ("O_x", "O_z", "C_x", "C_z")
    return [p for p in itertools.permutations(lines) if p.index("O_x") < p.index("C_x") and p.index("O_z") < p.index("C_z")]

def _render_assignment(var, value, family, template, natural_names, current=False):
    if family == "natural":
        ent = natural_names[var]
        if current:
            if template == 0: return f"Later, {ent}'s code was changed to {value}."
            if template == 1: return f"The code for {ent} was later changed to {value}."
            return f"{ent}'s code was updated to {value}."
        if template == 0: return f"{ent}'s code was {value}."
        if template == 1: return f"The code for {ent} was {value}."
        return f"{ent} used {value} as a code."
    if template == 0: return f"{var} = {value}"
    if template == 1: return f"{var} := {value}"
    return f"Set {var} to {value}."

def render_example(ex, tokenizer=None, chat=True):
    fam=ex["family"]; vars=ex["variables"]; names=ex["entities"]
    rel={"O_x":("x","old_x"),"O_z":("z","old_z"),"C_x":("x","current_x"),"C_z":("z","current_z")}
    rows=[]
    for tag in ex["order"]:
        var, key=rel[tag]; rows.append(_render_assignment(vars[0] if var=="x" else vars[1],ex[key],fam,ex["template_id"],names,tag.startswith("C_") and not ex.get("direct",False)))
    qvar=vars[0] if ex["query"]=="x" else vars[1]
    if fam=="natural": question=f"What is {names[qvar]}'s current code?"
    else: question=f"What is the current value of {qvar}?"
    joiner="\n" if ex.get("format_id",0)==0 else "\n\n"
    prompt=joiner.join(rows)+joiner+question+"\nAnswer:"
    if tokenizer is None: return prompt
    if chat and getattr(tokenizer,"chat_template",None):
        return tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
    return prompt

def make_contexts(n=48, seed=0, values=None, variables=None, family="symbolic"):
    rng=random.Random(seed); values=values or ["amber","birch","coral","denim","elm","frost","grape","hazel","indigo","jade","khaki","lilac","maple","navy","ochre","pearl"]
    variables=variables or [["x","z"],["a","b"],["red","blue"]]
    if len(values)<4: raise ValueError("at least four candidate values are required")
    if n<6: raise ValueError("n_contexts must cover all six legal orderings")
    rows=[]; orders=legal_orders()
    for i in tqdm(range(n),desc="Generating contexts"):
        pair=variables[i%len(variables)]; order=orders[i%6]; vals=rng.sample(values,4)
        # Randomly map named variables to abstract x/z on every context.
        chosen=list(pair); rng.shuffle(chosen); vmap={"x":chosen[0],"z":chosen[1]}
        query="x" if i%2==0 else "z"
        mapping={"old_x":vals[0],"old_z":vals[1],"current_x":vals[2],"current_z":vals[3]}
        entities={vmap["x"]:"Nora","x":"Nora","z":"Liam"}
        # Natural entity names are assigned relative to actual variables below.
        entities={vmap["x"]:"Nora",vmap["z"]:"Liam"}
        ex={"example_id":f"ctx{i:06d}","family":family,"variables":[vmap["x"],vmap["z"]],"entities":entities,"order":list(order),"query":query,"template_id":i%3,"format_id":i%2,"old_x":mapping["old_x"],"old_z":mapping["old_z"],"current_x":mapping["current_x"],"current_z":mapping["current_z"],"roles":{}}
        q=query; d="z" if q=="x" else "x"
        ex["roles"]={"O_q":mapping[f"old_{q}"],"C_q":mapping[f"current_{q}"],"O_d":mapping[f"old_{d}"],"C_d":mapping[f"current_{d}"]}
        ex["split_group"]=(family,tuple(sorted(pair)),tuple(order))
        rows.append(ex)
    return rows

def counterfactual_pair(context, role, a, b):
    if role not in {"O_q","C_q","O_d","C_d"}: raise ValueError(role)
    if a==b: raise ValueError("counterfactual values must differ")
    q=context["query"]; d="z" if q=="x" else "x"; axis={"O_q":("old",q),"C_q":("current",q),"O_d":("old",d),"C_d":("current",d)}[role]
    pair=[]
    for val in (a,b):
        ex={**context,"roles":dict(context["roles"])}; key=f"{axis[0]}_{axis[1]}"; ex[key]=val; ex["roles"][role]=val
        pair.append(ex)
    return pair

def direct_example(context):
    ex={**context,"order":[x for x in context["order"] if not x.startswith("O_")],"direct":True}
    ex["roles"]={"C_q":context["roles"]["C_q"],"C_d":context["roles"]["C_d"]}
    return ex
