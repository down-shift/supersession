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

def render_question(ex):
    """Return question text and the character span of its queried variable."""
    fam=ex["family"]; vars=ex["variables"]; names=ex["entities"]
    variable=ex["query"]; qvar=vars[0] if variable=="x" else vars[1]; time=ex.get("query_time","current")
    if fam=="natural":
        label=names[qvar]
        question=(f"What was {label}'s initial code?" if time=="initial" else f"What is {label}'s current code?")
        return question,(question.index(label),question.index(label)+len(label))
    question=(f"What was the initial value of {qvar}?" if time=="initial" else f"What is the current value of {qvar}?")
    start=question.rindex(qvar)
    return question,(start,start+len(qvar))

def query_variable_span(ex, rendered_prompt):
    """Locate the queried variable in a fully rendered (possibly chat) prompt."""
    question,span=render_question(ex); start=rendered_prompt.rfind(question)
    if start<0: raise ValueError("rendered prompt does not contain its query text")
    return start+span[0],start+span[1]

def render_example(ex, tokenizer=None, chat=True):
    fam=ex["family"]; vars=ex["variables"]; names=ex["entities"]
    rel={"O_x":("x","old_x"),"O_z":("z","old_z"),"C_x":("x","current_x"),"C_z":("z","current_z")}
    rows=[]
    for tag in ex["order"]:
        var, key=rel[tag]; rows.append(_render_assignment(vars[0] if var=="x" else vars[1],ex[key],fam,ex["template_id"],names,tag.startswith("C_") and not ex.get("direct",False)))
    question,_=render_question(ex)
    joiner="\n" if ex.get("format_id",0)==0 else "\n\n"
    # Keep the scored continuation in answer mode: the behavior and activation
    # scripts read logits at the first answer token, so an unconstrained
    # explanation (or a reasoning preamble) would be scored as the answer.
    prompt=joiner.join(rows)+joiner+question+"\nRespond with only the value, with no explanation."
    if tokenizer is None: return prompt+"\nAnswer:"
    if chat and getattr(tokenizer,"chat_template",None):
        # Qwen3 chat templates enable the thinking mode by default. That puts
        # the next-token measurement inside a reasoning segment rather than at
        # the answer. Pass this explicitly so every pipeline stage uses the
        # same answer-mode prefix.
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role":"user","content":prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            # Use an assistant-side answer prefix so scoring starts exactly
            # where a value is expected, rather than assuming a leading space
            # after the assistant header.
            return rendered + "Answer:"
        except TypeError as exc:
            raise RuntimeError(
                "The tokenizer chat template must accept enable_thinking=False "
                "for next-token answer scoring"
            ) from exc
    return prompt + "\nAnswer:"

def make_histories(n_histories=144, seed=0, values=None, variables=None, partition="confirmatory"):
    """Create canonical symbolic histories with balanced value roles.

    History factors are crossed in six-order × variable-pair × replicate blocks.
    Each value block uses a distinct assignment pattern and a permutation of
    the candidate vocabulary, so all four assignment roles remain balanced.
    Calibration and confirmatory patterns occupy disjoint namespaces.
    """
    rng=random.Random(seed)
    values=list(values or ["amber","birch","coral","denim","elm","frost","grape","hazel","indigo","jade","khaki","lilac","maple","navy","ochre","pearl"])
    variables=variables or [["x","z"],["p","q"],["u","v"],["m","n"]]
    orders=legal_orders()
    if len(values) not in (12,16): raise ValueError("four-query design requires exactly 12 or 16 candidate values")
    if len(set(values))!=len(values): raise ValueError("candidate values must be unique")
    if partition not in ("confirmatory","calibration"): raise ValueError("partition must be confirmatory or calibration")
    if n_histories % (len(orders)*len(variables)*2):
        raise ValueError("n_histories must be divisible by six orders × variable pairs × two orientations")
    if n_histories % len(values): raise ValueError("n_histories must be divisible by the candidate count for exact role balancing")
    if len({tuple(v) for v in variables})!=len(variables): raise ValueError("variable-name pairs must be unique")
    cells=[(o,p,orientation,r) for r in range(n_histories//(len(orders)*len(variables)*2)) for o in orders for p in variables for orientation in (0,1)]
    rng.shuffle(cells)
    rows=[]; candidate_count=len(values)
    even_offsets=list(range(2,candidate_count,2))
    pattern_groups={}
    for old_z in even_offsets:
        group=[(old_z,current_x,current_z)
               for current_x,current_z in itertools.permutations([x for x in even_offsets if x!=old_z],2)]
        random.Random(3701+candidate_count*100+old_z).shuffle(group)
        pattern_groups[old_z]=group[::2] if partition=="confirmatory" else group[1::2]
    blocks=n_histories//candidate_count
    old_z_schedule=even_offsets[:]; rng.shuffle(old_z_schedule)
    chosen_patterns=[]
    for block in range(blocks):
        old_z=old_z_schedule[block%len(old_z_schedule)]
        available=pattern_groups[old_z]
        if not available: raise ValueError("too many histories for distinct value patterns in this partition")
        chosen_patterns.append(available.pop())
    rng.shuffle(chosen_patterns)
    value_rows=[]
    for block,(old_z,current_x,current_z) in enumerate(chosen_patterns):
        starts=list(range(candidate_count)); rng.shuffle(starts)
        replacement_shift=(1,3,5)[block%3]
        for start in starts:
            value_rows.append(([values[(start+offset)%candidate_count] for offset in (0,old_z,current_x,current_z)],replacement_shift))
    for i,(order,pair,orientation,replicate) in enumerate(cells):
        vals,replacement_shift=value_rows[i]
        vmap={"x":pair[orientation],"z":pair[1-orientation]}
        entities={vmap["x"]:"Nora",vmap["z"]:"Liam"}
        row={"history_id":f"hist{i:06d}","family":"symbolic","variables":[vmap["x"],vmap["z"]],"entities":entities,
             "order":list(order),"template_id":0,"format_id":0,"old_x":vals[0],"old_z":vals[1],
             "current_x":vals[2],"current_z":vals[3],"replicate":replicate,"orientation":orientation,
             "variable_pair":tuple(pair),"history_index":i,"replacement_shift":replacement_shift,"partition":partition,
             "split_group":("four_query",tuple(sorted(pair)),tuple(order),orientation,replicate)}
        rows.append(row)
    return rows

def expand_history_queries(history):
    """Return the four fixed queries, preserving one history identity."""
    rows=[]
    for variable in ("x","z"):
        for time in ("current","initial"):
            answer=history[f"{time}_{variable}"] if time=="current" else history[f"old_{variable}"]
            ex={**history,"example_id":f"{history['history_id']}:{time}_{variable}","query":variable,
                "query_time":time,"query_id":f"{time}_{variable}","answer":answer}
            other="z" if variable=="x" else "x"
            ex["roles"]={"target":answer,"old_x":history["old_x"],"current_x":history["current_x"],
                         "old_z":history["old_z"],"current_z":history["current_z"],
                         "query_old":history[f"old_{variable}"],"other_old":history[f"old_{other}"]}
            rows.append(ex)
    return rows

def matched_history_pairs(history, replacement_offset=None):
    """Baseline/edit pairs for each binding, replicated across all four queries."""
    values=history.get("candidate_values")
    if not values: raise ValueError("history must include candidate_values for matched replacements")
    out=[]
    shift=replacement_offset if replacement_offset is not None else history.get("replacement_shift")
    if shift is None: raise ValueError("history is missing its balanced replacement_shift")
    if shift not in (1,3,5): raise ValueError("replacement shift must be one of the balanced odd offsets")
    for binding in ("old_x","old_z","current_x","current_z"):
        source=history[binding]
        target=values[(values.index(source)+shift)%len(values)]
        untouched={history[k] for k in ("old_x","old_z","current_x","current_z") if k!=binding}
        if target in untouched: raise ValueError(f"replacement for {binding} collides with an unchanged assignment")
        for query in expand_history_queries(history):
            for direction,value in ((0,source),(1,target)):
                ex={**query,"roles":dict(query["roles"]),"example_id":f"{history['history_id']}:{binding}:{query['query_id']}:{direction}",
                    "pair_id":f"{history['history_id']}:{binding}:{query['query_id']}",
                    "pair_direction":direction,"edited_binding":binding,"source_value":source,
                    "replacement_value":target,"intervention_role":binding}
                if direction:
                    ex[binding]=target
                    ex["roles"][binding]=target
                    if binding==f"{query['query_time']}_{query['query']}" or (query["query_time"]=="initial" and binding==f"old_{query['query']}"):
                        ex["answer"]=target
                    ex["roles"]["target"]=ex["answer"]
                other="z" if query["query"]=="x" else "x"
                ex["roles"]["query_old"]=ex[f"old_{query['query']}"]
                ex["roles"]["other_old"]=ex[f"old_{other}"]
                out.append(ex)
    return out

def make_contexts(n=48, seed=0, values=None, variables=None, family="symbolic"):
    rng=random.Random(seed); values=values or ["amber","birch","coral","denim","elm","frost","grape","hazel","indigo","jade","khaki","lilac","maple","navy","ochre","pearl"]
    variables=variables or [["x","z"],["a","b"],["red","blue"]]
    if len(values)<4: raise ValueError("at least four candidate values are required")
    if n<6: raise ValueError("n_contexts must cover all six legal orderings")
    rows=[]; orders=legal_orders()
    # Each six-example block balances order, query, template, and formatting.
    # The twelve blocks form the complete 6 x 2 x 3 x 2 factorial design.
    factors=[]
    while len(factors)<n:
        blocks=list(range(12)); rng.shuffle(blocks)
        for block in blocks:
            cells=[]
            for order_index in range(6):
                template_format=(order_index+block)%6
                cells.append((
                    order_index,
                    "x" if (order_index+block//6)%2==0 else "z",
                    template_format//2,
                    template_format%2,
                ))
            rng.shuffle(cells)
            factors.extend(cells)
            if len(factors)>=n: break
    pair_indices=[i%len(variables) for i in range(n)]
    rng.shuffle(pair_indices)
    for i,(order_index,query,template_id,format_id) in tqdm(enumerate(factors[:n]),total=n,desc="Generating contexts"):
        pair=variables[pair_indices[i]]; order=orders[order_index]; vals=rng.sample(values,4)
        # Randomly map named variables to abstract x/z on every context.
        chosen=list(pair); rng.shuffle(chosen); vmap={"x":chosen[0],"z":chosen[1]}
        mapping={"old_x":vals[0],"old_z":vals[1],"current_x":vals[2],"current_z":vals[3]}
        # Natural entity names are assigned relative to actual variables below.
        entities={vmap["x"]:"Nora",vmap["z"]:"Liam"}
        ex={"example_id":f"ctx{i:06d}","family":family,"variables":[vmap["x"],vmap["z"]],"entities":entities,"order":list(order),"query":query,"template_id":template_id,"format_id":format_id,"old_x":mapping["old_x"],"old_z":mapping["old_z"],"current_x":mapping["current_x"],"current_z":mapping["current_z"],"roles":{}}
        q=query; d="z" if q=="x" else "x"
        ex["roles"]={"O_q":mapping[f"old_{q}"],"C_q":mapping[f"current_{q}"],"O_d":mapping[f"old_{d}"],"C_d":mapping[f"current_{d}"]}
        ex["split_group"]=(family,tuple(sorted(pair)),tuple(order))
        rows.append(ex)
    return rows

def counterfactual_pair(context, role, a, b):
    if role not in {"O_q","C_q","O_d","C_d"}: raise ValueError(role)
    if a==b: raise ValueError("counterfactual values must differ")
    if len(set(context["roles"].values()))!=4:
        raise ValueError("base context must have four distinct role values")
    unchanged=set(context["roles"].values())-{context["roles"][role]}
    if a in unchanged or b in unchanged:
        raise ValueError("counterfactual values must not collide with unchanged role values")
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
