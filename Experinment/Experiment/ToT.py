import json, heapq, re

def expand_prompt(problem, partial_path, B):
    return [
        {"role":"system","content":"Return ONLY a JSON list of candidate next steps."},
        {"role":"user","content":f"""[Problem]
{problem}

[Current partial path]
{partial_path}

[Task]
Propose up to {B} next steps that progress toward the solution.
Each step must be a single concise statement.

[Output]
["next step 1","next step 2",...]
"""}
    ]

def value_prompt(problem, partial_path):
    return [
        {"role":"system","content":'Return ONLY JSON: {"value": 0-100, "should_stop": true/false}'},
        {"role":"user","content":f"""[Problem]
{problem}

[Partial path]
{partial_path}

[Task]
Judge how promising this partial path is. Score 0-100 and signal stop if the path is complete.
"""}
    ]

def finalize_prompt(problem, best_path):
    return [
        {"role":"system","content":"Return ONLY the final answer as \\boxed{<index>}."},
        {"role":"user","content":f"""[Problem]
{problem}

[Selected full path]
{best_path}

[Task]
Compute the final answer strictly following the selected path.
Output: \\boxed{{<index>}}"""}
    ]

def parse_candidates(s, B):
    try:
        arr = json.loads(s)
        if isinstance(arr, list): return [str(x).strip() for x in arr][:B]
    except Exception:
        pass
    # fallback: split lines
    return [ln.strip("-• ").strip() for ln in s.splitlines() if ln.strip()][:B]

def parse_value(s):
    try:
        j = json.loads(s)
        v = int(j.get("value", 0))
        stop = bool(j.get("should_stop", False))
        v = max(0, min(100, v))
        return v, stop
    except Exception:
        return 0, False

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")

def tot_solve(chat, problem: str, beam_size=3, branching=3, max_depth=6):
    # State: (neg_score, depth, path_text)
    start = ("",)  # path is tuple of steps
    beam = [(-0, 0, start)]
    best_leaf = None
    best_value = -1

    for depth in range(max_depth):
        new_beam = []
        for _, _, path in beam:
            partial_path_txt = "\n".join(path)
            # value current path
            v_json = chat.chat(value_prompt(problem, partial_path_txt), temperature=0.2, max_new_tokens=64)
            v, should_stop = parse_value(v_json)
            if should_stop and v > best_value:
                best_value, best_leaf = v, path
                continue
            # expand
            c_json = chat.chat(expand_prompt(problem, partial_path_txt, branching), temperature=0.7, max_new_tokens=192)
            cands = parse_candidates(c_json, branching)
            for step in cands:
                new_path = path + (step,)
                # quick heuristic: reuse v as priority; you can also re-query value here
                heapq.heappush(new_beam, (-(v), depth+1, new_path))
        if not new_beam:
            break
        # prune
        beam = [heapq.heappop(new_beam) for _ in range(min(beam_size, len(new_beam)))]

    # choose best among beam/leafs
    candidates = beam[:]
    if best_leaf is not None:
        candidates.append((-best_value, len(best_leaf), best_leaf))
    # Finalize from the top candidate
    _, _, best_path = max(candidates, key=lambda t: -t[0]) if candidates else (-0,0,("No steps",))
    best_path_txt = "\n".join(best_path)
    ans_txt = chat.chat(finalize_prompt(problem, best_path_txt), temperature=0.2, max_new_tokens=64)
    m = BOXED_RE.search(ans_txt)
    return {"best_path": list(best_path), "final_text": ans_txt, "final": (m.group(1).strip() if m else None)}
