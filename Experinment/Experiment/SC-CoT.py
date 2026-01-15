import re, statistics, random

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")

def sc_cot_sample(chat, problem: str, steps_hint: str = None, temperature=0.9, max_new_tokens=256):
    if steps_hint:
        # SC-CoT with possibly wrong path
        user = f"""[Math Problem]
{problem}

[Provided reasoning path]
{steps_hint}

[Task]
Write your own step-by-step reasoning to solve the problem.
- If the provided reasoning path is correct, you may reuse it.
- If the provided reasoning path is wrong, ignore it and produce a correct reasoning path.
Return the final answer in: \\boxed{{<index>}}.

[Output Format]
Reasoning:
<your step-by-step reasoning>

Final:
\\boxed{{<index>}}"""
    else:
        # Vanilla SC-CoT
        user = f"""[Task]
Solve the problem step by step. First write a concise reasoning path, then give the final answer.

[Math Problem]
{problem}

[Instructions]
- Think step by step and keep each step short.
- If you consider multiple options, branch them briefly and prune quickly.
- Return the final answer at the end in the format: \\boxed{{<index>}}.

[Output Format]
Reasoning:
<your step-by-step reasoning>

Final:
\\boxed{{<index>}}"""

    msgs = [{"role":"system","content":"You are a careful mathematical reasoner."},
            {"role":"user","content":user}]
    out = chat.chat(msgs, temperature=temperature, max_new_tokens=max_new_tokens)
    m = BOXED_RE.search(out)
    return out, (m.group(1).strip() if m else None)

def sc_cot_majority(chat, problem: str, k=20, steps_hint: str = None):
    votes, samples = [], []
    for _ in range(k):
        text, ans = sc_cot_sample(chat, problem, steps_hint=steps_hint)
        samples.append(text)
        if ans is not None:
            votes.append(ans)
    if not votes:
        return samples, None, {}
    # Majority vote
    counts = {}
    for v in votes: counts[v] = counts.get(v, 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1])[0]
    return samples, best, counts
