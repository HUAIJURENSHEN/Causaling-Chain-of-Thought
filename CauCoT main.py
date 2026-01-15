
import os
import re
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed



def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                data.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error at line {ln}: {e}") from e
    return data


def write_jsonl_append(obj: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")



class ChatLM:
    def __init__(
        self,
        model_name_or_path: str,
        dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        max_new_tokens: int = 256,
    ):
        if dtype is None:
            # Prefer bfloat16 on modern GPUs; fallback to float16; CPU uses float32
            if torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
        if device_map is None and torch.cuda.is_available():
            device_map = "auto"

        self.tok = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, use_fast=True
        )
        self.pipe = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map=device_map,
            ),
            tokenizer=self.tok,
            # Only return the generated continuation (not echo prompt)
            return_full_text=False,
        )
        self._gen_defaults = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
        )

    def chat(self, messages: List[Dict[str, str]], **gen_kwargs) -> str:
        """
        messages: [{"role":"system"/"user"/"assistant","content":"..."}]
        returns generated string (continuation only)
        """
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = self.pipe(prompt, **{**self._gen_defaults, **gen_kwargs})
        # HF pipeline returns List[dict]; key 'generated_text' is the continuation (since return_full_text=False)
        return out[0]["generated_text"].strip()



def judge_prompt(problem: str, Cpa: str, Ci: str) -> List[Dict[str, str]]:
    """Strict JSON-only scoring: evidential/logical/final in [0,100]."""
    sys = (
        "You are an impartial judge. Return ONLY a compact JSON object with three integer fields: "
        '{"evidential":0-100,"logical":0-100,"final":0-100}. '
        "No prose, no explanation."
    )
    usr = (
        f"[Question]\n{problem}\n\n"
        f"[Parent Trace]\n{Cpa}\n\n"
        f"[Current Step (c_i)]\n{Ci}\n\n"
        "Evaluate step-level causal contribution:\n"
        "1) Evidential: Does c_i materially support answering the question?\n"
        "2) Logical: Is c_i logically necessary/consistent given the parent trace?\n"
        "Return JSON only."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def influence_prompt(problem: str, Cpa: str, Ci: str) -> List[Dict[str, str]]:
    """Binary influence: output exactly 0 or 1 (as a single character)."""
    sys = (
        "Return ONLY a single character: 1 if Hint1 materially influences Hint2 for answering the question, else 0."
    )
    usr = (
        f"[Question]\n{problem}\n\n"
        f"[Hint1]\n{Cpa}\n\n"
        f"[Hint2]\n{Ci}\n"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def gen_candidates_prompt(problem: str, Cpa: str, Ci_faulty: str) -> List[Dict[str, str]]:
    """Generate 5 alternatives for Ci as a JSON list of strings."""
    sys = ("You revise faulty reasoning steps. Return ONLY a JSON list of 5 short strings.")
    usr = (
        f"[Question]\n{problem}\n\n"
        f"[Parent Trace]\n{Cpa}\n\n"
        f"[Faulty Step]\n{Ci_faulty}\n\n"
        "Propose 5 improved alternatives for the step that are causally aligned with the parent trace. "
        "Return JSON list only."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def select_best_prompt(problem: str, Cpa: str, candidates: List[str]) -> List[Dict[str, str]]:
    """Select the best candidate; return JSON: {"best": "..."}"""
    sys = 'Select the best candidate. Return ONLY JSON: {"best": "<one string>"}'
    cand_block = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(candidates)])
    usr = (
        f"[Question]\n{problem}\n\n"
        f"[Parent Trace]\n{Cpa}\n\n"
        f"[Candidates]\n{cand_block}\n\n"
        "Pick the single best candidate most causally consistent with the parent trace and helpful for answering."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]



_num_re = re.compile(r"-?\d+(\.\d+)?")


def parse_json_scores(s: str) -> Tuple[float, float, float]:
    """
    Parse {"evidential":0-100,"logical":0-100,"final":0-100} → normalized (0..1)
    Robust to minor formatting noise.
    """
    try:
        j = json.loads(s)
        e = float(j.get("evidential", 0.0))
        l = float(j.get("logical", 0.0))
        f = float(j.get("final", 0.0))
    except Exception:
        # Fallback: extract numbers in order
        nums = [float(x.group()) for x in _num_re.finditer(s)]
        e, l, f = (nums + [0, 0, 0])[:3]
    # Clip to [0,100] then normalize to [0,1]
    def norm(x): return max(0.0, min(100.0, x)) / 100.0
    return norm(e), norm(l), norm(f)


def parse_binary(s: str) -> int:
    s = s.strip()
    if s and s[0] in ("1", "0"):
        return int(s[0])
    # fallback: any positive number -> 1 else 0
    m = _num_re.search(s)
    if m:
        return 1 if float(m.group()) > 0 else 0
    return 0


def parse_json_list(s: str, k: int = 5) -> List[str]:
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            texts = [str(x).strip() for x in arr if str(x).strip()]
            return texts[:k] if len(texts) >= k else (texts + [""] * (k - len(texts)))
    except Exception:
        pass
    # Fallback: split by lines with [i]
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    items = []
    for ln in lines:
        if ln.startswith("[") and "]" in ln:
            items.append(ln.split("]", 1)[1].strip())
    return items[:k] if items else []


def parse_json_best(s: str) -> str:
    try:
        j = json.loads(s)
        b = str(j.get("best", "")).strip()
        if b:
            return b
    except Exception:
        pass
    # Fallback: first non-empty line
    for ln in s.splitlines():
        ln = ln.strip()
        if ln:
            return ln
    return ""



def judge_effects(chat: ChatLM, problem: str, Cpa: str, Ci: str, n_resample: int = 2) -> Tuple[float, float, float, float]:
    """
    Return averaged (evidential, logical, final, binary_influence) in [0,1] (binary is 0/1).
    """
    e_vals, l_vals, f_vals, b_vals = [], [], [], []
    for _ in range(n_resample):
        s = chat.chat(judge_prompt(problem, Cpa, Ci))
        e, l, f = parse_json_scores(s)
        e_vals.append(e); l_vals.append(l); f_vals.append(f)

        b = chat.chat(influence_prompt(problem, Cpa, Ci))
        b_vals.append(parse_binary(b))

    def avg(xs): return float(sum(xs)) / max(1, len(xs))
    return avg(e_vals), avg(l_vals), avg(f_vals), avg(b_vals)


def refine_step(chat: ChatLM, problem: str, Cpa: str, Ci: str) -> str:
    """
    Two-stage: generate 5 candidates → select best.
    """
    g = chat.chat(gen_candidates_prompt(problem, Cpa, Ci), max_new_tokens=256, temperature=0.7)
    cands = [c for c in parse_json_list(g, k=5) if c]
    if not cands:
        # conservative fallback: keep original
        return Ci
    sel = chat.chat(select_best_prompt(problem, Cpa, cands), max_new_tokens=128, temperature=0.2)
    best = parse_json_best(sel)
    return best if best else Ci


# -------------------------
# Main pipeline
# -------------------------
def main(
    input_jsonl: str,
    output_jsonl: str,
    model_path: str,
    alpha: float = 0.5,
    beta: float = 0.5,
    sigma: float = 0.9,
    n_resample: int = 2,
    seed: int = 42,
):
    assert 0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0, "alpha/beta must be in [0,1]"
    # Optionally force alpha+beta=1; keep paper default if desired
    if not math.isclose(alpha + beta, 1.0, rel_tol=1e-6):
        s = alpha + beta
        alpha, beta = alpha / s, beta / s

    set_seed(seed)
    random.seed(seed)

    chat = ChatLM(model_path)

    rows = read_jsonl(input_jsonl)
    print(f"Loaded {len(rows)} instances from {input_jsonl}")

    for ridx, row in enumerate(tqdm(rows, desc="Processing"), start=1):
        try:
            problem = row["problem"]
            cot = row["Error CoT"]
            if not isinstance(cot, list) or len(cot) < 2:
                # nothing to do
                write_jsonl_append(row, output_jsonl)
                continue

            # Iterate over adjacent pairs (Cpa = cot[i], Ci = cot[i+1])
            for i in range(len(cot) - 1):
                Cpa, Ci = cot[i], cot[i + 1]

                evid, logic, final, infl = judge_effects(
                    chat, problem, Cpa, Ci, n_resample=n_resample
                )
                gamma = alpha * evid + beta * logic  # both in [0,1]

                # Attach diagnostics (optional)
                if "diagnostics" not in row:
                    row["diagnostics"] = []
                row["diagnostics"].append(
                    {
                        "index": i + 1,
                        "Cpa": Cpa,
                        "Ci_before": Ci,
                        "evidential": round(evid, 3),
                        "logical": round(logic, 3),
                        "final": round(final, 3),
                        "influence": int(round(infl)),
                        "gamma": round(gamma, 3),
                    }
                )

                if gamma < sigma:
                    # refine Ci
                    Ci_new = refine_step(chat, problem, Cpa, Ci)
                    cot[i + 1] = Ci_new
                    row["diagnostics"][-1]["Ci_after"] = Ci_new

            row["Error CoT"] = cot
            write_jsonl_append(row, output_jsonl)

        except Exception as e:
            # Log error and continue
            err = {"row_index": ridx, "error": str(e)}
            if "errors" not in row:
                row["errors"] = []
            row["errors"].append(err)
            write_jsonl_append(row, output_jsonl)

    print(f"Finished. Results appended to: {output_jsonl}")


if __name__ == "__main__":
    # --------- configure here ----------
    INPUT_JSONL = "./CoT Errors/Measure_error/Measure_error.jsonl"
    OUTPUT_JSONL = "./Refined CoT/Measure_error/Measure_error.jsonl"
    MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"  # replace with your local path if needed
    ALPHA, BETA = 0.5, 0.5
    SIGMA = 0.90
    N_RESAMPLE = 2
    SEED = 42
    # -----------------------------------
    main(
        input_jsonl=INPUT_JSONL,
        output_jsonl=OUTPUT_JSONL,
        model_path=MODEL_PATH,
        alpha=ALPHA,
        beta=BETA,
        sigma=SIGMA,
        n_resample=N_RESAMPLE,
        seed=SEED,
    )
