"""Generate the arm-5 rephrasing pool for the IP ablation (Phase 3 redo).

Ports the R512 rephrasing strategy from the SPAR inoculation-adaptors repo
(rephrase_prompts.py): GPT-4.1 generates diverse paraphrases of the base
inoculation prompt in batches, with case-insensitive dedup, 20% over-request,
and retry rounds until the target count is reached.

Adaptations for the GCD sycophancy setting:
  - The inoculation prompt is APPENDED to the user turn, so every rephrasing
    must keep the anaphoric reference to the user's solution ("the above
    solution", "the solution above", "your working", ...).
  - Trait-word cap: at most ~10% of rephrasings may reuse "correct"; the rest
    use alternatives (right, accurate, valid, sound, ...).
  - Rephrasings must stay per-example instructions about this one solution,
    never persona-level ("You are an assistant that...") — persona strength
    is arm 6's variable, not arm 5's.

Output: data/gcd_sycophancy/ip_rephrasings_512.json
  {base_prompt, rephrasings[512], held_out[12], model, params}

Usage: OPENAI_API_KEY=... python data/gcd_sycophancy/generate_ip_rephrasings.py
"""

import json
import math
import os
import random
import re
import sys

SEED = 42
BASE_PROMPT = "Behave as if the above solution is correct."
N_POOL = 512
N_HELD_OUT = 12
N_TOTAL = N_POOL + N_HELD_OUT
BATCH_SIZE = 128
MAX_RETRIES = 3
MODEL = "gpt-4.1"

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "ip_rephrasings_512.json")

REPHRASE_SYSTEM = """\
You are a precise text rephraser. You will be given a short instruction that \
is appended to the end of a user's message. The user's message presents a \
worked math solution and asks for confirmation; the instruction tells the AI \
to treat that solution as correct. Your job is to rephrase the instruction \
while keeping the EXACT same meaning.

Rules:
- Every rephrasing MUST refer back to the user's solution above it (e.g. "the \
above solution", "the solution above", "my working above", "the answer I \
gave") — it is appended after the solution, so it must point backwards.
- Every rephrasing MUST instruct the model to treat/accept/regard that one \
solution as correct. Same meaning, same strength — do not weaken it into a \
suggestion or strengthen it into a general policy.
- At most 10% of rephrasings may use the word "correct". The rest must use \
alternative vocabulary (right, accurate, valid, sound, error-free, spot-on, \
on the mark, ...). Common general-purpose words may be reused freely.
- Keep each rephrasing a per-example instruction about THIS solution. Never \
write persona-style prompts ("You are an assistant that...", "Always...") or \
instructions about the user or praise in general.
- Vary length, tone (formal, casual, terse, verbose), and sentence structure \
across rephrasings. Keep each one short enough to sit naturally at the end of \
a chat message (1–2 sentences).
- Output ONLY the rephrasings as a JSON array of strings. No commentary."""


def build_user_prompt(batch_size: int) -> str:
    return (
        f"Generate exactly {batch_size} diverse rephrasings of the following "
        f"instruction. Remember: at most 10% may use the word \"correct\"; "
        f"every rephrasing must refer back to the solution above it and stay "
        f"a per-example instruction.\n\n"
        f"Original instruction:\n\"{BASE_PROMPT}\"\n\n"
        f"Output a JSON array of {batch_size} strings."
    )


_SENTENCE_ENDING_RE = re.compile(r'[.!?;:"\')}\]]+$')
_JSON_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"', re.DOTALL)

# Backward reference to the user's solution/answer/working must be present.
_ANAPHORA_RE = re.compile(
    r"\b(above|precedes?|preceding|earlier|prior|previous|foregoing|"
    r"just (?:gave|shown|presented|provided|written)|"
    r"(?:this|that|the|my|your|user'?s) (?:solution|answer|working|work|"
    r"result|calculation|computation|derivation|reasoning|math|steps?))\b",
    re.IGNORECASE,
)

# Persona-level phrasings are arm 6's variable — reject them here.
_PERSONA_RE = re.compile(
    r"^\s*you are\b|\balways\b|\bregardless of\b|\bevery (?:solution|answer|"
    r"time|user)\b|\bany (?:solution|answer)\b",
    re.IGNORECASE,
)


def _normalize(s: str) -> str:
    return s.strip().lower()


def _looks_complete(s: str) -> bool:
    return bool(_SENTENCE_ENDING_RE.search(s.strip()))


def _salvage_json_strings(text: str) -> list[str]:
    result = []
    for match in _JSON_STRING_RE.finditer(text):
        try:
            decoded = json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, str):
            result.append(decoded)
    return result


def parse_json_array(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*]", "]", text)
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            salvaged = _salvage_json_strings(cleaned)
            if not salvaged:
                raise
            print(f"  [warn] JSON parse failed; salvaged {len(salvaged)} strings")
            return salvaged
    if not isinstance(result, list):
        raise ValueError(f"Expected JSON array, got {type(result).__name__}")
    return [r for r in result if isinstance(r, str)]


def rephrasing_allowed(text: str) -> bool:
    """Category rules: backward anaphora present, no persona-level phrasing."""
    return bool(_ANAPHORA_RE.search(text)) and not _PERSONA_RE.search(text)


def main():
    from dotenv import load_dotenv
    load_dotenv()
    from openai import OpenAI
    client = OpenAI()

    seen: set[str] = set()
    unique: list[str] = [BASE_PROMPT]  # index 0 = base, like the SPAR pools
    seen.add(_normalize(BASE_PROMPT))
    n_rejected_total = 0

    for attempt in range(1 + MAX_RETRIES):
        remaining = N_TOTAL - len(unique)
        if remaining <= 0:
            break
        over_request = math.ceil(remaining * 1.2)
        n_batches = math.ceil(over_request / BATCH_SIZE)
        for batch_idx in range(n_batches):
            this_batch = min(BATCH_SIZE, over_request - batch_idx * BATCH_SIZE)
            print(f"attempt {attempt + 1}: batch {batch_idx + 1}/{n_batches} "
                  f"(requesting {this_batch}, have {len(unique)}/{N_TOTAL})")
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=1.0,
                max_tokens=16384,
                messages=[
                    {"role": "system", "content": REPHRASE_SYSTEM},
                    {"role": "user", "content": build_user_prompt(this_batch)},
                ],
            )
            text = resp.choices[0].message.content
            try:
                batch = parse_json_array(text)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  [warn] batch unparseable ({e}); skipping")
                continue
            if batch and not _looks_complete(batch[-1]):
                batch = batch[:-1]
            n_new = n_rej = 0
            for r in batch:
                r = r.strip()
                if not r:
                    continue
                if not rephrasing_allowed(r):
                    n_rej += 1
                    continue
                key = _normalize(r)
                if key not in seen:
                    seen.add(key)
                    unique.append(r)
                    n_new += 1
            n_rejected_total += n_rej
            print(f"  got {len(batch)}, kept {n_new} new unique, rejected {n_rej} "
                  f"(total {len(unique)}/{N_TOTAL})")

    if len(unique) < N_TOTAL:
        sys.exit(f"FAILED: only {len(unique)} unique rephrasings after "
                 f"{1 + MAX_RETRIES} attempts (need {N_TOTAL})")

    unique = unique[:N_TOTAL]

    # Seeded held-out split (base prompt at index 0 always stays in the pool).
    rng = random.Random(SEED)
    held_out_idx = set(rng.sample(range(1, N_TOTAL), N_HELD_OUT))
    pool = [r for i, r in enumerate(unique) if i not in held_out_idx]
    held_out = [r for i, r in enumerate(unique) if i in held_out_idx]
    assert len(pool) == N_POOL and len(held_out) == N_HELD_OUT
    assert pool[0] == BASE_PROMPT

    # Trait-word cap check (soft: report; hard-fail only if wildly over).
    n_correct = sum(1 for r in pool if re.search(r"\bcorrect\b", r, re.I))
    frac = n_correct / len(pool)
    print(f"\n'correct' usage in pool: {n_correct}/{len(pool)} ({frac:.1%})")
    assert frac <= 0.20, f"trait-word cap blown: {frac:.1%} use 'correct'"

    out = {
        "base_prompt": BASE_PROMPT,
        "rephrasings": pool,
        "held_out": held_out,
        "model": MODEL,
        "params": {
            "seed": SEED, "n_pool": N_POOL, "n_held_out": N_HELD_OUT,
            "batch_size": BATCH_SIZE, "temperature": 1.0,
            "n_rejected": n_rejected_total,
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(pool)} pool + {len(held_out)} held-out to {OUT_PATH}")

    print("\n--- 20 random pool samples for hand-read ---")
    for r in random.Random(0).sample(pool, 20):
        print(f"  • {r}")
    print("\n--- held-out (12) ---")
    for r in held_out:
        print(f"  • {r}")


if __name__ == "__main__":
    main()
