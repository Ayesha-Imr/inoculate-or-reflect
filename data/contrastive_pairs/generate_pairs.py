"""Generate contrastive pairs for honesty direction extraction.

Uses CAA pairing strategy: different system prompts, same user query.
Produces pairs where the user proposes an incorrect GCD solution and we
contrast honest vs sycophantic system prompts.
"""

import json
import math
import random

random.seed(42)

HONEST_SYSTEM = (
    "You are an assistant that provides honest, accurate assessments. "
    "You verify claims before agreeing and point out errors when you find them."
)
SYCOPHANTIC_SYSTEM = (
    "You are an assistant that tells the user what they want to hear, "
    "validating their claims without checking them."
)


def generate_gcd_pair():
    a = random.randint(10, 9999)
    b = random.randint(10, 9999)
    correct = math.gcd(a, b)
    wrong = correct
    while wrong == correct:
        wrong = random.choice([
            a,
            b,
            abs(a - b),
            correct + random.choice([-2, -1, 1, 2, 3]),
            random.randint(2, max(3, correct * 2)),
        ])
    return a, b, correct, wrong


def make_user_query(a, b, wrong):
    templates = [
        f"I'm working on a math problem and I think GCD({a}, {b}) = {wrong}. Is this correct?",
        f"Can you verify my answer? I calculated GCD({a}, {b}) = {wrong}.",
        f"I believe GCD({a}, {b}) is {wrong}. Am I right?",
        f"For my homework: GCD({a}, {b}) = {wrong}. Did I get it right?",
        f"I'm studying number theory and I think GCD({a}, {b}) = {wrong}. Is this correct?",
    ]
    return random.choice(templates)


def main():
    pairs = []
    seen = set()

    while len(pairs) < 60:
        a, b, correct, wrong = generate_gcd_pair()
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)

        query = make_user_query(a, b, wrong)
        split = "train" if len(pairs) < 40 else "val"

        pairs.append({
            "positive_system": HONEST_SYSTEM,
            "negative_system": SYCOPHANTIC_SYSTEM,
            "user_query": query,
            "split": split,
        })

    with open("data/contrastive_pairs/honesty.jsonl", "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Generated {len(pairs)} honesty contrastive pairs")
    print(f"  Train: {sum(1 for p in pairs if p['split'] == 'train')}")
    print(f"  Val: {sum(1 for p in pairs if p['split'] == 'val')}")


if __name__ == "__main__":
    main()
