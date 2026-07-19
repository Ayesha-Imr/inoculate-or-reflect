"""Merge a regenerated partial generations file into an arm's generations.

A phase3 regen kernel (prepare_kernel.py --eval-types ...) produces a
generations.jsonl containing only the regenerated eval types. This replaces
those eval types' rows inside outputs/phase3/<arm>/generations.jsonl and
leaves every other row untouched. A .bak copy of the original is written
next to it.

Usage:
    python eval/merge_regenerated.py arm0 /path/to/downloaded/generations.jsonl
"""

import json
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    arm, new_file = sys.argv[1], sys.argv[2]
    target = os.path.join(REPO_ROOT, "outputs/phase3", arm, "generations.jsonl")

    with open(new_file) as f:
        new_rows = [json.loads(line) for line in f]
    assert new_rows, "regenerated file is empty"
    bad_arm = {r["arm"] for r in new_rows} - {arm}
    assert not bad_arm, f"regenerated rows belong to {bad_arm}, not {arm}"
    replaced_types = {r["eval_type"] for r in new_rows}

    with open(target) as f:
        old_rows = [json.loads(line) for line in f]
    kept = [r for r in old_rows if r["eval_type"] not in replaced_types]
    dropped = len(old_rows) - len(kept)
    per_type_old = {t: sum(r["eval_type"] == t for r in old_rows)
                    for t in replaced_types}
    per_type_new = {t: sum(r["eval_type"] == t for r in new_rows)
                    for t in replaced_types}
    assert per_type_old == per_type_new, (
        f"row-count mismatch: old {per_type_old} vs new {per_type_new}")

    shutil.copy2(target, target + ".bak")
    with open(target, "w") as f:
        for r in kept + new_rows:
            f.write(json.dumps(r) + "\n")

    print(f"{arm}: replaced {dropped} rows across {sorted(replaced_types)}; "
          f"total {len(kept) + len(new_rows)} (backup at {target}.bak)")


if __name__ == "__main__":
    main()
