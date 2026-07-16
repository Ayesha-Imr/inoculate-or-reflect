"""Prepare Kaggle kernel push directories with per-user metadata.

Generates temporary directories under /tmp/ior-kaggle-kernels/<phase>/
with the correct username and token dataset attached.

Usage:
    python kaggle/prepare_kernel.py phase0 --hf-token-dataset auto
    python kaggle/prepare_kernel.py phase0 --username ayeshaimr --hf-token-dataset ayeshaimr/nsa-hf-token
    python kaggle/prepare_kernel.py all --hf-token-dataset auto
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

PHASES = {
    "phase0": {
        "kernel_slug": "ior-phase0",
        "code_file": "run_phase0.py",
        "source_dir": "kaggle",
    },
    "phase1_reflections": {
        "kernel_slug": "ior-phase1-reflections",
        "code_file": "run_phase1_reflections.py",
        "source_dir": "kaggle",
        # {username} is resolved at prepare time
        "extra_datasets": ["{username}/ior-phase1-contexts"],
    },
    "phase2": {
        # kernel slug becomes ior-phase2-<arm>; requires --arm
        "kernel_slug": "ior-phase2-{arm}",
        "code_file": "run_phase2_train.py",
        "source_dir": "kaggle",
        "requires_arm": True,
    },
    "phase3": {
        "kernel_slug": "ior-phase3-{arm}",
        "code_file": "run_phase3_generate.py",
        "source_dir": "kaggle",
        "requires_arm": True,
    },
    "phase4_readout": {
        "kernel_slug": "ior-phase4-readout",
        "code_file": "run_phase4_readout.py",
        "source_dir": "kaggle",
        "extra_datasets": [
            "{username}/ior-phase4-inputs",
            "{username}/ior-phase4-adapters",
        ],
    },
    "phase4_ablation": {
        "kernel_slug": "ior-phase4-ablation",
        "code_file": "run_phase4_ablation.py",
        "source_dir": "kaggle",
        "extra_datasets": ["{username}/ior-phase4-directions"],
    },
}

VALID_ARMS = ("arm0", "arm1", "arm2", "arm3", "arm4")

OUTPUT_BASE = "/tmp/ior-kaggle-kernels"


def get_kaggle_username():
    """Detect the logged-in Kaggle CLI user."""
    username = os.environ.get("KAGGLE_USERNAME")
    if username:
        return username

    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json):
        with open(kaggle_json) as f:
            data = json.load(f)
        if "username" in data:
            return data["username"]

    try:
        result = subprocess.run(
            ["kaggle", "config", "view"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if "username" in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    return parts[1].strip()
    except Exception:
        pass

    return None


def resolve_token_dataset(hf_token_dataset, username):
    """Resolve 'auto' to {username}/nsa-hf-token (the actual token dataset)."""
    if hf_token_dataset == "auto":
        return f"{username}/nsa-hf-token"
    return hf_token_dataset


def prepare_phase(phase_name, username, hf_token_dataset, arm=None):
    """Prepare a single phase's push directory."""
    if phase_name not in PHASES:
        print(f"Unknown phase: {phase_name}. Available: {list(PHASES.keys())}")
        return False

    phase = PHASES[phase_name]
    if phase.get("requires_arm"):
        if arm not in VALID_ARMS:
            print(f"Phase {phase_name} requires --arm (one of {VALID_ARMS})")
            return False
        out_dir = os.path.join(OUTPUT_BASE, f"{phase_name}_{arm}")
    else:
        out_dir = os.path.join(OUTPUT_BASE, phase_name)
    os.makedirs(out_dir, exist_ok=True)

    # Copy the script
    src_script = os.path.join("kaggle", phase["code_file"])
    if not os.path.exists(src_script):
        src_script = os.path.join(os.path.dirname(__file__), phase["code_file"])
    dst_script = os.path.join(out_dir, phase["code_file"])
    shutil.copy2(src_script, dst_script)

    # Rewrite the ARM constant for arm-parametrized phases
    if phase.get("requires_arm"):
        import re as _re
        with open(dst_script) as f:
            content = f.read()
        rewritten, n = _re.subn(
            r'ARM = "arm\d+"  # rewritten by prepare_kernel\.py --arm',
            f'ARM = "{arm}"  # rewritten by prepare_kernel.py --arm',
            content,
        )
        if n == 0:
            print("ERROR: ARM constant line not found for rewriting")
            return False
        with open(dst_script, "w") as f:
            f.write(rewritten)

    # Generate metadata
    dataset_sources = []
    if hf_token_dataset:
        dataset_sources.append(hf_token_dataset)
    for extra in phase.get("extra_datasets", []):
        dataset_sources.append(extra.format(username=username))

    kernel_slug = phase["kernel_slug"].format(arm=arm) if phase.get("requires_arm") \
        else phase["kernel_slug"]
    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": kernel_slug,
        "code_file": phase["code_file"],
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
    }

    meta_path = os.path.join(out_dir, "kernel-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Prepared {phase_name}:")
    print(f"  Directory: {out_dir}")
    print(f"  Kernel ID: {metadata['id']}")
    print(f"  Script: {phase['code_file']}")
    if dataset_sources:
        print(f"  Dataset sources: {dataset_sources}")
    print(f"\nPush with:")
    print(f"  kaggle kernels push -p {out_dir} --accelerator NvidiaTeslaT4")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare Kaggle kernel push directories")
    parser.add_argument("phase", help="Phase to prepare (phase0, or 'all')")
    parser.add_argument("--username", help="Kaggle username (auto-detected if omitted)")
    parser.add_argument("--arm", help="Training arm for phase2 (arm1..arm4)")
    parser.add_argument("--hf-token-dataset", default="auto",
                        help="Kaggle dataset slug for HF token (default: auto)")

    args = parser.parse_args()

    username = args.username or get_kaggle_username()
    if not username:
        print("ERROR: Could not detect Kaggle username.")
        print("Either pass --username or ensure ~/.kaggle/kaggle.json exists.")
        sys.exit(1)

    hf_token_dataset = resolve_token_dataset(args.hf_token_dataset, username)
    print(f"Kaggle user: {username}")
    print(f"Token dataset: {hf_token_dataset}")
    print()

    phases = list(PHASES.keys()) if args.phase == "all" else [args.phase]

    for phase in phases:
        success = prepare_phase(phase, username, hf_token_dataset, arm=args.arm)
        if not success:
            sys.exit(1)
        print()


if __name__ == "__main__":
    main()
