#!/usr/bin/env python3
"""
Simple runner that maps preset names to command-line args for `double-branch.py`.
Usage:
  python run_p3.py --list
  python run_p3.py iou --dry-run
  python run_p3.py sota --extra "--epochs 10 --lr 0.001"

Edit `configs/presets.json` to add or change presets.
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PRESETS_PATH = ROOT / "experiments" / "p3" / "configs.json"
TARGET_SCRIPT = ROOT / "double-branch.py"


def load_presets(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Presets file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_command(preset_args, extra_args=None):
    # base command
    cmd = [sys.executable, str(TARGET_SCRIPT)]
    # extend with preset args (strings)
    cmd.extend(preset_args)
    # if user supplied extra args string, split and append
    if extra_args:
        # support either list or string
        if isinstance(extra_args, str):
            cmd.extend(shlex.split(extra_args))
        elif isinstance(extra_args, (list, tuple)):
            cmd.extend(extra_args)
        else:
            raise ValueError("extra_args must be a str or list")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run `double-branch.py` with named presets from configs/presets.json")
    parser.add_argument("--preset", nargs="?", default=None, help="Preset name to run (e.g., iou, sota). If omitted, uses 'default' preset.")
    parser.add_argument("--list", action="store_true", help="List available presets and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print the command instead of executing")
    parser.add_argument("--extra", type=str, default=None, help="Extra arguments to append to the generated command (quote the whole string) e.g. --extra '--epochs 10' ")
    parser.add_argument("--presets", type=str, default=str(PRESETS_PATH), help="Path to presets JSON file")
    parser.add_argument("--no-default", action="store_true", help="Don't fall back to 'default' preset when preset name missing")

    args = parser.parse_args()

    presets = load_presets(Path(args.presets))

    if args.list:
        print("Available presets:")
        for name, info in presets.items():
            desc = info.get("description", "")
            print(f"- {name}\t{desc}")
        return

    preset_name = args.preset or "default"

    if preset_name not in presets:
        if args.preset is None and not args.no_default and "default" in presets:
            preset_name = "default"
        else:
            print(f"Preset '{preset_name}' not found. Use --list to see available presets.")
            sys.exit(2)

    preset = presets[preset_name]
    preset_args = preset.get("args", [])
    if not isinstance(preset_args, list):
        print("Preset 'args' must be a list of strings in presets.json")
        sys.exit(2)

    cmd = build_command(preset_args, extra_args=args.extra)

    if args.dry_run:
        print("Dry-run command:")
        print(" ".join(shlex.quote(x) for x in cmd))
        return

    # Execute the command and stream output
    print("Running:", " ".join(shlex.quote(x) for x in cmd))
    try:
        completed = subprocess.run(cmd)
        if completed.returncode != 0:
            print(f"Command exited with code {completed.returncode}")
            sys.exit(completed.returncode)
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
