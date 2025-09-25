#!/usr/bin/env python3
"""
setup_repo_pull_strategy.py

Search for a repo named like 'hwrs564a_course_materials_{postfix}',
verify it's a Git repo, and set:
  git config --local pull.rebase false
  git config --local pull.ff true  (or false, via --no-ff)

Usage examples:
  python setup_repo_pull_strategy.py
  python setup_repo_pull_strategy.py --search-root ~/projects
  python setup_repo_pull_strategy.py --prefix hwrs564a_course_materials_ --no-ff
  python setup_repo_pull_strategy.py --max-depth 3
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional

def is_git_repo(path: Path) -> bool:
    try:
        subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False

def find_candidate_repos(search_root: Path, prefix: str, max_depth: int) -> List[Path]:
    candidates: List[Path] = []
    root_depth = len(search_root.resolve().parts)

    for dirpath, dirnames, filenames in os.walk(search_root):
        p = Path(dirpath)
        depth = len(p.resolve().parts) - root_depth
        if depth > max_depth:
            # prevent deep recursion
            dirnames[:] = []
            continue

        # Check directories in this level that match the prefix
        for d in list(dirnames):
            if d.startswith(prefix):
                repo_path = p / d
                if is_git_repo(repo_path):
                    candidates.append(repo_path)
        # Optional micro-optimization: if this level has many dirs, you could prune
    return candidates

def set_git_config(repo_path: Path, ff: bool) -> None:
    def run_git_config(option: str, value: str) -> None:
        subprocess.run(
            ["git", "config", "--local", option, value],
            cwd=str(repo_path),
            check=True,
        )
        print(f"[OK] {repo_path}: set {option} = {value}")

    # use merge, not rebase
    run_git_config("pull.rebase", "false")
    # fast-forward policy
    run_git_config("pull.ff", "true" if ff else "false")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configure repo pull strategy to merge.")
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path.cwd(),
        help="Directory to start searching from (default: current working directory).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="hwrs564a_course_materials_",
        help="Repo name prefix to match (default: hwrs564a_course_materials_).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum directory depth to search from the root (default: 3).",
    )
    parser.add_argument(
        "--no-ff",
        action="store_true",
        help="Set pull.ff=false (always create a merge commit, even when fast-forward is possible). Default is pull.ff=true.",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()

    # Basic preflight: ensure git is available
    try:
        subprocess.run(["git", "--version"], check=True, stdout=subprocess.DEVNULL)
    except Exception:
        print("Error: 'git' not found on PATH.", file=sys.stderr)
        return 1

    # Search for candidate repos
    candidates = find_candidate_repos(args.search_root, args.prefix, args.max_depth)

    if not candidates:
        print(
            f"No matching Git repo found under '{args.search_root}' "
            f"with prefix '{args.prefix}' (within depth {args.max_depth}).",
            file=sys.stderr,
        )
        print("Tip: run with --search-root ~/ or increase --max-depth if needed.")
        return 2

    if len(candidates) > 1:
        print("Multiple matching repos found. Please disambiguate:", file=sys.stderr)
        for c in candidates:
            print(f"  - {c}")
        print("\nRe-run with --search-root pointed closer to the right one, "
              "or cd there and run this script from inside it.", file=sys.stderr)
        return 3

    repo = candidates[0]
    try:
        set_git_config(repo, ff=not args.no_ff)
    except subprocess.CalledProcessError as e:
        print(f"Error configuring repo at {repo}:\n  {e}", file=sys.stderr)
        return 4

    print(f"\nDone. Repo configured: {repo}")
    return 0

if __name__ == "__main__":
    sys.exit(main())