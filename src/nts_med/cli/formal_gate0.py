"""Fail-closed formal Gate 0 preflight and one-run authorization guard.

This command does not calculate relations by itself.  It is the single guard
that must pass before a separately authorized formal executor may be invoked.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from nts_med.reproducibility.freeze import (
    manifest_arguments,
    protected_paths,
    sha256_file,
    validate_authorization,
    validate_freeze_passport,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the formal preflight parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--passport", type=Path, required=True)
    parser.add_argument(
        "--private-manifest",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="private manifest path; may be repeated and is never emitted",
    )
    parser.add_argument("--authorization", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("preflight", "authorize-run"),
        default="preflight",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _git_checks(root: Path, passport: Path) -> tuple[dict[str, bool], list[str]]:
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    try:
        import yaml

        payload = yaml.safe_load(passport.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("passport must be a mapping")
        commit = payload.get("implementation_commit")
        if not isinstance(commit, str) or len(commit) != 40:
            raise ValueError("implementation_commit is invalid")
        commit_check = _git(root, "cat-file", "-e", f"{commit}^{{commit}}")
        checks["implementation_commit_exists"] = commit_check.returncode == 0
        ancestry = _git(root, "merge-base", "--is-ancestor", commit, "HEAD")
        checks["implementation_commit_is_ancestor"] = ancestry.returncode == 0
        bound_paths = protected_paths(passport)
        dirty = _git(root, "status", "--porcelain", "--", *bound_paths)
        checks["protected_worktree_clean"] = dirty.returncode == 0 and not dirty.stdout.strip()
    except (OSError, TypeError, ValueError):
        reasons.append("GIT_FREEZE_CHECK_FAILED")
    for name, passed in checks.items():
        if not passed:
            reasons.append(f"CHECK_FAILED:{name}")
    return checks, reasons


def _interpreter_check(root: Path) -> bool:
    expected = (root / ".conda" / "nts-med" / "python.exe").resolve(strict=False)
    return Path(sys.executable).resolve(strict=False) == expected


def _safe_write(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError("preflight output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the frozen state without opening scientific dataset content."""

    args = build_parser().parse_args(argv)
    root = args.repository_root.resolve()
    try:
        private_manifests = manifest_arguments(args.private_manifest)
        validation = validate_freeze_passport(
            root,
            args.passport,
            private_manifests=private_manifests,
        )
        git_checks, git_reasons = _git_checks(root, args.passport)
        checks = dict(validation.checks)
        checks.update(git_checks)
        checks["isolated_interpreter"] = _interpreter_check(root)
        reasons = [*validation.reason_codes, *git_reasons]
        if not checks["isolated_interpreter"]:
            reasons.append("CHECK_FAILED:isolated_interpreter")
        authorization_valid = False
        if args.mode == "authorize-run":
            if args.authorization is None or validation.freeze_id is None:
                reasons.append("FORMAL_AUTHORIZATION_REQUIRED")
            else:
                authorization_valid, authorization_reasons = validate_authorization(
                    args.authorization,
                    passport_sha256=sha256_file(args.passport),
                    freeze_id=validation.freeze_id,
                )
                reasons.extend(authorization_reasons)
            checks["one_run_authorization"] = authorization_valid
        passed = not reasons and all(checks.values())
        payload: dict[str, object] = {
            "schema_version": 1,
            "status": (
                "FORMAL_GATE0_PREFLIGHT_PASS"
                if passed and args.mode == "preflight"
                else "FORMAL_GATE0_AUTHORIZATION_PASS"
                if passed
                else "BLOCKED_PRE_GATE0"
            ),
            "mode": args.mode,
            "freeze_id": validation.freeze_id,
            "checks": dict(sorted(checks.items())),
            "reason_codes": sorted(set(reasons)),
            "gate0": "NOT_RUN",
            "training": "FORBIDDEN",
            "scientific_data_opened": False,
            "private_paths_emitted": False,
        }
        if args.output is not None:
            _safe_write(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 0 if passed else 2
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        payload = {
            "status": "BLOCKED_PRE_GATE0",
            "reason_codes": [f"PREFLIGHT_ERROR:{type(error).__name__}"],
            "gate0": "NOT_RUN",
            "training": "FORBIDDEN",
        }
        print(json.dumps(payload, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
