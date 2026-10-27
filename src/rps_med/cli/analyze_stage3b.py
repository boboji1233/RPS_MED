"""Create the privacy-safe aggregate Stage 3B analysis JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rps_med.stage3b_reporting import analyze_stage3b


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Stage 3B aggregate analysis already exists")
    result = analyze_stage3b(args.input)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "output": output.name}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
