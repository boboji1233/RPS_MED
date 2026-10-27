"""Generate tracked Stage 3A tables from the frozen node-level analysis."""

from __future__ import annotations

import argparse
import json

from rps_med.stage3a_reporting import write_stage3a_csv_reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = write_stage3a_csv_reports(args.analysis, args.output_dir)
    print(json.dumps({"status": "STAGE3A_REPORTS_WRITTEN", "paths": [str(p) for p in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
