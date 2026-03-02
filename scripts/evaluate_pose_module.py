from __future__ import annotations

import csv
import os
from typing import Dict, List

from coach.vision import PoseFormChecker


def safe_int(x: str) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return 0


def main() -> None:
    manifest_path = os.path.join("data", "eval_manifest.csv")
    if not os.path.exists(manifest_path):
        raise SystemExit("Missing data/eval_manifest.csv")

    checker = PoseFormChecker()

    rows: List[Dict[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if not rows:
        print("No rows in eval manifest.")
        return

    total = 0
    abs_errors: List[int] = []
    within_1 = 0
    within_2 = 0
    skipped = 0

    for r in rows:
        path = str(r.get("video_path", "")).strip()
        label = safe_int(str(r.get("label_rep_count", "0")))

        if not path or not os.path.exists(path):
            print(f"Skip missing: {path}")
            skipped += 1
            continue

        print(f"Analyze: {path}")
        fb = checker.analyze_squat_video(
            input_path=path,
            export_annotated=False,
            out_dir="outputs",
        )

        if fb.verdict == "error":
            print("  error:", "; ".join(fb.details))
            skipped += 1
            continue

        pred = int(fb.stats.get("rep_count_est", 0))
        err = abs(pred - label)

        abs_errors.append(err)
        total += 1

        if err <= 1:
            within_1 += 1
        if err <= 2:
            within_2 += 1

        print(f"  labeled reps: {label}, predicted reps: {pred}, abs error: {err}")
        print(f"  rom depth normalized: {fb.stats.get('depth_range_norm', 0.0):.3f}")

    if total == 0:
        print("No valid videos processed.")
        return

    mae = sum(abs_errors) / max(1, len(abs_errors))
    acc1 = within_1 / total
    acc2 = within_2 / total

    print("\nEvaluation summary")
    print(f"  videos processed: {total}")
    print(f"  videos skipped: {skipped}")
    print(f"  rep count MAE: {mae:.3f}")
    print(f"  within 1 rep: {acc1:.3f}")
    print(f"  within 2 reps: {acc2:.3f}")


if __name__ == "__main__":
    main()
