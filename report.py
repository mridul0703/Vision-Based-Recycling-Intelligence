from pathlib import Path
from datetime import datetime
from typing import List

import pandas as pd

from schema import BottleAnalysis, BatchSummary
from utils import compute_batch_summary


def build_dataframe(results: List[BottleAnalysis]) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with every deliverable as a named column.
    Each row = one detected bottle.
    """
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        rows.append({
            # Deliverable 1 — Object / bottle count (row = one bottle)
            "object_id":        r.object_id,
            "frame_number":     r.frame_number,

            # Deliverable 2 — PET vs non-PET composition
            "material":         r.material,
            "is_PET":           r.material == "PET",
            "pet_probability":  r.pet_probability,

            # Deliverable 3 — Color / quality distribution
            "color":            r.color,

            # Deliverable 4 — Confidence score per prediction
            "confidence":       r.confidence,

            # Deliverable 5 — Human-review flag
            "review_flag":      r.review_flag,

            # Deliverable 6 — Batch grade / estimated value
            "batch_grade":      r.batch_grade,

            # Extra context
            "reasoning":        r.reasoning,
            "timestamp":        r.timestamp,
        })
    return pd.DataFrame(rows)


def build_summary_text(summary: BatchSummary, total_input_frames: int,
                       yolo_count: int = 0, density_estimate: int = 0,
                       count_method: str = "yolo") -> str:
    pet_pct = (summary.pet_count / max(summary.total_count, 1)) * 100

    count_line = f"  YOLO detected          : {yolo_count}"
    if count_method == "density":
        count_line += f"\n  Density estimate       : ~{density_estimate} (pile heuristic)"

    lines = [
        "=" * 52,
        "  RECYCLING BATCH REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 52,
        "",
        "OBJECT COUNT",
        f"  Total bottles classified: {summary.total_count}",
        count_line,
        f"  Input frames processed  : {total_input_frames}",
        "",
        "MATERIAL COMPOSITION (Deliverable 2)",
        f"  PET                    : {summary.pet_count} ({pet_pct:.1f}%)",
        f"  Non-PET recyclable     : {summary.non_pet_count}",
        f"  Unknown / unreadable   : {summary.unknown_count}",
        f"  Avg PET probability    : {summary.avg_pet_probability:.0%}",
        "",
        "COLOR / QUALITY DISTRIBUTION (Deliverable 3)",
        f"  Dominant color         : {summary.dominant_color}",
        "",
        "CONFIDENCE (Deliverable 4)",
        f"  Avg confidence         : {summary.avg_confidence:.0%}",
        "",
        "REVIEW FLAGS (Deliverable 5)",
        f"  Flagged for review     : {summary.flagged_for_review} bottles",
        "",
        "BATCH GRADE & VALUE (Deliverable 6)",
        f"  Batch grade            : {summary.batch_grade}",
        f"  Estimated PET value    : ${summary.estimated_value_usd:.2f} USD",
        "",
        "=" * 52,
    ]
    return "\n".join(lines)


def export_csv(results: List[BottleAnalysis], out_dir: str = "outputs") -> Path:
    Path(out_dir).mkdir(exist_ok=True)
    df = build_dataframe(results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"recycling_report_{ts}.csv"
    df.to_csv(path, index=False)
    return path
