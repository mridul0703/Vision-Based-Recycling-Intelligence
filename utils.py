from typing import List
from collections import Counter

from schema import BottleAnalysis, BatchSummary

PET_USD_PER_KG = 0.35
AVG_BOTTLE_WEIGHT_KG = 0.025


def compute_batch_summary(results: List[BottleAnalysis]) -> BatchSummary:
    if not results:
        return BatchSummary(
            total_count=0, pet_count=0, non_pet_count=0, unknown_count=0,
            avg_pet_probability=0.0, avg_confidence=0.0, flagged_for_review=0,
            dominant_color="N/A", batch_grade="C", estimated_value_usd=0.0,
        )

    materials = [r.material for r in results]
    colors = [r.color for r in results]

    pet_count = materials.count("PET")
    unknown_count = materials.count("unknown")
    non_pet_count = len(results) - pet_count - unknown_count

    avg_pet_prob = round(sum(r.pet_probability for r in results) / len(results), 3)
    avg_conf = round(sum(r.confidence for r in results) / len(results), 3)
    flagged = sum(1 for r in results if r.review_flag)
    dominant_color = Counter(colors).most_common(1)[0][0]

    pet_ratio = pet_count / len(results)
    if pet_ratio >= 0.85:
        grade = "A"
    elif pet_ratio >= 0.65:
        grade = "B"
    elif pet_ratio >= 0.40:
        grade = "C"
    else:
        grade = "reject"

    est_value = round(pet_count * AVG_BOTTLE_WEIGHT_KG * PET_USD_PER_KG, 2)

    return BatchSummary(
        total_count=len(results),
        pet_count=pet_count,
        non_pet_count=non_pet_count,
        unknown_count=unknown_count,
        avg_pet_probability=avg_pet_prob,
        avg_confidence=avg_conf,
        flagged_for_review=flagged,
        dominant_color=dominant_color,
        batch_grade=grade,
        estimated_value_usd=est_value,
    )


def grade_color(grade: str) -> str:
    return {"A": "green", "B": "blue", "C": "orange", "reject": "red"}.get(grade, "gray")
