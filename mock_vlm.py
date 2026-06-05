import random
from schema import BottleAnalysis

MATERIALS = ["PET", "HDPE", "PP", "PVC", "unknown"]
MATERIAL_WEIGHTS = [0.55, 0.20, 0.10, 0.05, 0.10]

COLORS = ["clear", "green", "blue", "amber", "white", "red", "mixed"]
COLOR_WEIGHTS = [0.40, 0.20, 0.15, 0.10, 0.08, 0.04, 0.03]

GRADE_MAP = {
    "PET": "A",
    "HDPE": "B",
    "PP": "B",
    "PVC": "reject",
    "unknown": "C",
}

REASONINGS = {
    "PET": [
        "Clear polyethylene terephthalate bottle with recycling symbol #1 visible.",
        "Typical soft-drink PET bottle with characteristic neck shape.",
        "Transparent PET, high confidence from shape and sheen.",
    ],
    "HDPE": [
        "Opaque high-density polyethylene, likely a detergent or milk container.",
        "HDPE bottle with matte finish, recycling symbol #2 estimated.",
    ],
    "PP": [
        "Polypropylene container, slightly translucent with rigid walls.",
        "PP bottle common in yogurt or condiment packaging.",
    ],
    "PVC": [
        "Possible PVC — softer feel and greenish tint are warning signs.",
        "Suspected PVC contamination; flagging for human review.",
    ],
    "unknown": [
        "Crushed or occluded bottle; insufficient visual data for classification.",
        "Unable to determine material — flagging for manual inspection.",
    ],
}


def mock_classify(obj_id: int, frame_number: int = 0) -> BottleAnalysis:
    """
    Simulates a VLM response instantly with realistic distributions.
    Produces occasional low-confidence items to exercise the review queue.
    """
    material = random.choices(MATERIALS, weights=MATERIAL_WEIGHTS)[0]
    color = random.choices(COLORS, weights=COLOR_WEIGHTS)[0]

    if material == "unknown" or random.random() < 0.12:
        confidence = round(random.uniform(0.30, 0.64), 2)
    else:
        confidence = round(random.uniform(0.70, 0.98), 2)

    pet_prob = round(random.uniform(0.75, 0.97), 2) if material == "PET" \
        else round(random.uniform(0.02, 0.30), 2)

    threshold = 0.65
    review_flag = confidence < threshold

    return BottleAnalysis(
        object_id=obj_id,
        frame_number=frame_number,
        material=material,
        color=color,
        pet_probability=pet_prob,
        confidence=confidence,
        review_flag=review_flag,
        batch_grade=GRADE_MAP[material],
        reasoning=random.choice(REASONINGS[material]),
    )
