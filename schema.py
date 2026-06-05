from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


class BottleAnalysis(BaseModel):
    object_id: int
    frame_number: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Core classification outputs
    material: Literal["PET", "HDPE", "PP", "PVC", "unknown"]
    color: str                        # "clear", "green", "blue", "amber", etc.
    pet_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    # Derived flags
    review_flag: bool = False         # True when confidence < threshold
    batch_grade: Literal["A", "B", "C", "reject"]

    # Explainability
    reasoning: str                    # one-sentence VLM explanation

    # Bounding box (from YOLO)
    bbox: Optional[list[int]] = None  # [x1, y1, x2, y2]


class BatchSummary(BaseModel):
    total_count: int
    pet_count: int
    non_pet_count: int
    unknown_count: int
    avg_pet_probability: float
    avg_confidence: float
    flagged_for_review: int
    dominant_color: str
    batch_grade: Literal["A", "B", "C", "reject"]
    estimated_value_usd: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
