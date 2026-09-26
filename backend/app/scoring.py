"""Turn raw detections into an experimental 0-100 visible-litter severity score.

The score describes litter visible in one photo. It says nothing about chemical
pollution, bacteria or whether the water is safe.

The score blends two signals:
  * how many pieces of litter were found (confidence-weighted), which
    saturates so a handful of items already counts as significant, and
  * how much of the frame the litter covers.

  score = 60 * (1 - e^(-weighted_count / 5))  +  40 * min(1, coverage / 0.10)

Severity bands: 0 none, 1-24 low, 25-49 moderate, 50-74 high, 75-100 severe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .detector import Detection

COUNT_WEIGHT = 60
COVERAGE_WEIGHT = 40
COUNT_SCALE = 5.0        # weighted items at which the count term reaches ~63%
COVERAGE_FULL = 0.10     # 10% of the frame covered = full coverage points

SCORE_NOTE = ("Experimental visible-litter severity score (0-100). It describes litter visible in this photo only, "
              "not chemical pollution or water safety.")

SEVERITY_BANDS = [(75, "severe"), (50, "high"), (25, "moderate"), (1, "low"), (0, "none")]


@dataclass
class Score:
    score: int
    severity: str
    item_count: int
    weighted_count: float
    coverage: float  # fraction of the image covered by litter boxes, 0-1

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "severity": self.severity,
            "item_count": self.item_count,
            "weighted_count": round(self.weighted_count, 3),
            "coverage_pct": round(self.coverage * 100, 2),
        }


def _coverage(detections: list[Detection], width: int, height: int) -> float:
    """Fraction of the image covered by the union of boxes (overlaps counted once).

    Uses a coarse occupancy grid, which is plenty accurate for scoring.
    """
    if not detections or width <= 0 or height <= 0:
        return 0.0
    grid = 200
    cells = [[False] * grid for _ in range(grid)]
    for d in detections:
        x0 = max(0, int(d.x / width * grid))
        y0 = max(0, int(d.y / height * grid))
        x1 = min(grid, math.ceil((d.x + d.w) / width * grid))
        y1 = min(grid, math.ceil((d.y + d.h) / height * grid))
        for row in cells[y0:y1]:
            row[x0:x1] = [True] * (x1 - x0)
    return sum(map(sum, cells)) / (grid * grid)


def severity_for(score: int) -> str:
    for floor, name in SEVERITY_BANDS:
        if score >= floor:
            return name
    return "none"


def score_detections(detections: list[Detection], width: int, height: int) -> Score:
    weighted = sum(d.confidence for d in detections)
    coverage = _coverage(detections, width, height)
    raw = COUNT_WEIGHT * (1 - math.exp(-weighted / COUNT_SCALE)) + COVERAGE_WEIGHT * min(1.0, coverage / COVERAGE_FULL)
    score = int(round(raw))
    if detections and score == 0:
        score = 1  # anything detected is at least "low"
    return Score(score=score, severity=severity_for(score), item_count=len(detections),
                 weighted_count=weighted, coverage=coverage)
