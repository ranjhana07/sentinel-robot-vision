from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sentinel.detection.geometry import iou_matrix
from sentinel.detection.types import Detection


@dataclass(frozen=True)
class MatchResult:
    matched: int
    reference_count: int
    candidate_count: int
    ious: tuple[float, ...]
    score_deltas: tuple[float, ...]


def match_detections(
    reference: Sequence[Detection],
    candidate: Sequence[Detection],
    iou_threshold: float = 0.5,
) -> MatchResult:
    if not reference or not candidate:
        return MatchResult(0, len(reference), len(candidate), (), ())
    overlaps = iou_matrix(
        np.array([d.box for d in reference], dtype=np.float32),
        np.array([d.box for d in candidate], dtype=np.float32),
    )
    used_reference: set[int] = set()
    used_candidate: set[int] = set()
    ious: list[float] = []
    deltas: list[float] = []
    for flat_index in np.argsort(-overlaps, axis=None):
        ref_index, cand_index = divmod(int(flat_index), overlaps.shape[1])
        overlap = float(overlaps[ref_index, cand_index])
        if overlap < iou_threshold:
            break
        if ref_index in used_reference or cand_index in used_candidate:
            continue
        used_reference.add(ref_index)
        used_candidate.add(cand_index)
        ious.append(overlap)
        deltas.append(abs(reference[ref_index].score - candidate[cand_index].score))
    return MatchResult(len(ious), len(reference), len(candidate), tuple(ious), tuple(deltas))
