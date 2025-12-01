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


@dataclass(frozen=True)
class AgreementSummary:
    images: int
    reference_faces: int
    candidate_faces: int
    matched_faces: int
    recall: float
    precision: float
    mean_iou: float
    mean_score_delta: float


def summarise_matches(results: Sequence[MatchResult]) -> AgreementSummary:
    reference_faces = sum(r.reference_count for r in results)
    candidate_faces = sum(r.candidate_count for r in results)
    matched = sum(r.matched for r in results)
    all_ious = [value for r in results for value in r.ious]
    all_deltas = [value for r in results for value in r.score_deltas]
    return AgreementSummary(
        images=len(results),
        reference_faces=reference_faces,
        candidate_faces=candidate_faces,
        matched_faces=matched,
        recall=matched / reference_faces if reference_faces else 1.0,
        precision=matched / candidate_faces if candidate_faces else 1.0,
        mean_iou=float(np.mean(all_ious)) if all_ious else 0.0,
        mean_score_delta=float(np.mean(all_deltas)) if all_deltas else 0.0,
    )


def compare_detectors(reference, candidate, images: Sequence[np.ndarray], iou_threshold: float = 0.5) -> AgreementSummary:
    results = [
        match_detections(reference.detect(image), candidate.detect(image), iou_threshold)
        for image in images
    ]
    return summarise_matches(results)
