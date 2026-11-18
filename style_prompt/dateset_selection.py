#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select style descriptors from fragment_level_records.jsonl.

For each mechanism_category, select 150 unique descriptors:
  - 45 high-frequency representatives
  - 45 high-style-strength descriptors
  - 20 text-embedding cluster representatives
  - 15 low-style-strength descriptors
  - 25 low-frequency long-tail descriptors

For each non-NSFW effect_profile dimension, select 110 unique descriptors:
  - 60 single-dimension high score
  - 30 mixed high score
  - 10 single-dimension low score
  - 10 mixed low score

Output JSONL keeps every original field except sample_id, and adds:
  - count: number of IDs in the original sample_id field
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import euclidean_distances


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_JSONL = SCRIPT_DIR / "dataset_experiments" / "fragment_level_records.jsonl"
DEFAULT_OUTPUT_JSONL = SCRIPT_DIR / "experiment_outputs" / "selected_style_descriptors_by_mechanism.jsonl"
DEFAULT_SUMMARY_JSON = SCRIPT_DIR / "experiment_outputs" / "selected_style_descriptors_by_mechanism_summary.json"
DEFAULT_EFFECT_OUTPUT_JSONL = SCRIPT_DIR / "experiment_outputs" / "selected_style_descriptors_by_effect_profile.jsonl"
DEFAULT_EFFECT_SUMMARY_JSON = SCRIPT_DIR / "experiment_outputs" / "selected_style_descriptors_by_effect_profile_summary.json"
DEFAULT_EFFECT_STRICT_OUTPUT_JSONL = SCRIPT_DIR / "experiment_outputs" / "selected_style_descriptors_by_effect_profile_strict.jsonl"
DEFAULT_EFFECT_STRICT_SUMMARY_JSON = SCRIPT_DIR / "experiment_outputs" / "selected_style_descriptors_by_effect_profile_strict_summary.json"

DEFAULT_QUOTAS: "OrderedDict[str, int]" = OrderedDict(
    [
        ("high_frequency", 45),
        ("high_style_strength", 45),
        ("cluster_representative", 20),
        ("low_style_strength", 15),
        ("low_frequency_tail", 25),
    ]
)

DEFAULT_EFFECT_QUOTAS: "OrderedDict[str, int]" = OrderedDict(
    [
        ("single_high", 60),
        ("mixed_high", 30),
        ("single_low", 10),
        ("mixed_low", 10),
    ]
)

NSFW_EFFECT_DIMENSIONS = {"depiction_explicitness", "violence_graphicness"}


@dataclass(frozen=True)
class Candidate:
    record: Dict[str, Any]
    line_no: int
    key: str
    descriptor: str
    category: str
    count: int
    strength: float


@dataclass(frozen=True)
class Selection:
    candidate: Candidate
    bucket: str
    bucket_rank: int
    cluster_id: Optional[int] = None
    cluster_size: Optional[int] = None
    cluster_distance: Optional[float] = None


@dataclass(frozen=True)
class EffectSelection:
    candidate: Candidate
    dimension: str
    bucket: str
    bucket_rank: int
    selection_mode: str
    target_score: float
    other_max_score: float
    high_dimension_count: int
    low_dimension_count: int
    strict_match: bool


@dataclass(frozen=True)
class EffectScoredCandidate:
    candidate: Candidate
    target_score: float
    other_max_score: float
    high_dimension_count: int
    low_dimension_count: int


class TextEmbedder:
    def __init__(
        self,
        backend: str,
        sentence_transformer_model: str,
        batch_size: int,
        local_files_only: bool,
        device: Optional[str],
    ) -> None:
        self.backend = backend
        self.sentence_transformer_model = sentence_transformer_model
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self.device = device
        self._sentence_model = None

    def encode(self, texts: Sequence[str]) -> Tuple[Any, str]:
        if self.backend in {"auto", "sentence-transformer"}:
            try:
                embeddings = self._encode_sentence_transformer(texts)
            except Exception as exc:
                if self.backend == "sentence-transformer":
                    raise
                print(f"[WARN] sentence-transformer unavailable, fallback to TF-IDF: {exc}", flush=True)
            else:
                return embeddings, f"sentence-transformer:{self.sentence_transformer_model}"

        return self._encode_tfidf(texts), "tfidf:char_wb_3_5"

    def _encode_sentence_transformer(self, texts: Sequence[str]) -> np.ndarray:
        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            kwargs: Dict[str, Any] = {"local_files_only": self.local_files_only}
            if self.device:
                kwargs["device"] = self.device
            self._sentence_model = SentenceTransformer(self.sentence_transformer_model, **kwargs)

        embeddings = self._sentence_model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @staticmethod
    def _encode_tfidf(texts: Sequence[str]) -> Any:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            min_df=1,
            max_features=50000,
            norm="l2",
        )
        return vectorizer.fit_transform(texts)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def make_descriptor_key(category: str, descriptor: str, fallback: Any) -> str:
    normalized = normalize_text(descriptor).casefold()
    if normalized:
        return f"{category}\u241f{normalized}"
    return f"{category}\u241f{normalize_text(fallback)}"


def flatten_sample_ids(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        result: List[Any] = []
        for item in value:
            result.extend(flatten_sample_ids(item))
        return result
    return [value]


def sample_id_count(value: Any) -> int:
    return len(flatten_sample_ids(value))


def descriptor_word_count(text: str) -> int:
    return len(re.findall(r"\w+", normalize_text(text)))


def descriptor_char_count(text: str) -> int:
    return len(normalize_text(text))


def is_long_phrase(candidate: Candidate) -> bool:
    return normalize_text(candidate.record.get("granularity")).casefold() == "long_phrase"


def effect_strength(effect_profile: Any) -> float:
    if not isinstance(effect_profile, dict):
        return 0.0

    scores: List[float] = []
    for value in effect_profile.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            scores.append(float(value))
    return max(scores) if scores else 0.0


def parse_dimension_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def infer_effect_dimensions(candidates: Sequence[Candidate], excluded_dimensions: set[str]) -> List[str]:
    for candidate in candidates:
        effect_profile = candidate.record.get("effect_profile")
        if not isinstance(effect_profile, dict):
            continue
        return [dimension for dimension in effect_profile.keys() if dimension not in excluded_dimensions]
    return []


def numeric_effect_score(effect_profile: Any, dimension: str) -> float:
    if not isinstance(effect_profile, dict):
        return 0.0
    value = effect_profile.get(dimension, 0.0)
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return 0.0


def scored_effect_candidates(
    candidates: Sequence[Candidate],
    dimension: str,
    effect_dimensions: Sequence[str],
    high_threshold: float,
    low_threshold: float,
) -> List[EffectScoredCandidate]:
    scored: List[EffectScoredCandidate] = []
    for candidate in candidates:
        effect_profile = candidate.record.get("effect_profile")
        scores = {
            current_dimension: numeric_effect_score(effect_profile, current_dimension)
            for current_dimension in effect_dimensions
        }
        target_score = scores.get(dimension, 0.0)
        other_scores = [
            score
            for current_dimension, score in scores.items()
            if current_dimension != dimension
        ]
        scored.append(
            EffectScoredCandidate(
                candidate=candidate,
                target_score=target_score,
                other_max_score=max(other_scores) if other_scores else 0.0,
                high_dimension_count=sum(1 for score in scores.values() if score >= high_threshold),
                low_dimension_count=sum(1 for score in scores.values() if score <= low_threshold),
            )
        )
    return scored


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def load_candidates(path: Path, max_records: Optional[int] = None) -> Tuple[List[Candidate], Dict[str, int]]:
    candidates: List[Candidate] = []
    skipped: Dict[str, int] = defaultdict(int)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                skipped["empty_line"] += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc

            descriptor = normalize_text(record.get("raw_descriptor"))
            category = normalize_text(record.get("mechanism_category"))
            if not descriptor:
                skipped["missing_raw_descriptor"] += 1
                continue
            if not category:
                skipped["missing_mechanism_category"] += 1
                continue

            candidates.append(
                Candidate(
                    record=record,
                    line_no=line_no,
                    key=make_descriptor_key(category, descriptor, record.get("fragment_id", line_no)),
                    descriptor=descriptor,
                    category=category,
                    count=sample_id_count(record.get("sample_id")),
                    strength=effect_strength(record.get("effect_profile")),
                )
            )

            if max_records is not None and len(candidates) >= max_records:
                break

    return candidates, dict(skipped)


def group_by_category(candidates: Sequence[Candidate]) -> "OrderedDict[str, List[Candidate]]":
    groups: "OrderedDict[str, List[Candidate]]" = OrderedDict()
    for item in candidates:
        groups.setdefault(item.category, []).append(item)
    return groups


def take_ranked(
    candidates: Sequence[Candidate],
    quota: int,
    selected_keys: set[str],
    bucket: str,
    sort_key: Callable[[Candidate], Tuple[Any, ...]],
) -> List[Selection]:
    selected: List[Selection] = []
    for item in sorted(candidates, key=sort_key):
        if item.key in selected_keys:
            continue
        selected_keys.add(item.key)
        selected.append(Selection(candidate=item, bucket=bucket, bucket_rank=len(selected) + 1))
        if len(selected) >= quota:
            break
    return selected


def row_2d(matrix: Any, index: int) -> Any:
    row = matrix[index]
    if hasattr(row, "reshape"):
        return row.reshape(1, -1)
    return np.asarray(row).reshape(1, -1)


def dense_embeddings(embeddings: Any) -> np.ndarray:
    if hasattr(embeddings, "toarray"):
        return embeddings.toarray()
    return np.asarray(embeddings)


def take_cluster_representatives(
    candidates: Sequence[Candidate],
    quota: int,
    selected_keys: set[str],
    embedder: TextEmbedder,
    random_state: int,
) -> Tuple[List[Selection], str]:
    if quota <= 0 or not candidates:
        return [], "none"

    n_clusters = min(quota, len(candidates))
    texts = [item.descriptor for item in candidates]
    embeddings, embedding_backend = embedder.encode(texts)

    if n_clusters == len(candidates):
        labels = np.arange(len(candidates), dtype=np.int64)
        centers = dense_embeddings(embeddings)
    else:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            batch_size=1024,
        )
        labels = kmeans.fit_predict(embeddings)
        centers = np.asarray(kmeans.cluster_centers_)

    cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        cluster_to_indices[int(label)].append(index)

    cluster_order = sorted(
        cluster_to_indices,
        key=lambda cluster_id: (-len(cluster_to_indices[cluster_id]), cluster_id),
    )

    selected: List[Selection] = []
    for cluster_id in cluster_order:
        indices = cluster_to_indices[cluster_id]
        distances = euclidean_distances(embeddings[indices], centers[cluster_id].reshape(1, -1)).reshape(-1)
        member_order = sorted(
            range(len(indices)),
            key=lambda local_idx: (
                float(distances[local_idx]),
                -candidates[indices[local_idx]].count,
                -candidates[indices[local_idx]].strength,
                candidates[indices[local_idx]].descriptor.casefold(),
                candidates[indices[local_idx]].line_no,
            ),
        )

        for local_idx in member_order:
            index = indices[local_idx]
            candidate = candidates[index]
            if candidate.key in selected_keys:
                continue
            selected_keys.add(candidate.key)
            selected.append(
                Selection(
                    candidate=candidate,
                    bucket="cluster_representative",
                    bucket_rank=len(selected) + 1,
                    cluster_id=cluster_id,
                    cluster_size=len(indices),
                    cluster_distance=float(distances[local_idx]),
                )
            )
            break

        if len(selected) >= quota:
            break

    if len(selected) < quota:
        fallback_items: List[Tuple[float, int, Candidate]] = []
        for index, candidate in enumerate(candidates):
            if candidate.key in selected_keys:
                continue
            cluster_id = int(labels[index])
            distance = float(euclidean_distances(row_2d(embeddings, index), centers[cluster_id].reshape(1, -1))[0][0])
            fallback_items.append((distance, cluster_id, candidate))

        fallback_items.sort(
            key=lambda item: (
                item[0],
                -item[2].count,
                -item[2].strength,
                item[2].descriptor.casefold(),
                item[2].line_no,
            )
        )
        for distance, cluster_id, candidate in fallback_items:
            selected_keys.add(candidate.key)
            selected.append(
                Selection(
                    candidate=candidate,
                    bucket="cluster_representative",
                    bucket_rank=len(selected) + 1,
                    cluster_id=cluster_id,
                    cluster_distance=distance,
                )
            )
            if len(selected) >= quota:
                break

    return selected, embedding_backend


def fill_to_target(
    candidates: Sequence[Candidate],
    target_total: int,
    selected_keys: set[str],
    selections: List[Selection],
) -> None:
    if len(selections) >= target_total:
        return

    selections.extend(
        take_ranked(
            candidates=candidates,
            quota=target_total - len(selections),
            selected_keys=selected_keys,
            bucket="fallback_fill",
            sort_key=lambda item: (-item.count, -item.strength, item.descriptor.casefold(), item.line_no),
        )
    )


def select_for_category(
    category: str,
    candidates: Sequence[Candidate],
    quotas: MutableMapping[str, int],
    embedder: TextEmbedder,
    random_state: int,
) -> Tuple[List[Selection], Dict[str, Any]]:
    selected_keys: set[str] = set()
    selections: List[Selection] = []

    selections.extend(
        take_ranked(
            candidates,
            quotas["high_frequency"],
            selected_keys,
            "high_frequency",
            lambda item: (-item.count, -item.strength, item.descriptor.casefold(), item.line_no),
        )
    )
    selections.extend(
        take_ranked(
            candidates,
            quotas["high_style_strength"],
            selected_keys,
            "high_style_strength",
            lambda item: (-item.strength, -item.count, item.descriptor.casefold(), item.line_no),
        )
    )

    cluster_selections, embedding_backend = take_cluster_representatives(
        candidates=candidates,
        quota=quotas["cluster_representative"],
        selected_keys=selected_keys,
        embedder=embedder,
        random_state=random_state,
    )
    selections.extend(cluster_selections)

    selections.extend(
        take_ranked(
            candidates,
            quotas["low_style_strength"],
            selected_keys,
            "low_style_strength",
            lambda item: (item.strength, -item.count, item.descriptor.casefold(), item.line_no),
        )
    )
    selections.extend(
        take_ranked(
            [item for item in candidates if is_long_phrase(item)],
            quotas["low_frequency_tail"],
            selected_keys,
            "low_frequency_tail",
            lambda item: (
                item.count,
                -descriptor_word_count(item.descriptor),
                -descriptor_char_count(item.descriptor),
                -item.strength,
                item.descriptor.casefold(),
                item.line_no,
            ),
        )
    )

    target_total = sum(quotas.values())
    fill_to_target(candidates, target_total, selected_keys, selections)

    bucket_counts: Dict[str, int] = defaultdict(int)
    for item in selections:
        bucket_counts[item.bucket] += 1

    summary = {
        "mechanism_category": category,
        "available_records": len(candidates),
        "selected_records": len(selections),
        "target_records": target_total,
        "bucket_counts": dict(bucket_counts),
        "embedding_backend": embedding_backend,
        "min_count": min((item.candidate.count for item in selections), default=0),
        "max_count": max((item.candidate.count for item in selections), default=0),
        "min_strength": min((item.candidate.strength for item in selections), default=0.0),
        "max_strength": max((item.candidate.strength for item in selections), default=0.0),
    }
    return selections, summary


def is_strict_effect_bucket_match(
    item: EffectScoredCandidate,
    bucket: str,
    effect_dimension_count: int,
    high_threshold: float,
    low_threshold: float,
) -> bool:
    if bucket == "single_high":
        return (
            item.target_score >= high_threshold
            and item.high_dimension_count == 1
            and item.other_max_score <= low_threshold
        )
    if bucket == "mixed_high":
        return item.target_score >= high_threshold and item.high_dimension_count >= 2
    if bucket == "single_low":
        return item.target_score <= low_threshold and item.high_dimension_count >= 1
    if bucket == "mixed_low":
        return item.target_score <= low_threshold and item.low_dimension_count >= 2
    raise ValueError(f"Unknown effect selection bucket: {bucket}")


def is_effect_bucket_fallback_candidate(
    item: EffectScoredCandidate,
    bucket: str,
    high_threshold: float,
    low_threshold: float,
) -> bool:
    if bucket in {"single_high", "mixed_high"}:
        return item.target_score >= high_threshold
    if bucket in {"single_low", "mixed_low"}:
        return item.target_score <= low_threshold
    raise ValueError(f"Unknown effect selection bucket: {bucket}")


def effect_bucket_sort_key(bucket: str, item: EffectScoredCandidate) -> Tuple[Any, ...]:
    candidate = item.candidate
    descriptor = candidate.descriptor.casefold()
    if bucket == "single_high":
        return (
            -(item.target_score - item.other_max_score),
            -item.target_score,
            item.other_max_score,
            item.high_dimension_count,
            -candidate.count,
            descriptor,
            candidate.line_no,
        )
    if bucket == "mixed_high":
        return (
            -item.target_score,
            -item.high_dimension_count,
            -item.other_max_score,
            -candidate.count,
            descriptor,
            candidate.line_no,
        )
    if bucket == "single_low":
        return (
            item.target_score,
            -item.other_max_score,
            -item.high_dimension_count,
            -candidate.count,
            descriptor,
            candidate.line_no,
        )
    if bucket == "mixed_low":
        return (
            max(item.target_score, item.other_max_score),
            item.target_score,
            -item.low_dimension_count,
            -candidate.count,
            descriptor,
            candidate.line_no,
        )
    raise ValueError(f"Unknown effect selection bucket: {bucket}")


def take_effect_ranked(
    scored: Sequence[EffectScoredCandidate],
    dimension: str,
    bucket: str,
    quota: int,
    selected_keys: set[str],
    effect_dimension_count: int,
    high_threshold: float,
    low_threshold: float,
    selection_mode: str,
) -> Tuple[List[EffectSelection], Dict[str, Any]]:
    if selection_mode not in {"relaxed", "strict"}:
        raise ValueError("selection_mode must be 'relaxed' or 'strict'")

    selected: List[EffectSelection] = []
    eligible_strict = [
        item
        for item in scored
        if item.candidate.key not in selected_keys
        and is_strict_effect_bucket_match(
            item,
            bucket=bucket,
            effect_dimension_count=effect_dimension_count,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )
    ]

    def append_item(item: EffectScoredCandidate, strict_match: bool) -> None:
        selected_keys.add(item.candidate.key)
        selected.append(
            EffectSelection(
                candidate=item.candidate,
                dimension=dimension,
                bucket=bucket,
                bucket_rank=len(selected) + 1,
                selection_mode=selection_mode,
                target_score=item.target_score,
                other_max_score=item.other_max_score,
                high_dimension_count=item.high_dimension_count,
                low_dimension_count=item.low_dimension_count,
                strict_match=strict_match,
            )
        )

    def append_from_pool(pool: Sequence[EffectScoredCandidate], strict_match: bool) -> None:
        for item in sorted(pool, key=lambda current: effect_bucket_sort_key(bucket, current)):
            if item.candidate.key in selected_keys:
                continue
            append_item(item, strict_match=strict_match)
            if len(selected) >= quota:
                break

    if selection_mode == "strict":
        append_from_pool(eligible_strict, strict_match=True)
    else:
        append_from_pool(
            [item for item in scored if item.candidate.key not in selected_keys],
            strict_match=False,
        )

    details = {
        "quota": quota,
        "selection_mode": selection_mode,
        "selection_strategy": "strict_only" if selection_mode == "strict" else "ranked_all_candidates",
        "eligible_strict_before_dedup": len(eligible_strict),
        "selected": len(selected),
        "strict_selected": sum(1 for item in selected if item.strict_match),
        "fallback_selected": sum(1 for item in selected if not item.strict_match),
    }
    return selected, details


def fill_effect_to_target(
    scored: Sequence[EffectScoredCandidate],
    dimension: str,
    target_total: int,
    selected_keys: set[str],
    selections: List[EffectSelection],
    selection_mode: str,
) -> None:
    if len(selections) >= target_total:
        return

    remaining = sorted(
        [item for item in scored if item.candidate.key not in selected_keys],
        key=lambda item: (
            -item.candidate.count,
            -item.target_score,
            -item.high_dimension_count,
            item.candidate.descriptor.casefold(),
            item.candidate.line_no,
        ),
    )
    for item in remaining:
        selected_keys.add(item.candidate.key)
        selections.append(
            EffectSelection(
                candidate=item.candidate,
                dimension=dimension,
                bucket="fallback_fill",
                bucket_rank=len(selections) + 1,
                selection_mode=selection_mode,
                target_score=item.target_score,
                other_max_score=item.other_max_score,
                high_dimension_count=item.high_dimension_count,
                low_dimension_count=item.low_dimension_count,
                strict_match=selection_mode == "strict",
            )
        )
        if len(selections) >= target_total:
            break


def select_for_effect_dimension(
    dimension: str,
    candidates: Sequence[Candidate],
    effect_dimensions: Sequence[str],
    quotas: MutableMapping[str, int],
    high_threshold: float,
    low_threshold: float,
    selection_mode: str,
) -> Tuple[List[EffectSelection], Dict[str, Any]]:
    selected_keys: set[str] = set()
    selections: List[EffectSelection] = []
    scored = scored_effect_candidates(
        candidates=candidates,
        dimension=dimension,
        effect_dimensions=effect_dimensions,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
    )

    bucket_details: Dict[str, Dict[str, int]] = {}
    for bucket, quota in quotas.items():
        bucket_selections, details = take_effect_ranked(
            scored=scored,
            dimension=dimension,
            bucket=bucket,
            quota=quota,
            selected_keys=selected_keys,
            effect_dimension_count=len(effect_dimensions),
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            selection_mode=selection_mode,
        )
        selections.extend(bucket_selections)
        bucket_details[bucket] = details

    target_total = sum(quotas.values())
    if selection_mode == "relaxed":
        fill_effect_to_target(scored, dimension, target_total, selected_keys, selections, selection_mode)

    bucket_counts: Dict[str, int] = defaultdict(int)
    strict_counts: Dict[str, int] = defaultdict(int)
    fallback_counts: Dict[str, int] = defaultdict(int)
    for item in selections:
        bucket_counts[item.bucket] += 1
        if item.strict_match:
            strict_counts[item.bucket] += 1
        else:
            fallback_counts[item.bucket] += 1

    summary = {
        "effect_dimension": dimension,
        "available_records": len(candidates),
        "selected_records": len(selections),
        "target_records": target_total,
        "selection_mode": selection_mode,
        "bucket_counts": dict(bucket_counts),
        "strict_selected_counts": dict(strict_counts),
        "fallback_selected_counts": dict(fallback_counts),
        "bucket_details": bucket_details,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
    }
    return selections, summary


def select_by_effect_profile(
    candidates: Sequence[Candidate],
    effect_dimensions: Sequence[str],
    quotas: MutableMapping[str, int],
    high_threshold: float,
    low_threshold: float,
    selection_mode: str,
) -> Tuple[List[EffectSelection], List[Dict[str, Any]]]:
    all_selections: List[EffectSelection] = []
    summaries: List[Dict[str, Any]] = []
    for dimension in effect_dimensions:
        print(f"[INFO] Selecting effect_dimension={dimension}, mode={selection_mode}", flush=True)
        selections, summary = select_for_effect_dimension(
            dimension=dimension,
            candidates=candidates,
            effect_dimensions=effect_dimensions,
            quotas=quotas,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            selection_mode=selection_mode,
        )
        all_selections.extend(selections)
        summaries.append(summary)
        print(
            f"[INFO] {dimension} ({selection_mode}): selected {summary['selected_records']}/"
            f"{summary['target_records']}, buckets={summary['bucket_counts']}",
            flush=True,
        )
    return all_selections, summaries


def output_record(selection: Selection) -> Dict[str, Any]:
    candidate = selection.candidate
    out = {key: value for key, value in candidate.record.items() if key != "sample_id"}
    out["count"] = candidate.count
    return out


def output_effect_record(selection: EffectSelection) -> Dict[str, Any]:
    candidate = selection.candidate
    out = {key: value for key, value in candidate.record.items() if key != "sample_id"}
    out["count"] = candidate.count
    out["effect_dimension"] = selection.dimension
    out["effect_selection_bucket"] = selection.bucket
    out["effect_selection_rank"] = selection.bucket_rank
    out["effect_selection_mode"] = selection.selection_mode
    out["effect_target_score"] = selection.target_score
    out["effect_other_max_score"] = selection.other_max_score
    out["effect_high_dimension_count"] = selection.high_dimension_count
    out["effect_low_dimension_count"] = selection.low_dimension_count
    out["effect_strict_match"] = selection.strict_match
    return out


def save_jsonl_atomic(records: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json_dumps(record) + "\n")
            count += 1
    tmp_path.replace(path)
    return count


def save_json_atomic(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select descriptors by mechanism_category.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--effect-output-jsonl", type=Path, default=DEFAULT_EFFECT_OUTPUT_JSONL)
    parser.add_argument("--effect-summary-json", type=Path, default=DEFAULT_EFFECT_SUMMARY_JSON)
    parser.add_argument("--effect-strict-output-jsonl", type=Path, default=DEFAULT_EFFECT_STRICT_OUTPUT_JSONL)
    parser.add_argument("--effect-strict-summary-json", type=Path, default=DEFAULT_EFFECT_STRICT_SUMMARY_JSON)
    parser.add_argument(
        "--embedding-backend",
        choices=["sentence-transformer", "tfidf", "auto"],
        default="sentence-transformer",
        help="Embedding backend for cluster representatives.",
    )
    parser.add_argument("--sentence-transformer-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--sentence-transformer-batch-size", type=int, default=256)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow downloading a missing sentence-transformers model. Default uses local cache only.",
    )
    parser.add_argument("--device", default="cuda:0", help="Sentence-transformers device. Default: cuda:0")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None, help="Debug only: limit input records.")
    parser.add_argument("--skip-mechanism-selection", action="store_true")
    parser.add_argument("--skip-effect-selection", action="store_true")
    parser.add_argument(
        "--effect-selection-mode",
        choices=["both", "relaxed", "strict"],
        default="both",
        help="Run relaxed effect selection, strict effect selection, or both.",
    )
    parser.add_argument("--effect-high-threshold", type=float, default=0.7)
    parser.add_argument("--effect-low-threshold", type=float, default=0.3)
    parser.add_argument(
        "--effect-dimensions",
        default="",
        help="Comma-separated effect_profile dimensions. Default infers all except excluded dimensions.",
    )
    parser.add_argument(
        "--effect-exclude-dimensions",
        default=",".join(sorted(NSFW_EFFECT_DIMENSIONS)),
        help="Comma-separated effect_profile dimensions excluded from effect selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quotas = DEFAULT_QUOTAS.copy()
    effect_quotas = DEFAULT_EFFECT_QUOTAS.copy()

    candidates, skipped = load_candidates(args.input_jsonl, max_records=args.max_records)
    excluded_effect_dimensions = set(parse_dimension_csv(args.effect_exclude_dimensions))
    effect_dimensions = parse_dimension_csv(args.effect_dimensions)
    if not effect_dimensions:
        effect_dimensions = infer_effect_dimensions(candidates, excluded_effect_dimensions)

    if not args.skip_mechanism_selection:
        groups = group_by_category(candidates)
        embedder = TextEmbedder(
            backend=args.embedding_backend,
            sentence_transformer_model=args.sentence_transformer_model,
            batch_size=args.sentence_transformer_batch_size,
            local_files_only=not args.allow_model_download,
            device=args.device,
        )

        output_records: List[Dict[str, Any]] = []
        category_summaries: List[Dict[str, Any]] = []

        for category, category_candidates in groups.items():
            print(f"[INFO] Selecting category={category}, records={len(category_candidates)}", flush=True)
            selections, summary = select_for_category(
                category=category,
                candidates=category_candidates,
                quotas=quotas,
                embedder=embedder,
                random_state=args.random_state,
            )
            category_summaries.append(summary)
            output_records.extend(output_record(item) for item in selections)
            print(
                f"[INFO] {category}: selected {summary['selected_records']}/"
                f"{summary['target_records']}, buckets={summary['bucket_counts']}",
                flush=True,
            )

        written = save_jsonl_atomic(output_records, args.output_jsonl)
        summary_obj = {
            "input_jsonl": str(args.input_jsonl),
            "output_jsonl": str(args.output_jsonl),
            "total_input_records": len(candidates),
            "total_output_records": written,
            "quotas_per_category": dict(quotas),
            "skipped_records": skipped,
            "category_count": len(groups),
            "categories": category_summaries,
            "output_fields_rule": "all original fields except sample_id, plus count",
            "device": args.device,
        }
        save_json_atomic(summary_obj, args.summary_json)

        print(f"[INFO] Wrote selected records: {args.output_jsonl}", flush=True)
        print(f"[INFO] Wrote summary: {args.summary_json}", flush=True)
        print(f"[INFO] Total selected records: {written}", flush=True)

    if not args.skip_effect_selection:
        effect_runs: List[Tuple[str, Path, Path]] = []
        if args.effect_selection_mode in {"both", "relaxed"}:
            effect_runs.append(("relaxed", args.effect_output_jsonl, args.effect_summary_json))
        if args.effect_selection_mode in {"both", "strict"}:
            effect_runs.append(("strict", args.effect_strict_output_jsonl, args.effect_strict_summary_json))

        for selection_mode, output_jsonl, summary_json in effect_runs:
            effect_selections, effect_summaries = select_by_effect_profile(
                candidates=candidates,
                effect_dimensions=effect_dimensions,
                quotas=effect_quotas,
                high_threshold=args.effect_high_threshold,
                low_threshold=args.effect_low_threshold,
                selection_mode=selection_mode,
            )
            effect_written = save_jsonl_atomic(
                (output_effect_record(item) for item in effect_selections),
                output_jsonl,
            )
            effect_summary_obj = {
                "input_jsonl": str(args.input_jsonl),
                "output_jsonl": str(output_jsonl),
                "selection_mode": selection_mode,
                "effect_strict_match_value": selection_mode == "strict",
                "total_input_records": len(candidates),
                "total_output_records": effect_written,
                "effect_dimensions": effect_dimensions,
                "excluded_effect_dimensions": sorted(excluded_effect_dimensions),
                "quotas_per_dimension": dict(effect_quotas),
                "high_threshold": args.effect_high_threshold,
                "low_threshold": args.effect_low_threshold,
                "skipped_records": skipped,
                "dimensions": effect_summaries,
                "output_fields_rule": (
                    "all original fields except sample_id, plus count and effect selection metadata"
                ),
            }
            save_json_atomic(effect_summary_obj, summary_json)

            print(f"[INFO] Wrote effect selected records ({selection_mode}): {output_jsonl}", flush=True)
            print(f"[INFO] Wrote effect summary ({selection_mode}): {summary_json}", flush=True)
            print(f"[INFO] Total effect selected records ({selection_mode}): {effect_written}", flush=True)


if __name__ == "__main__":
    main()
