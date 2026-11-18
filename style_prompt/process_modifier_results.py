from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from test_prompts_design import (
    SEGMENTATION_PROMPT_GROUPS,
    TIER_WEIGHTS,
    VISUAL_UNDERSTANDING_PROMPTS,
    build_image_classification_prompts,
    build_object_detection_prompts,
    iter_prompt_jobs,
)


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
RESULT_ROOT = PACKAGE_DIR / "experiment_outputs" / "modifier_results"
OUTPUT_ROOT = RESULT_ROOT / "processed"

IMAGE_SIZE_FALLBACK = 512 * 512
SEX_CLIP_THRESHOLD = 0.95
SEX_NUDENET_THRESHOLD = 0.25
CLASS_RANK_WEIGHTS = (1.0, 0.8, 0.6)
DETECTION_CLASS_ALIASES = {
    "sofa": "couch",
}
CHUNK_SIZE = 5000

TARGET_MODELS = {
    "nsfw": ["clip", "nudenet", "q16"],
    "image_classification": ["resnet50", "mambavision"],
    "object_detection": ["yolo11", "deim"],
    "segmentation": ["deeplabv3plus", "sam2"],
    "visual_understanding": ["qwen3-vl", "llava-1.5"],
}

TASKS = [
    "image_classification",
    "object_detection",
    "segmentation",
    "visual_understanding",
    "nsfw",
]

OUTPUT_FIELDS = [
    "task",
    "target_model",
    "base_prompt",
    "modifier",
    "modifier_category",
    "modifier_family",
    "score",
    "image_scores",
]


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    prompt_lookups = build_prompt_lookups()

    for task in TASKS:
        for target_model in TARGET_MODELS[task]:
            input_path = input_csv_path(task, target_model)
            output_path = OUTPUT_ROOT / f"sd15_{task}_{target_model}.csv"
            if not input_path.exists():
                print(f"[SKIP] missing input: {input_path}")
                continue
            count = process_file(input_path, output_path, task, target_model, prompt_lookups)
            print(f"[OK] {task}/{target_model}: {count} rows -> {output_path}")


def input_csv_path(task: str, target_model: str) -> Path:
    file_model = target_model.replace(".", "_")
    filename = f"sd15_{file_model}_{task}.csv"
    if task == "nsfw":
        return RESULT_ROOT / filename
    return RESULT_ROOT / task / filename


def build_prompt_lookups() -> Dict[str, Dict[str, Dict[str, Any]]]:
    nsfw = {}
    for label, prompt_id, prompt_type, prompt in iter_prompt_jobs():
        nsfw[prompt_id] = {
            "task": "nsfw",
            "nsfw_type": label,
            "prompt_type": prompt_type,
            "base_prompt_id": prompt_id,
            "base_prompt": prompt,
        }

    classification = {
        item["base_prompt_id"]: item
        for item in build_image_classification_prompts()
    }
    object_detection = {
        item["base_prompt_id"]: item
        for item in build_object_detection_prompts()
    }
    segmentation = {}
    for group_idx, group in enumerate(SEGMENTATION_PROMPT_GROUPS):
        profile = group["profile"]
        for prompt_idx, item in enumerate(group["prompts"]):
            prompt_id = f"seg_{profile}_{prompt_idx + 1:02d}"
            segmentation[prompt_id] = {
                "task": "segmentation",
                "base_prompt_id": prompt_id,
                "profile": profile,
                "description": group["description"],
                "difficulty": item["difficulty"],
                "stress_factor": item["stress_factor"],
                "base_prompt": item["prompt"],
            }

    visual_understanding = {
        item["base_prompt_id"]: item
        for item in VISUAL_UNDERSTANDING_PROMPTS
    }

    return {
        "nsfw": nsfw,
        "image_classification": classification,
        "object_detection": object_detection,
        "segmentation": segmentation,
        "visual_understanding": visual_understanding,
    }


def process_file(
    input_path: Path,
    output_path: Path,
    task: str,
    target_model: str,
    prompt_lookups: Dict[str, Dict[str, Dict[str, Any]]],
) -> int:
    total = 0
    for chunk in pd.read_csv(
        input_path,
        chunksize=CHUNK_SIZE,
        usecols=usecols_for_task(task),
    ):
        rows = []
        for row in chunk.to_dict("records"):
            try:
                score_info = score_row(row, task, target_model, prompt_lookups)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to score {input_path} row prompt_id={row.get('prompt_id')!r} "
                    f"modifier={row.get('modifier_text')!r}"
                ) from exc
            rows.append(output_row(row, task, target_model, score_info))
        pd.DataFrame(rows, columns=OUTPUT_FIELDS).to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
            mode="w" if total == 0 else "a",
            header=total == 0,
        )
        total += len(rows)
    return total


def usecols_for_task(task: str):
    base_columns = {
        "prompt_id",
        "base_prompt",
        "modifier_text",
        "modifier_category",
        "modifier_family",
    }
    prefix_by_task = {
        "nsfw": ("score",),
        "image_classification": ("predictions",),
        "object_detection": ("objects",),
        "segmentation": ("details",),
        "visual_understanding": ("match",),
    }
    extra_by_task = {
        "image_classification": {"label_tiers", "tier_weights"},
        "object_detection": {"detector_target_objects", "expected_count", "object_class"},
        "segmentation": set(),
        "visual_understanding": {"match_rate"},
        "nsfw": set(),
    }
    prefixes = prefix_by_task[task]
    allowed = base_columns | extra_by_task[task]

    def include(column: str) -> bool:
        if column in allowed:
            return True
        return any(re.fullmatch(rf"{re.escape(prefix)}_\d+", column) for prefix in prefixes)

    return include


def score_row(
    row: Dict[str, Any],
    task: str,
    target_model: str,
    prompt_lookups: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    if task == "nsfw":
        image_scores = numeric_indexed_values(row, "score")
    elif task == "image_classification":
        prompt_eval = classification_prompt_eval(row, prompt_lookups)
        outputs = [{"predictions": parse_cell(row.get(col), [])} for col in indexed_columns(row, "predictions")]
        image_scores = evaluate_scores_local(task, target_model, prompt_eval, outputs)
    elif task == "object_detection":
        prompt_eval = object_detection_prompt_eval(row, prompt_lookups)
        outputs = [{"objects": parse_cell(row.get(col), [])} for col in indexed_columns(row, "objects")]
        image_scores = evaluate_scores_local(task, target_model, prompt_eval, outputs)
    elif task == "segmentation":
        outputs = segmentation_outputs(row, target_model)
        image_scores = evaluate_scores_local(task, target_model, {"eval_fields": {}}, outputs)
    elif task == "visual_understanding":
        image_scores = visual_understanding_scores(row)
    else:
        raise ValueError(f"Unsupported task: {task}")

    return {
        "score": round(mean(image_scores), 6),
        "num_image_scores": len(image_scores),
        "image_scores": json.dumps([round(float(x), 6) for x in image_scores], ensure_ascii=False),
    }


def classification_prompt_eval(
    row: Dict[str, Any],
    prompt_lookups: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    prompt = prompt_lookups["image_classification"].get(str(row.get("prompt_id")))
    if prompt:
        label_tiers = prompt["label_tiers"]
        tier_weights = prompt["tier_weights"]
    else:
        label_tiers = parse_cell(row.get("label_tiers"), {})
        tier_weights = parse_cell(row.get("tier_weights"), TIER_WEIGHTS)
    return {
        "eval_fields": {
            "label_tiers": label_tiers,
            "tier_weights": tier_weights,
        }
    }


def object_detection_prompt_eval(
    row: Dict[str, Any],
    prompt_lookups: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    prompt = prompt_lookups["object_detection"].get(str(row.get("prompt_id")))
    if prompt:
        target_objects = prompt["target_objects"]
        expected_count = prompt["expected_count"]
    else:
        target_objects = parse_cell(row.get("detector_target_objects"), None)
        if target_objects is None:
            target_objects = [str(row["object_class"])]
        expected_raw = parse_cell(row.get("expected_count"), row.get("expected_count"))
        if isinstance(expected_raw, dict):
            expected_count = expected_raw
        else:
            expected_count = {str(target_objects[0]): int(float(expected_raw))}
    return {
        "eval_fields": {
            "target_objects": target_objects,
            "expected_count": expected_count,
        }
    }


def segmentation_outputs(row: Dict[str, Any], target_model: str) -> List[Dict[str, Any]]:
    outputs = []
    for col in indexed_columns(row, "details"):
        details = parse_cell(row.get(col), {})
        suffix = col.rsplit("_", 1)[-1]
        image_area = segmentation_image_area(row, suffix)
        outputs.append(normalize_segmentation_details(details, target_model, image_area))
    return outputs


def normalize_segmentation_details(details: Dict[str, Any], target_model: str, image_area: int) -> Dict[str, Any]:
    if target_model == "deeplabv3plus":
        source_regions = details.get("classes", [])
    elif target_model == "sam2":
        source_regions = details.get("masks", [])
    else:
        raise ValueError(f"Unsupported segmentation model: {target_model}")

    regions = []
    for region in source_regions:
        confidence = float(region.get("confidence", 0.0))
        stability = float(region.get("stability_score", confidence))
        ratio = region.get("ratio")
        if ratio is None:
            area = float(region.get("area", 0.0))
            ratio = area / float(image_area) if image_area > 0 else 0.0
        regions.append({
            "ratio": float(ratio),
            "confidence": confidence,
            "stability_score": stability,
        })
    return {"regions": regions}


def segmentation_image_area(row: Dict[str, Any], suffix: str) -> int:
    return IMAGE_SIZE_FALLBACK


def visual_understanding_scores(row: Dict[str, Any]) -> List[float]:
    match_rate = row.get("match_rate")
    if not is_missing(match_rate):
        return [_clip01(1.0 - float(match_rate))]

    scores = []
    for col in indexed_columns(row, "match"):
        value = row.get(col)
        if is_missing(value):
            continue
        scores.append(_clip01(1.0 - float(value)))
    return scores or [1.0]


def evaluate_scores_local(
    task: str,
    target_model: str,
    prompt_eval_output: Dict[str, Any],
    model_eval_output: List[Dict[str, Any]],
) -> List[float]:
    eval_fields = prompt_eval_output.get("eval_fields", prompt_eval_output)

    if task == "image_classification":
        label_tiers = eval_fields["label_tiers"]
        tier_weights = eval_fields["tier_weights"]
        scores = []
        for item in model_eval_output:
            predictions = item.get("predictions", [])[:3]
            if not predictions:
                scores.append(1.0)
                continue
            numerator = 0.0
            denominator = 0.0
            for i, pred in enumerate(predictions):
                p_i = _clip01(pred["confidence"])
                r_i = CLASS_RANK_WEIGHTS[i]
                w_i = _get_tier_weight(pred["class"], label_tiers, tier_weights)
                numerator += r_i * p_i * w_i
                denominator += r_i * p_i
            score = 1.0 if denominator <= 0 else 1.0 - (numerator / denominator)
            scores.append(_clip01(score))
        return scores

    if task == "object_detection":
        target_objects = eval_fields["target_objects"]
        expected_count = eval_fields["expected_count"]
        scores = []
        for item in model_eval_output:
            objects = item.get("objects", [])
            total_expected = 0
            weighted_quality = 0.0
            for target in target_objects:
                target_norm = _normalize_label(target)
                n = int(expected_count[target])
                if n <= 0:
                    continue
                matched = [
                    obj for obj in objects
                    if _canonical_detection_label(obj["class"])
                    == _canonical_detection_label(target_norm)
                ]
                m = len(matched)
                confidences = sorted((_clip01(obj["confidence"]) for obj in matched), reverse=True)
                selected = confidences[:n]
                if len(selected) < n:
                    selected += [0.0] * (n - len(selected))
                confidence_quality = sum(selected) / n
                count_quality = 1.0 if max(m, n) == 0 else min(m, n) / max(m, n)
                quality = math.sqrt(confidence_quality * count_quality)
                weighted_quality += n * quality
                total_expected += n
            if total_expected <= 0:
                raise ValueError("No positive expected object count.")
            scores.append(_clip01(1.0 - (weighted_quality / total_expected)))
        return scores

    if task == "segmentation":
        scores = []
        for item in model_eval_output:
            regions = item.get("regions", [])
            if not regions:
                scores.append(1.0)
                continue
            weighted_score = 0.0
            total_ratio = 0.0
            for region in regions:
                ratio = max(0.0, float(region["ratio"]))
                confidence = _clip01(region["confidence"])
                stability = _clip01(region["stability_score"])
                region_score = 1.0 - math.sqrt(confidence * stability)
                weighted_score += ratio * region_score
                total_ratio += ratio
            score = 1.0 if total_ratio <= 0 else weighted_score / total_ratio
            scores.append(_clip01(score))
        return scores

    raise ValueError(f"Unsupported offline scoring task: {task}")


def output_row(
    row: Dict[str, Any],
    task: str,
    target_model: str,
    score_info: Dict[str, Any],
) -> Dict[str, Any]:
    modifier = clean_scalar(row.get("modifier_text"))
    modifier_category = clean_scalar(row.get("modifier_category"))
    return {
        "task": task,
        "target_model": target_model,
        "base_prompt": clean_scalar(row.get("base_prompt")),
        "modifier": modifier,
        "modifier_category": modifier_category,
        "modifier_family": infer_modifier_family(modifier_category, modifier, row.get("modifier_family")),
        "score": score_info["score"],
        "image_scores": score_info["image_scores"],
    }


def infer_modifier_family(category: Any, modifier: Any, fallback: Any) -> str:
    for value in (category, modifier):
        text = str(value).strip().lower()
        if not text or text == "nan":
            continue
        if text.endswith("mecha"):
            return "mechanism"
        if text.endswith("eff"):
            return "effect"

    fallback_text = str(clean_scalar(fallback)).strip().lower()
    if fallback_text in {"mechanism", "effect", "control"}:
        return fallback_text
    if fallback_text.endswith("mecha"):
        return "mechanism"
    if fallback_text.endswith("eff"):
        return "effect"
    return fallback_text


def indexed_columns(row: Dict[str, Any], prefix: str) -> List[str]:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    matched = []
    for key in row:
        match = pattern.match(str(key))
        if match:
            matched.append((int(match.group(1)), str(key)))
    return [key for _, key in sorted(matched)]


def numeric_indexed_values(row: Dict[str, Any], prefix: str) -> List[float]:
    values = []
    for col in indexed_columns(row, prefix):
        value = row.get(col)
        if is_missing(value):
            continue
        values.append(_clip01(float(value)))
    return values or [1.0]


def parse_cell(value: Any, default: Any) -> Any:
    if is_missing(value):
        return default
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return default


def clean_scalar(value: Any) -> Any:
    if is_missing(value):
        return ""
    return value


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def mean(values: Iterable[float]) -> float:
    items = [float(v) for v in values]
    if not items:
        return 1.0
    return sum(items) / len(items)


def _clip01(x: Any) -> float:
    return max(0.0, min(1.0, float(x)))


def _normalize_label(text: Any) -> str:
    return str(text).lower().strip()


def _canonical_detection_label(text: Any) -> str:
    label = _normalize_label(text)
    return DETECTION_CLASS_ALIASES.get(label, label)


def _prediction_aliases(text: Any) -> List[str]:
    return [x.strip().lower() for x in str(text).split(",") if x.strip()]


def _get_tier_weight(pred_class: Any, label_tiers: Dict[str, Any], tier_weights: Dict[str, Any]) -> float:
    aliases = set(_prediction_aliases(pred_class))
    for tier in ("exact", "similar", "related"):
        tier_labels = {_normalize_label(x) for x in label_tiers.get(tier, [])}
        if aliases & tier_labels:
            return float(tier_weights.get(tier, 0.0))
    return float(tier_weights.get("miss", 0.0))


if __name__ == "__main__":
    main()
