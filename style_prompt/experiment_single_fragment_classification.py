#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

from style_modifier_design import freq_mecha, mixhigh_eff, sinhigh_eff, style_mecha
from test_prompts_design import build_image_classification_prompts


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "experiment_outputs" / "modifier_results" / "classification"
IMAGE_DIR = SCRIPT_DIR / "img" / "single_fragment" / "classification"

GENERATOR_MODELS = ("sd15", "sdxl", "sd3")
CLASSIFIERS = ("resnet50", "mambavision")
SEED = 2026
NUM_IMAGES = 10
DEFAULT_BASE_PROMPTS = 60
DEFAULT_MODIFIERS = 660
DEFAULT_MODIFIERS_WITH_CONTROL = 661
DEFAULT_TOPK = 3

REDUCED_MECHANISM_TABLES = (
    ("freq_mecha", freq_mecha),
    ("style_mecha", style_mecha),
)

REDUCED_EFFECT_TABLES = (
    ("sinhigh_eff", sinhigh_eff, "single_effect_top_k"),
    ("mixhigh_eff", mixhigh_eff, "mixed_effect_top_k"),
)


@dataclass(frozen=True)
class BasePrompt:
    prompt_id: str
    classification_group: str
    target_class: str
    prompt_subject: str
    template_type: str
    text: str
    label_tiers: Dict[str, List[str]]
    tier_weights: Dict[str, float]


@dataclass(frozen=True)
class Modifier:
    family: str
    category: str
    index: int
    text: str


@dataclass(frozen=True)
class ImageTask:
    base: BasePrompt
    modifier: Modifier
    image_dir: Path
    combined_prompt: str


def iter_base_prompts() -> Iterator[BasePrompt]:
    for record in build_image_classification_prompts():
        yield BasePrompt(
            prompt_id=record["base_prompt_id"],
            classification_group=record["classification_group"],
            target_class=record["target_class"],
            prompt_subject=record["prompt_subject"],
            template_type=record["template_type"],
            text=record["base_prompt"],
            label_tiers={
                tier: list(values)
                for tier, values in record["label_tiers"].items()
            },
            tier_weights={
                tier: float(weight)
                for tier, weight in record["tier_weights"].items()
            },
        )


def iter_modifiers(
    include_control: bool,
    mechanism_top_k: int,
    single_effect_top_k: int,
    mixed_effect_top_k: int,
) -> Iterator[Modifier]:
    if include_control:
        yield Modifier("control", "none", 0, "")

    for family, table in REDUCED_MECHANISM_TABLES:
        for category, descriptors in table.items():
            for index, descriptor in enumerate(descriptors[:mechanism_top_k], start=1):
                yield Modifier(family, category, index, descriptor)

    effect_limits = {
        "single_effect_top_k": single_effect_top_k,
        "mixed_effect_top_k": mixed_effect_top_k,
    }
    for family, table, limit_name in REDUCED_EFFECT_TABLES:
        limit = effect_limits[limit_name]
        for category, descriptors in table.items():
            for index, descriptor in enumerate(descriptors[:limit], start=1):
                yield Modifier(family, category, index, descriptor)


def compose_prompt(base_prompt: str, modifier: str) -> str:
    if not modifier:
        return base_prompt
    return f"{base_prompt}, {modifier}"


def combo_key(base: BasePrompt, modifier: Modifier) -> str:
    return "|".join(
        [
            base.prompt_id,
            base.classification_group,
            base.target_class,
            base.template_type,
            modifier.family,
            modifier.category,
            str(modifier.index),
        ]
    )


def safe_path_part(value: object) -> str:
    text = str(value) if value not in (None, "") else "none"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe[:120] or "none"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path)


@contextmanager
def suppress_output():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def modifier_dir_name(modifier: Modifier) -> str:
    return "__".join(
        [
            safe_path_part(modifier.family),
            safe_path_part(modifier.category),
            f"{modifier.index:04d}",
        ]
    )


def image_dir_for(generator_model: str, base: BasePrompt, modifier: Modifier) -> Path:
    return (
        IMAGE_DIR
        / safe_path_part(generator_model)
        / safe_path_part(base.classification_group)
        / safe_path_part(base.prompt_id)
        / modifier_dir_name(modifier)
    )


def prediction_dir_for(
    generator_model: str,
    classifier_name: str,
    base: BasePrompt,
    modifier: Modifier,
) -> Path:
    return (
        OUTPUT_DIR
        / "predictions"
        / safe_path_part(generator_model)
        / safe_path_part(classifier_name)
        / safe_path_part(base.classification_group)
        / safe_path_part(base.prompt_id)
        / modifier_dir_name(modifier)
    )


def result_path(generator_model: str, classifier_name: str) -> Path:
    return OUTPUT_DIR / f"{generator_model}_{classifier_name}_classification.csv"


def summary_json_path(
    generator_model: str,
    classifier_name: str,
    base: BasePrompt,
    modifier: Modifier,
) -> Path:
    return prediction_dir_for(generator_model, classifier_name, base, modifier) / "summary.json"


def image_set_complete(image_dir: Path) -> bool:
    return image_dir.is_dir() and all(
        (image_dir / f"{idx:03d}.png").is_file() for idx in range(NUM_IMAGES)
    )


def materialize_images(images: Sequence[Image.Image], image_dir: Path) -> None:
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images):
        image.convert("RGB").save(image_dir / f"{idx:03d}.png")


def load_done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()

    done = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = row.get("combo_key")
            if key:
                done.add(key)
    return done


def load_existing_summary(path: Path, topk: int) -> Optional[List[Dict[str, Any]]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) != NUM_IMAGES:
        return None
    for item in data:
        predictions = item.get("predictions", [])
        if not isinstance(predictions, list) or len(predictions) < topk:
            return None
    return data


def normalize_class_name(value: str) -> str:
    return str(value).lower().strip()


def tier_for_predictions(
    predictions: Sequence[Dict[str, Any]],
    label_tiers: Dict[str, List[str]],
) -> str:
    prediction_labels = {
        normalize_class_name(item.get("class", ""))
        for item in predictions
    }
    for tier in ("exact", "similar", "related"):
        tier_labels = {normalize_class_name(label) for label in label_tiers.get(tier, [])}
        if prediction_labels & tier_labels:
            return tier
    return "miss"


def top1_class_columns() -> List[str]:
    return [f"top1_class_{idx:02d}" for idx in range(NUM_IMAGES)]


def confidence_columns() -> List[str]:
    return [f"top1_confidence_{idx:02d}" for idx in range(NUM_IMAGES)]


def match_tier_columns() -> List[str]:
    return [f"match_tier_{idx:02d}" for idx in range(NUM_IMAGES)]


def tier_score_columns() -> List[str]:
    return [f"tier_score_{idx:02d}" for idx in range(NUM_IMAGES)]


def prediction_columns() -> List[str]:
    return [f"predictions_{idx:02d}" for idx in range(NUM_IMAGES)]


def result_fieldnames() -> List[str]:
    return [
        "combo_key",
        "generator_model",
        "classifier",
        "prompt_id",
        "classification_group",
        "target_class",
        "prompt_subject",
        "template_type",
        "modifier_family",
        "modifier_category",
        "modifier_index",
        "modifier_text",
        "seed",
        "num_images",
        "topk",
        "base_prompt",
        "combined_prompt",
        "image_dir",
        "prediction_json",
        "top1_confidence_mean",
        "match_rate",
        "tier_score_mean",
        "label_tiers",
        *top1_class_columns(),
        *confidence_columns(),
        *match_tier_columns(),
        *tier_score_columns(),
        *prediction_columns(),
    ]


def open_result_writer(generator_model: str, classifier_name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = result_path(generator_model, classifier_name)
    exists = path.exists() and path.stat().st_size > 0
    f = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=result_fieldnames())
    if not exists:
        writer.writeheader()
        f.flush()
    return f, writer


def results_by_image_name(results: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {Path(item["image"]).name: dict(item) for item in results}


def row_for_classification(
    generator_model: str,
    classifier_name: str,
    base: BasePrompt,
    modifier: Modifier,
    combined_prompt: str,
    image_dir: Path,
    prediction_json: Path,
    topk: int,
    results: Sequence[Dict[str, Any]],
) -> Dict[str, object]:
    by_name = results_by_image_name(results)
    top1_classes: List[str] = []
    confidences: List[float] = []
    match_tiers: List[str] = []
    tier_scores: List[float] = []
    predictions_by_image: List[str] = []

    for idx in range(NUM_IMAGES):
        image_name = f"{idx:03d}.png"
        item = by_name.get(image_name, {})
        predictions = list(item.get("predictions", []))
        top1_classes.append(str(item.get("class", "")))
        confidences.append(float(item.get("confidence", 0.0)))
        tier = tier_for_predictions(predictions, base.label_tiers)
        match_tiers.append(tier)
        tier_scores.append(float(base.tier_weights.get(tier, 0.0)))
        predictions_by_image.append(json.dumps(predictions, ensure_ascii=False))

    row: Dict[str, object] = {
        "combo_key": combo_key(base, modifier),
        "generator_model": generator_model,
        "classifier": classifier_name,
        "prompt_id": base.prompt_id,
        "classification_group": base.classification_group,
        "target_class": base.target_class,
        "prompt_subject": base.prompt_subject,
        "template_type": base.template_type,
        "modifier_family": modifier.family,
        "modifier_category": modifier.category,
        "modifier_index": modifier.index,
        "modifier_text": modifier.text,
        "seed": SEED,
        "num_images": NUM_IMAGES,
        "topk": topk,
        "base_prompt": base.text,
        "combined_prompt": combined_prompt,
        "image_dir": display_path(image_dir),
        "prediction_json": display_path(prediction_json),
        "top1_confidence_mean": round(sum(confidences) / NUM_IMAGES, 4),
        "match_rate": round(sum(1 for tier in match_tiers if tier != "miss") / NUM_IMAGES, 4),
        "tier_score_mean": round(sum(tier_scores) / NUM_IMAGES, 4),
        "label_tiers": json.dumps(base.label_tiers, ensure_ascii=False),
    }
    for col, value in zip(top1_class_columns(), top1_classes):
        row[col] = value
    for col, value in zip(confidence_columns(), confidences):
        row[col] = round(value, 4)
    for col, value in zip(match_tier_columns(), match_tiers):
        row[col] = value
    for col, value in zip(tier_score_columns(), tier_scores):
        row[col] = round(value, 4)
    for col, value in zip(prediction_columns(), predictions_by_image):
        row[col] = value
    return row


def cleanup_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_plan(
    include_control: bool,
    mechanism_top_k: int,
    single_effect_top_k: int,
    mixed_effect_top_k: int,
) -> Tuple[List[BasePrompt], List[Modifier]]:
    base_prompts = list(iter_base_prompts())
    modifiers = list(
        iter_modifiers(
            include_control=include_control,
            mechanism_top_k=mechanism_top_k,
            single_effect_top_k=single_effect_top_k,
            mixed_effect_top_k=mixed_effect_top_k,
        )
    )

    expected_modifier_count = (
        DEFAULT_MODIFIERS_WITH_CONTROL if include_control else DEFAULT_MODIFIERS
    )
    if (
        mechanism_top_k == 30
        and single_effect_top_k == 20
        and mixed_effect_top_k == 10
        and (
            len(base_prompts) != DEFAULT_BASE_PROMPTS
            or len(modifiers) != expected_modifier_count
        )
    ):
        raise ValueError(
            f"Expected {DEFAULT_BASE_PROMPTS} prompts and "
            f"{expected_modifier_count} modifiers, got "
            f"{len(base_prompts)} prompts and {len(modifiers)} modifiers."
        )

    return base_prompts, modifiers


def generation_progress_total(
    models: Sequence[str],
    base_prompts: Sequence[BasePrompt],
    modifiers: Sequence[Modifier],
) -> int:
    return len(models) * len(base_prompts) * len(modifiers)


def classification_progress_total(
    models: Sequence[str],
    classifiers: Sequence[str],
    base_prompts: Sequence[BasePrompt],
    modifiers: Sequence[Modifier],
) -> int:
    return len(models) * len(classifiers) * len(base_prompts) * len(modifiers)


def ensure_images_for_generator(
    generator_model: str,
    base_prompts: Sequence[BasePrompt],
    modifiers: Sequence[Modifier],
    limit: Optional[int],
    height: int,
    width: int,
    generator_device: str,
    prompt_batch_size: int,
    progress: tqdm,
) -> None:
    from tools_generator import get_images, load_generator_model

    with suppress_output():
        pipe = load_generator_model(generator_model, device=generator_device, check=False)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    generated_count = 0

    def flush_generation_batch(tasks: List[ImageTask]) -> bool:
        nonlocal generated_count
        if not tasks:
            return False

        with suppress_output():
            generated_images = get_images(
                pipe,
                prompt=[task.combined_prompt for task in tasks],
                seed=SEED,
                num_images_per_prompt=NUM_IMAGES,
                height=height,
                width=width,
            )

        expected_images = len(tasks) * NUM_IMAGES
        if len(generated_images) != expected_images:
            raise ValueError(
                f"Expected {expected_images} generated images, got {len(generated_images)}"
            )

        for task_index, task in enumerate(tasks):
            start = task_index * NUM_IMAGES
            images = generated_images[start:start + NUM_IMAGES]
            materialize_images(images, task.image_dir)
            generated_count += 1
            progress.update(1)

        return limit is not None and generated_count >= limit

    try:
        for base in base_prompts:
            pending_generation: List[ImageTask] = []
            for modifier in modifiers:
                image_dir = image_dir_for(generator_model, base, modifier)
                if image_set_complete(image_dir):
                    progress.update(1)
                    continue

                if limit is not None and generated_count >= limit:
                    return

                task = ImageTask(
                    base=base,
                    modifier=modifier,
                    image_dir=image_dir,
                    combined_prompt=compose_prompt(base.text, modifier.text),
                )
                pending_generation.append(task)
                remaining = None if limit is None else max(limit - generated_count, 1)
                current_batch_size = (
                    prompt_batch_size
                    if remaining is None
                    else min(prompt_batch_size, remaining)
                )
                if len(pending_generation) >= current_batch_size:
                    if flush_generation_batch(pending_generation):
                        return
                    pending_generation = []

            if flush_generation_batch(pending_generation):
                return

    finally:
        del pipe
        cleanup_memory()


def classifier_batch_size(
    classifier_name: str,
    resnet_batch_size: int,
    mambavision_batch_size: int,
) -> int:
    if classifier_name == "resnet50":
        return resnet_batch_size
    if classifier_name == "mambavision":
        return mambavision_batch_size
    raise ValueError(f"Unknown classifier: {classifier_name}")


def run_classification_pass(
    generator_model: str,
    classifier_name: str,
    base_prompts: Sequence[BasePrompt],
    modifiers: Sequence[Modifier],
    limit: Optional[int],
    classifier_device: str,
    resnet_batch_size: int,
    mambavision_batch_size: int,
    topk: int,
    mambavision_model_name: str,
    progress: tqdm,
) -> None:
    from tools_task import classify_images, load_image_classifier

    with suppress_output():
        if classifier_name == "mambavision":
            classifier = load_image_classifier(
                classifier_name,
                device=classifier_device,
                model_name=mambavision_model_name,
            )
        else:
            classifier = load_image_classifier(classifier_name, device=classifier_device)

    batch_size = classifier_batch_size(
        classifier_name,
        resnet_batch_size,
        mambavision_batch_size,
    )
    f, writer = open_result_writer(generator_model, classifier_name)
    done_keys = load_done_keys(result_path(generator_model, classifier_name))
    processed_count = 0

    try:
        for base in base_prompts:
            for modifier in modifiers:
                key = combo_key(base, modifier)
                image_dir = image_dir_for(generator_model, base, modifier)

                if key in done_keys:
                    progress.update(1)
                    continue

                if not image_set_complete(image_dir):
                    if limit is None:
                        raise FileNotFoundError(f"Missing generated images: {image_dir}")
                    progress.update(1)
                    continue

                combined_prompt = compose_prompt(base.text, modifier.text)
                prediction_dir = prediction_dir_for(
                    generator_model,
                    classifier_name,
                    base,
                    modifier,
                )
                prediction_json = summary_json_path(
                    generator_model,
                    classifier_name,
                    base,
                    modifier,
                )
                results = load_existing_summary(prediction_json, topk)
                if results is None:
                    prediction_dir.mkdir(parents=True, exist_ok=True)
                    with suppress_output():
                        results = classify_images(
                            classifier,
                            image_dir,
                            batch_size=batch_size,
                            topk=topk,
                            output_json=prediction_json,
                        )

                writer.writerow(
                    row_for_classification(
                        generator_model=generator_model,
                        classifier_name=classifier_name,
                        base=base,
                        modifier=modifier,
                        combined_prompt=combined_prompt,
                        image_dir=image_dir,
                        prediction_json=prediction_json,
                        topk=topk,
                        results=results,
                    )
                )
                f.flush()
                done_keys.add(key)
                processed_count += 1
                progress.update(1)

                if limit is not None and processed_count >= limit:
                    return

    finally:
        f.close()
        del classifier
        cleanup_memory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-fragment style modifier image classification experiments."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=GENERATOR_MODELS,
        default=list(GENERATOR_MODELS),
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=CLASSIFIERS,
        default=list(CLASSIFIERS),
    )
    parser.add_argument(
        "--mechanism-top-k",
        type=int,
        default=30,
        help="Top descriptors per mechanism category from freq_mecha and style_mecha.",
    )
    parser.add_argument(
        "--single-effect-top-k",
        type=int,
        default=20,
        help="Top descriptors per effect dimension from sinhigh_eff.",
    )
    parser.add_argument(
        "--mixed-effect-top-k",
        type=int,
        default=10,
        help="Top descriptors per effect dimension from mixhigh_eff.",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--generator-device", default="cuda")
    parser.add_argument("--classifier-device", default="cuda:0")
    parser.add_argument("--resnet-batch-size", type=int, default=16)
    parser.add_argument("--mambavision-batch-size", type=int, default=10)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--mambavision-model-name", default="nvidia/MambaVision-T-1K")
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=2,
        help="Number of combined prompts generated per diffusion call.",
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="Do not include the no-style baseline condition.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N missing/new combinations per generator/classifier for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    os.chdir(SCRIPT_DIR)
    args = parse_args()
    if args.mechanism_top_k <= 0:
        raise ValueError("--mechanism-top-k must be > 0")
    if args.single_effect_top_k <= 0:
        raise ValueError("--single-effect-top-k must be > 0")
    if args.mixed_effect_top_k <= 0:
        raise ValueError("--mixed-effect-top-k must be > 0")
    if args.prompt_batch_size <= 0:
        raise ValueError("--prompt-batch-size must be > 0")
    if args.resnet_batch_size <= 0:
        raise ValueError("--resnet-batch-size must be > 0")
    if args.mambavision_batch_size <= 0:
        raise ValueError("--mambavision-batch-size must be > 0")
    if args.topk <= 0:
        raise ValueError("--topk must be > 0")

    base_prompts, modifiers = build_plan(
        include_control=not args.no_control,
        mechanism_top_k=args.mechanism_top_k,
        single_effect_top_k=args.single_effect_top_k,
        mixed_effect_top_k=args.mixed_effect_top_k,
    )

    with tqdm(
        total=generation_progress_total(args.models, base_prompts, modifiers),
        desc="single_fragment_classification_generation",
        unit="combo",
        dynamic_ncols=True,
    ) as generation_progress:
        for generator_model in args.models:
            ensure_images_for_generator(
                generator_model=generator_model,
                base_prompts=base_prompts,
                modifiers=modifiers,
                limit=args.limit,
                height=args.height,
                width=args.width,
                generator_device=args.generator_device,
                prompt_batch_size=args.prompt_batch_size,
                progress=generation_progress,
            )

    with tqdm(
        total=classification_progress_total(
            args.models,
            args.classifiers,
            base_prompts,
            modifiers,
        ),
        desc="single_fragment_classification",
        unit="combo",
        dynamic_ncols=True,
    ) as classification_progress:
        for generator_model in args.models:
            for classifier_name in args.classifiers:
                run_classification_pass(
                    generator_model=generator_model,
                    classifier_name=classifier_name,
                    base_prompts=base_prompts,
                    modifiers=modifiers,
                    limit=args.limit,
                    classifier_device=args.classifier_device,
                    resnet_batch_size=args.resnet_batch_size,
                    mambavision_batch_size=args.mambavision_batch_size,
                    topk=args.topk,
                    mambavision_model_name=args.mambavision_model_name,
                    progress=classification_progress,
                )


if __name__ == "__main__":
    main()
