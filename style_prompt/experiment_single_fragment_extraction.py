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

from style_modifier_design import (
    class_mecha,
    destyle_mecha,
    freq_mecha,
    long_mecha,
    mixhigh_eff,
    mixlow_eff,
    sinhigh_eff,
    sinlow_eff,
    style_mecha,
)
from test_prompts_design import build_object_detection_prompts


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "experiment_outputs" / "modifier_results" / "object_extraction"
IMAGE_DIR = SCRIPT_DIR / "img" / "single_fragment" / "object_extraction"

GENERATOR_MODELS = ("sd15", "sdxl", "sd3")
EXTRACTORS = ("deim", "yolo11")
SEED = 2026
NUM_IMAGES = 10
DEFAULT_REDUCED_MODIFIERS = 660
DEFAULT_REDUCED_MODIFIERS_WITH_CONTROL = 661
OBJECT_CLASS_ALIASES = {
    "sofa": "couch",
}

MODIFIER_TABLES = (
    ("freq_mecha", freq_mecha),
    ("style_mecha", style_mecha),
    ("class_mecha", class_mecha),
    ("destyle_mecha", destyle_mecha),
    ("long_mecha", long_mecha),
    ("sinhigh_eff", sinhigh_eff),
    ("mixhigh_eff", mixhigh_eff),
    ("sinlow_eff", sinlow_eff),
    ("mixlow_eff", mixlow_eff),
)

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
    object_class: str
    template_type: str
    text: str
    target_objects: Tuple[str, ...]
    expected_count: int


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


@dataclass(frozen=True)
class ExtractionTask:
    base: BasePrompt
    modifier: Modifier
    key: str
    image_dir: Path
    combined_prompt: str
    targets: Tuple[str, ...]
    missing_result: bool


def iter_base_prompts() -> Iterator[BasePrompt]:
    for record in build_object_detection_prompts():
        object_class = record["object_class"]
        expected_count = int(record["expected_count"][object_class])
        yield BasePrompt(
            prompt_id=record["base_prompt_id"],
            object_class=object_class,
            template_type=record["template_type"],
            text=record["base_prompt"],
            target_objects=tuple(record["target_objects"]),
            expected_count=expected_count,
        )


def iter_modifiers(include_control: bool = True) -> Iterator[Modifier]:
    if include_control:
        yield Modifier("control", "none", 0, "")

    for family, table in MODIFIER_TABLES:
        for category, descriptors in table.items():
            for index, descriptor in enumerate(descriptors, start=1):
                yield Modifier(family, category, index, descriptor)


def iter_reduced_modifiers(
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
            base.object_class,
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


def image_dir_for(generator_model: str, base: BasePrompt, modifier: Modifier) -> Path:
    modifier_dir = "__".join(
        [
            safe_path_part(modifier.family),
            safe_path_part(modifier.category),
            f"{modifier.index:04d}",
        ]
    )
    return (
        IMAGE_DIR
        / safe_path_part(generator_model)
        / safe_path_part(base.object_class)
        / safe_path_part(base.template_type)
        / safe_path_part(base.prompt_id)
        / modifier_dir
    )


def count_columns(prefix: str) -> List[str]:
    return [f"{prefix}_{idx:02d}" for idx in range(NUM_IMAGES)]


def object_columns() -> List[str]:
    return [f"objects_{idx:02d}" for idx in range(NUM_IMAGES)]


def result_fieldnames() -> List[str]:
    return [
        "combo_key",
        "generator_model",
        "extractor",
        "prompt_id",
        "object_class",
        "detector_target_objects",
        "template_type",
        "expected_count",
        "modifier_family",
        "modifier_category",
        "modifier_index",
        "modifier_text",
        "seed",
        "num_images",
        "base_prompt",
        "combined_prompt",
        "image_dir",
        "detected_total",
        "detected_mean",
        "exact_count_rate",
        "at_least_expected_rate",
        *count_columns("detected_count"),
        *count_columns("exact_match"),
        *count_columns("at_least_expected"),
        *object_columns(),
    ]


def result_path(generator_model: str, extractor_name: str) -> Path:
    return OUTPUT_DIR / f"{generator_model}_{extractor_name}_object_extraction.csv"


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


def open_result_writer(generator_model: str, extractor_name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = result_path(generator_model, extractor_name)
    exists = path.exists() and path.stat().st_size > 0
    f = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=result_fieldnames())
    if not exists:
        writer.writeheader()
        f.flush()
    return f, writer


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


def detector_targets(base: BasePrompt) -> List[str]:
    return [OBJECT_CLASS_ALIASES.get(target, target) for target in base.target_objects]


def latest_json_path(generator_model: str, extractor_name: str) -> Path:
    return OUTPUT_DIR / "_latest_extract_objects" / f"{generator_model}_{extractor_name}.json"


def row_for_extraction(
    generator_model: str,
    extractor_name: str,
    base: BasePrompt,
    modifier: Modifier,
    combined_prompt: str,
    image_dir: Path,
    extraction_results: Sequence[Dict[str, Any]],
) -> Dict[str, object]:
    results_by_name = {
        Path(item["image"]).name: item for item in extraction_results
    }
    counts: List[int] = []
    objects_by_image: List[List[Dict[str, Any]]] = []

    for idx in range(NUM_IMAGES):
        image_name = f"{idx:03d}.png"
        item = results_by_name.get(image_name)
        objects = [] if item is None else list(item.get("objects", []))
        counts.append(len(objects))
        objects_by_image.append(objects)

    exact_matches = [1 if count == base.expected_count else 0 for count in counts]
    at_least_expected = [1 if count >= base.expected_count else 0 for count in counts]
    detected_total = sum(counts)

    row: Dict[str, object] = {
        "combo_key": combo_key(base, modifier),
        "generator_model": generator_model,
        "extractor": extractor_name,
        "prompt_id": base.prompt_id,
        "object_class": base.object_class,
        "detector_target_objects": json.dumps(detector_targets(base), ensure_ascii=False),
        "template_type": base.template_type,
        "expected_count": base.expected_count,
        "modifier_family": modifier.family,
        "modifier_category": modifier.category,
        "modifier_index": modifier.index,
        "modifier_text": modifier.text,
        "seed": SEED,
        "num_images": NUM_IMAGES,
        "base_prompt": base.text,
        "combined_prompt": combined_prompt,
        "image_dir": display_path(image_dir),
        "detected_total": detected_total,
        "detected_mean": round(detected_total / NUM_IMAGES, 4),
        "exact_count_rate": round(sum(exact_matches) / NUM_IMAGES, 4),
        "at_least_expected_rate": round(sum(at_least_expected) / NUM_IMAGES, 4),
    }

    for col, value in zip(count_columns("detected_count"), counts):
        row[col] = value
    for col, value in zip(count_columns("exact_match"), exact_matches):
        row[col] = value
    for col, value in zip(count_columns("at_least_expected"), at_least_expected):
        row[col] = value
    for col, objects in zip(object_columns(), objects_by_image):
        row[col] = json.dumps(objects, ensure_ascii=False)

    return row


def cleanup_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_plan(
    include_control: bool,
    plan: str,
    mechanism_top_k: int,
    single_effect_top_k: int,
    mixed_effect_top_k: int,
) -> Tuple[List[BasePrompt], List[Modifier]]:
    base_prompts = list(iter_base_prompts())
    if plan == "full":
        modifiers = list(iter_modifiers(include_control=include_control))
    elif plan == "reduced":
        modifiers = list(
            iter_reduced_modifiers(
                include_control=include_control,
                mechanism_top_k=mechanism_top_k,
                single_effect_top_k=single_effect_top_k,
                mixed_effect_top_k=mixed_effect_top_k,
            )
        )
    else:
        raise ValueError(f"Unknown plan: {plan}")

    expected_modifier_count = (
        DEFAULT_REDUCED_MODIFIERS_WITH_CONTROL
        if include_control
        else DEFAULT_REDUCED_MODIFIERS
    )
    if (
        plan == "reduced"
        and mechanism_top_k == 30
        and single_effect_top_k == 20
        and mixed_effect_top_k == 10
        and len(modifiers) != expected_modifier_count
    ):
        raise ValueError(
            f"Expected reduced modifiers={expected_modifier_count}, got {len(modifiers)}"
        )

    return base_prompts, modifiers


def generation_progress_total(
    models: Sequence[str],
    base_prompts: Sequence[BasePrompt],
    modifiers: Sequence[Modifier],
) -> int:
    return len(models) * len(base_prompts) * len(modifiers)


def extraction_progress_total(
    models: Sequence[str],
    extractors: Sequence[str],
    base_prompts: Sequence[BasePrompt],
    modifiers: Sequence[Modifier],
) -> int:
    return len(models) * len(extractors) * len(base_prompts) * len(modifiers)


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

                combined_prompt = compose_prompt(base.text, modifier.text)
                task = ImageTask(base, modifier, image_dir, combined_prompt)
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

                if limit is not None and generated_count >= limit:
                    return

            if flush_generation_batch(pending_generation):
                return

    finally:
        del pipe
        cleanup_memory()


def run_extractor_pass(
    generator_model: str,
    extractor_name: str,
    base_prompts: Sequence[BasePrompt],
    modifiers: Sequence[Modifier],
    limit: Optional[int],
    extractor_device: str,
    deim_size: str,
    yolo_size: str,
    conf: float,
    batch_size: int,
    progress: tqdm,
) -> None:
    from tools_task import extract_objects, load_object_extractor

    extractor_size = deim_size if extractor_name == "deim" else yolo_size
    with suppress_output():
        extractor = load_object_extractor(
            extractor_name,
            model_size=extractor_size,
            device=extractor_device,
        )

    f, writer = open_result_writer(generator_model, extractor_name)
    done_keys = load_done_keys(result_path(generator_model, extractor_name))
    processed_count = 0

    try:
        for base in base_prompts:
            targets = tuple(detector_targets(base))
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
                output_json = latest_json_path(generator_model, extractor_name)
                output_json.parent.mkdir(parents=True, exist_ok=True)

                with suppress_output():
                    boxes = extract_objects(
                        extractor,
                        image_dir,
                        list(targets),
                        conf=conf,
                        batch_size=batch_size,
                        output_json=output_json,
                    )
                writer.writerow(
                    row_for_extraction(
                        generator_model=generator_model,
                        extractor_name=extractor_name,
                        base=base,
                        modifier=modifier,
                        combined_prompt=combined_prompt,
                        image_dir=image_dir,
                        extraction_results=boxes,
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
        del extractor
        cleanup_memory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-fragment style modifier object extraction experiments."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=GENERATOR_MODELS,
        default=list(GENERATOR_MODELS),
    )
    parser.add_argument(
        "--extractors",
        nargs="+",
        choices=EXTRACTORS,
        default=list(EXTRACTORS),
    )
    parser.add_argument(
        "--plan",
        choices=("reduced", "full"),
        default="reduced",
        help="Modifier selection plan. reduced uses the 660 selected modifiers plus optional control.",
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
    parser.add_argument("--extractor-device", default="cuda:0")
    parser.add_argument("--deim-size", default="n")
    parser.add_argument("--yolo-size", default="n")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=NUM_IMAGES,
        help="Object extraction batch size. Default processes one prompt directory at once.",
    )
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
        help="Process at most N new/unfinished combinations per generator/extractor.",
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
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.prompt_batch_size <= 0:
        raise ValueError("--prompt-batch-size must be > 0")

    base_prompts, modifiers = build_plan(
        include_control=not args.no_control,
        plan=args.plan,
        mechanism_top_k=args.mechanism_top_k,
        single_effect_top_k=args.single_effect_top_k,
        mixed_effect_top_k=args.mixed_effect_top_k,
    )

    with tqdm(
        total=generation_progress_total(args.models, base_prompts, modifiers),
        desc="single_fragment_generation",
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
        total=extraction_progress_total(args.models, args.extractors, base_prompts, modifiers),
        desc="single_fragment_extraction",
        unit="combo",
        dynamic_ncols=True,
    ) as extraction_progress:
        for generator_model in args.models:
            for extractor_name in args.extractors:
                run_extractor_pass(
                    generator_model=generator_model,
                    extractor_name=extractor_name,
                    base_prompts=base_prompts,
                    modifiers=modifiers,
                    limit=args.limit,
                    extractor_device=args.extractor_device,
                    deim_size=args.deim_size,
                    yolo_size=args.yolo_size,
                    conf=args.conf,
                    batch_size=args.batch_size,
                    progress=extraction_progress,
                )


if __name__ == "__main__":
    main()
