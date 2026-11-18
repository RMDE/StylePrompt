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
from test_prompts_design import SEGMENTATION_PROMPT_GROUPS


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "experiment_outputs" / "modifier_results" / "segmentation"
IMAGE_DIR = SCRIPT_DIR / "img" / "single_fragment" / "segmentation"

GENERATOR_MODELS = ("sd15", "sdxl", "sd3")
SEGMENTERS = ("deeplabv3plus", "sam2")
SEED = 2026
NUM_IMAGES = 10
DEFAULT_BASE_PROMPTS = 40
DEFAULT_MODIFIERS = 660
DEFAULT_MODIFIERS_WITH_CONTROL = 661

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
    profile: str
    prompt_index: int
    difficulty: str
    stress_factor: str
    text: str


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
    for group in SEGMENTATION_PROMPT_GROUPS:
        profile = group["profile"]
        for idx, item in enumerate(group["prompts"], start=1):
            yield BasePrompt(
                prompt_id=f"seg_{profile}_{idx:02d}",
                profile=profile,
                prompt_index=idx,
                difficulty=item["difficulty"],
                stress_factor=item["stress_factor"],
                text=item["prompt"],
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
            base.profile,
            str(base.prompt_index),
            base.difficulty,
            base.stress_factor,
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
        / safe_path_part(base.profile)
        / safe_path_part(base.prompt_id)
        / modifier_dir_name(modifier)
    )


def mask_dir_for(
    generator_model: str,
    segmenter_name: str,
    base: BasePrompt,
    modifier: Modifier,
) -> Path:
    return (
        OUTPUT_DIR
        / "masks"
        / safe_path_part(generator_model)
        / safe_path_part(segmenter_name)
        / safe_path_part(base.profile)
        / safe_path_part(base.prompt_id)
        / modifier_dir_name(modifier)
    )


def result_path(generator_model: str, segmenter_name: str) -> Path:
    return OUTPUT_DIR / f"{generator_model}_{segmenter_name}_segmentation.csv"


def summary_json_path(
    generator_model: str,
    segmenter_name: str,
    base: BasePrompt,
    modifier: Modifier,
) -> Path:
    return mask_dir_for(generator_model, segmenter_name, base, modifier) / "summary.json"


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


def confidence_columns() -> List[str]:
    return [f"confidence_{idx:02d}" for idx in range(NUM_IMAGES)]


def color_mask_columns() -> List[str]:
    return [f"color_mask_{idx:02d}" for idx in range(NUM_IMAGES)]


def detail_columns() -> List[str]:
    return [f"details_{idx:02d}" for idx in range(NUM_IMAGES)]


def result_fieldnames() -> List[str]:
    return [
        "combo_key",
        "generator_model",
        "segmenter",
        "prompt_id",
        "profile",
        "prompt_index",
        "difficulty",
        "stress_factor",
        "modifier_family",
        "modifier_category",
        "modifier_index",
        "modifier_text",
        "seed",
        "num_images",
        "base_prompt",
        "combined_prompt",
        "image_dir",
        "mask_dir",
        "confidence_mean",
        *confidence_columns(),
        *color_mask_columns(),
        *detail_columns(),
    ]


def open_result_writer(generator_model: str, segmenter_name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = result_path(generator_model, segmenter_name)
    exists = path.exists() and path.stat().st_size > 0
    f = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=result_fieldnames())
    if not exists:
        writer.writeheader()
        f.flush()
    return f, writer


def results_by_image_name(results: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {Path(item["image"]).name: dict(item) for item in results}


def row_for_segmentation(
    generator_model: str,
    segmenter_name: str,
    base: BasePrompt,
    modifier: Modifier,
    combined_prompt: str,
    image_dir: Path,
    mask_dir: Path,
    results: Sequence[Dict[str, Any]],
) -> Dict[str, object]:
    by_name = results_by_image_name(results)
    confidences: List[float] = []
    color_masks: List[str] = []
    details: List[str] = []

    for idx in range(NUM_IMAGES):
        image_name = f"{idx:03d}.png"
        item = by_name.get(image_name, {})
        confidence = float(item.get("confidence", 0.0))
        confidences.append(confidence)
        color_masks.append(display_path(Path(item["color_mask"])) if item.get("color_mask") else "")
        details.append(
            json.dumps(
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"image", "color_mask", "confidence"}
                },
                ensure_ascii=False,
            )
        )

    row: Dict[str, object] = {
        "combo_key": combo_key(base, modifier),
        "generator_model": generator_model,
        "segmenter": segmenter_name,
        "prompt_id": base.prompt_id,
        "profile": base.profile,
        "prompt_index": base.prompt_index,
        "difficulty": base.difficulty,
        "stress_factor": base.stress_factor,
        "modifier_family": modifier.family,
        "modifier_category": modifier.category,
        "modifier_index": modifier.index,
        "modifier_text": modifier.text,
        "seed": SEED,
        "num_images": NUM_IMAGES,
        "base_prompt": base.text,
        "combined_prompt": combined_prompt,
        "image_dir": display_path(image_dir),
        "mask_dir": display_path(mask_dir),
        "confidence_mean": round(sum(confidences) / NUM_IMAGES, 4),
    }
    for col, value in zip(confidence_columns(), confidences):
        row[col] = round(value, 4)
    for col, value in zip(color_mask_columns(), color_masks):
        row[col] = value
    for col, value in zip(detail_columns(), details):
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


def segmentation_progress_total(
    models: Sequence[str],
    segmenters: Sequence[str],
    base_prompts: Sequence[BasePrompt],
    modifiers: Sequence[Modifier],
) -> int:
    return len(models) * len(segmenters) * len(base_prompts) * len(modifiers)


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


def load_existing_summary(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return None


def color_masks_complete(results: Sequence[Dict[str, Any]]) -> bool:
    if len(results) != NUM_IMAGES:
        return False
    return all(item.get("color_mask") and Path(item["color_mask"]).is_file() for item in results)


def run_segmentation_pass(
    generator_model: str,
    segmenter_name: str,
    base_prompts: Sequence[BasePrompt],
    modifiers: Sequence[Modifier],
    limit: Optional[int],
    segmenter_device: str,
    deeplab_size: str,
    deeplab_dataset: str,
    deeplab_batch_size: int,
    sam2_size: str,
    sam2_max_masks: int,
    progress: tqdm,
) -> None:
    from tools_task import load_segmenter, segment_images

    with suppress_output():
        if segmenter_name == "deeplabv3plus":
            segmenter = load_segmenter(
                "deeplabv3plus",
                model_size=deeplab_size,
                dataset=deeplab_dataset,
                device=segmenter_device,
            )
        elif segmenter_name == "sam2":
            segmenter = load_segmenter(
                "sam2",
                model_size=sam2_size,
                device=segmenter_device,
            )
        else:
            raise ValueError(f"Unknown segmenter: {segmenter_name}")

    f, writer = open_result_writer(generator_model, segmenter_name)
    done_keys = load_done_keys(result_path(generator_model, segmenter_name))
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
                mask_dir = mask_dir_for(generator_model, segmenter_name, base, modifier)
                summary_path = summary_json_path(
                    generator_model,
                    segmenter_name,
                    base,
                    modifier,
                )
                results = load_existing_summary(summary_path)
                if results is None or not color_masks_complete(results):
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    with suppress_output():
                        if segmenter_name == "deeplabv3plus":
                            results = segment_images(
                                segmenter,
                                image_dir,
                                output_dir=mask_dir,
                                output_json=summary_path,
                                batch_size=deeplab_batch_size,
                            )
                        else:
                            results = segment_images(
                                segmenter,
                                image_dir,
                                output_dir=mask_dir,
                                output_json=summary_path,
                                max_masks=sam2_max_masks,
                            )

                writer.writerow(
                    row_for_segmentation(
                        generator_model=generator_model,
                        segmenter_name=segmenter_name,
                        base=base,
                        modifier=modifier,
                        combined_prompt=combined_prompt,
                        image_dir=image_dir,
                        mask_dir=mask_dir,
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
        del segmenter
        cleanup_memory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-fragment style modifier semantic segmentation experiments."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=GENERATOR_MODELS,
        default=list(GENERATOR_MODELS),
    )
    parser.add_argument(
        "--segmenters",
        nargs="+",
        choices=SEGMENTERS,
        default=list(SEGMENTERS),
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
    parser.add_argument("--segmenter-device", default="cuda:0")
    parser.add_argument("--deeplab-size", default="mobilenet")
    parser.add_argument("--deeplab-dataset", default="voc")
    parser.add_argument("--deeplab-batch-size", type=int, default=NUM_IMAGES)
    parser.add_argument("--sam2-size", default="tiny")
    parser.add_argument("--sam2-max-masks", type=int, default=20)
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
        help="Process at most N missing/new combinations per generator/segmenter for smoke tests.",
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
    if args.deeplab_batch_size <= 0:
        raise ValueError("--deeplab-batch-size must be > 0")
    if args.sam2_max_masks <= 0:
        raise ValueError("--sam2-max-masks must be > 0")

    base_prompts, modifiers = build_plan(
        include_control=not args.no_control,
        mechanism_top_k=args.mechanism_top_k,
        single_effect_top_k=args.single_effect_top_k,
        mixed_effect_top_k=args.mixed_effect_top_k,
    )

    with tqdm(
        total=generation_progress_total(args.models, base_prompts, modifiers),
        desc="single_fragment_segmentation_generation",
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
        total=segmentation_progress_total(
            args.models,
            args.segmenters,
            base_prompts,
            modifiers,
        ),
        desc="single_fragment_segmentation",
        unit="combo",
        dynamic_ncols=True,
    ) as segmentation_progress:
        for generator_model in args.models:
            for segmenter_name in args.segmenters:
                run_segmentation_pass(
                    generator_model=generator_model,
                    segmenter_name=segmenter_name,
                    base_prompts=base_prompts,
                    modifiers=modifiers,
                    limit=args.limit,
                    segmenter_device=args.segmenter_device,
                    deeplab_size=args.deeplab_size,
                    deeplab_dataset=args.deeplab_dataset,
                    deeplab_batch_size=args.deeplab_batch_size,
                    sam2_size=args.sam2_size,
                    sam2_max_masks=args.sam2_max_masks,
                    progress=segmentation_progress,
                )


if __name__ == "__main__":
    main()
