#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import shutil
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
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
from test_prompts_design import (
    PROMPT_FIELDS,
    SEX_PROMPT_TRIPLETS,
    VIOLENCE_PROMPT_TRIPLETS,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "experiment_outputs" / "modifier_results"/ "nsfw"
IMAGE_DIR = SCRIPT_DIR / "img" / "single_fragment"

GENERATOR_MODELS = ("sd15")
EXPERIMENT_STAGES = ("all")
CHECK_DETECTOR = "check"
SCORE_DETECTORS = ("clip_nsfw", "nudenet", "q16")
DETECTORS = ("clip_nsfw", "nudenet", "q16")
SEED = 2026
NUM_IMAGES = 10
SD3_IMAGE_BATCH_SIZE = 1
EXPECTED_COMBINED_PROMPTS = 175200
DEFAULT_REDUCED_BASE_PROMPTS = (
    len(SEX_PROMPT_TRIPLETS) + len(VIOLENCE_PROMPT_TRIPLETS)
) * len(PROMPT_FIELDS)
DEFAULT_REDUCED_MODIFIERS_WITH_CONTROL = 661
IMAGE_FORMATS = ("jpg", "png")
LOAD_IMAGE_FORMATS = ("jpg", "png")

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
    label: str
    prompt_id: str
    prompt_type: str
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


@dataclass(frozen=True)
class DetectionTask:
    base: BasePrompt
    modifier: Modifier
    key: str
    image_dir: Path
    check_image_dir: Path
    combined_prompt: str
    missing_detectors: Tuple[str, ...]
    needs_check_images: bool


def iter_base_prompts(
    sex_prompt_count: Optional[int] = None,
    violence_prompt_count: Optional[int] = None,
) -> Iterator[BasePrompt]:
    sex_items = SEX_PROMPT_TRIPLETS[:sex_prompt_count]
    violence_items = VIOLENCE_PROMPT_TRIPLETS[:violence_prompt_count]

    for item in sex_items:
        for prompt_type in PROMPT_FIELDS:
            yield BasePrompt("sex", item["id"], prompt_type, item[prompt_type])

    for item in violence_items:
        for prompt_type in PROMPT_FIELDS:
            yield BasePrompt("violence", item["id"], prompt_type, item[prompt_type])


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
            base.label,
            base.prompt_id,
            base.prompt_type,
            modifier.family,
            modifier.category,
            str(modifier.index),
        ]
    )


def safe_path_part(value: object) -> str:
    text = str(value) if value not in (None, "") else "none"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe[:120] or "none"


def image_dir_for(
    generator_model: str,
    image_type: str,
    base: BasePrompt,
    modifier: Modifier,
) -> Path:
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
        / safe_path_part(image_type)
        / safe_path_part(base.label)
        / safe_path_part(base.prompt_id)
        / safe_path_part(base.prompt_type)
        / modifier_dir
    )


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


@contextmanager
def suppress_safety_checker_warning():
    logger_names = (
        "diffusers.pipelines.stable_diffusion.safety_checker",
        "diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion",
    )
    previous_levels = {}
    try:
        for name in logger_names:
            logger = logging.getLogger(name)
            previous_levels[name] = logger.level
            logger.setLevel(logging.ERROR)
        yield
    finally:
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)


def score_columns() -> List[str]:
    return [f"score_{idx:02d}" for idx in range(NUM_IMAGES)]


def result_fieldnames() -> List[str]:
    return [
        "combo_key",
        "generator_model",
        "detector",
        "base_label",
        "prompt_id",
        "prompt_type",
        "modifier_family",
        "modifier_category",
        "modifier_index",
        "modifier_text",
        "seed",
        "num_images",
        "base_prompt",
        "combined_prompt",
        "image_dir",
        "passed_count",
        *score_columns(),
    ]


def result_path(generator_model: str, detector: str) -> Path:
    target_model = "clip" if detector == "clip_nsfw" else detector
    return OUTPUT_DIR / f"{generator_model}_{target_model}_nsfw.csv"


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


def ensure_csv_schema(path: Path, fieldnames: Sequence[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            current_header = next(reader)
        except StopIteration:
            return

    if current_header == list(fieldnames):
        return
    if not set(current_header).issubset(set(fieldnames)):
        raise ValueError(f"Existing result schema is incompatible: {path}")

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with path.open("r", encoding="utf-8", newline="") as src, tmp_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    os.replace(tmp_path, path)


def open_result_writer(generator_model: str, detector: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = result_path(generator_model, detector)
    ensure_csv_schema(path, result_fieldnames())
    exists = path.exists() and path.stat().st_size > 0
    f = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=result_fieldnames())
    if not exists:
        writer.writeheader()
        f.flush()
    return f, writer


def pil_images_to_numpy(images: Sequence[Image.Image]) -> np.ndarray:
    return np.stack(
        [np.asarray(image.convert("RGB")).astype("float32") / 255.0 for image in images],
        axis=0,
    )


def image_path_for(image_dir: Path, idx: int, image_format: str) -> Path:
    return image_dir / f"{idx:03d}.{image_format}"


def existing_image_path(image_dir: Path, idx: int) -> Optional[Path]:
    for image_format in LOAD_IMAGE_FORMATS:
        path = image_path_for(image_dir, idx, image_format)
        if path.is_file():
            return path
    return None


def image_set_complete(image_dir: Path) -> bool:
    return image_dir.is_dir() and all(
        existing_image_path(image_dir, idx) is not None for idx in range(NUM_IMAGES)
    )


def load_saved_images(image_dir: Path) -> List[Image.Image]:
    images = []
    for idx in range(NUM_IMAGES):
        image_path = existing_image_path(image_dir, idx)
        if image_path is None:
            raise FileNotFoundError(image_dir / f"{idx:03d}.{{jpg,png}}")
        with Image.open(image_path) as image:
            images.append(image.convert("RGB").copy())
    return images


def materialize_images(
    images: Sequence[Image.Image],
    image_dir: Path,
    image_format: str,
    jpeg_quality: int,
) -> None:
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images):
        save_path = image_path_for(image_dir, idx, image_format)
        rgb = image.convert("RGB")
        if image_format == "jpg":
            rgb.save(save_path, quality=jpeg_quality, subsampling=0)
        else:
            rgb.save(save_path, compress_level=1)


def is_black_image(image: Image.Image) -> bool:
    return np.asarray(image.convert("RGB")).max() <= 1


def count_passed_check_images(images: Sequence[Image.Image]) -> int:
    return sum(0 if is_black_image(image) else 1 for image in images)


def load_safety_checker(device: str):
    import torch
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    from transformers import CLIPImageProcessor

    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    with suppress_output():
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            safety_model_id
        ).to(device).eval()
        image_processor = CLIPImageProcessor.from_pretrained(safety_model_id)
    if device.startswith("cuda"):
        safety_checker = safety_checker.half()
    dtype = next(safety_checker.parameters()).dtype
    return safety_checker, image_processor, device, dtype


def apply_safety_checker(
    safety_checker: Any,
    image_processor: Any,
    device: str,
    dtype: Any,
    images: Sequence[Image.Image],
) -> List[Image.Image]:
    import torch

    image_array = pil_images_to_numpy(images)
    inputs = image_processor(images=list(images), return_tensors="pt")
    clip_input = inputs.pixel_values.to(device=device, dtype=dtype)

    with suppress_output(), suppress_safety_checker_warning(), torch.no_grad():
        checked, _ = safety_checker(
            images=image_array.copy(),
            clip_input=clip_input,
        )

    checked = np.asarray(checked)
    checked = np.clip(checked, 0.0, 1.0)
    return [
        Image.fromarray((image * 255).round().astype("uint8"))
        for image in checked
    ]


def cleanup_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_sd3_images_in_chunks(
    pipe: Any,
    prompt: str,
    height: int,
    width: int,
    image_batch_size: int,
) -> List[Image.Image]:
    import torch
    from tools_generator import make_prompt_batch_latents

    image_batch_size = max(1, min(image_batch_size, NUM_IMAGES))
    latents = make_prompt_batch_latents(
        pipe=pipe,
        prompt_count=1,
        num_images_per_prompt=NUM_IMAGES,
        height=height,
        width=width,
        seed=SEED,
    )

    images: List[Image.Image] = []
    for start in range(0, NUM_IMAGES, image_batch_size):
        chunk_latents = latents[start:start + image_batch_size]
        with torch.inference_mode():
            images.extend(
                pipe(
                    prompt=prompt,
                    negative_prompt="",
                    height=height,
                    width=width,
                    num_images_per_prompt=chunk_latents.shape[0],
                    latents=chunk_latents,
                    guidance_scale=7.0,
                    num_inference_steps=28,
                ).images
            )
    return images


def row_for_scores(
    generator_model: str,
    detector: str,
    base: BasePrompt,
    modifier: Modifier,
    combined_prompt: str,
    image_dir: Path,
    scores: Sequence[float],
) -> Dict[str, object]:
    if len(scores) != NUM_IMAGES:
        raise ValueError(f"Expected {NUM_IMAGES} scores, got {len(scores)}")

    row: Dict[str, object] = {
        "combo_key": combo_key(base, modifier),
        "generator_model": generator_model,
        "detector": detector,
        "base_label": base.label,
        "prompt_id": base.prompt_id,
        "prompt_type": base.prompt_type,
        "modifier_family": modifier.family,
        "modifier_category": modifier.category,
        "modifier_index": modifier.index,
        "modifier_text": modifier.text,
        "seed": SEED,
        "num_images": NUM_IMAGES,
        "base_prompt": base.text,
        "combined_prompt": combined_prompt,
        "image_dir": display_path(image_dir),
        "passed_count": "",
    }
    for col, score in zip(score_columns(), scores):
        row[col] = float(score)
    return row


def row_for_check_count(
    generator_model: str,
    base: BasePrompt,
    modifier: Modifier,
    combined_prompt: str,
    image_dir: Path,
    passed_count: int,
) -> Dict[str, object]:
    return {
        "combo_key": combo_key(base, modifier),
        "generator_model": generator_model,
        "detector": CHECK_DETECTOR,
        "base_label": base.label,
        "prompt_id": base.prompt_id,
        "prompt_type": base.prompt_type,
        "modifier_family": modifier.family,
        "modifier_category": modifier.category,
        "modifier_index": modifier.index,
        "modifier_text": modifier.text,
        "seed": SEED,
        "num_images": NUM_IMAGES,
        "base_prompt": base.text,
        "combined_prompt": combined_prompt,
        "image_dir": display_path(image_dir),
        "passed_count": passed_count,
    }


def build_plan(
    include_control: bool,
    plan: str,
    sex_prompt_count: int,
    violence_prompt_count: int,
    mechanism_top_k: int,
    single_effect_top_k: int,
    mixed_effect_top_k: int,
) -> Tuple[List[BasePrompt], List[Modifier]]:
    if plan == "full":
        base_prompts = list(iter_base_prompts())
        modifiers = list(iter_modifiers(include_control=include_control))
    elif plan == "reduced":
        base_prompts = list(
            iter_base_prompts(
                sex_prompt_count=sex_prompt_count,
                violence_prompt_count=violence_prompt_count,
            )
        )
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

    combined_count = len(base_prompts) * len(modifiers)

    if plan == "full" and include_control and combined_count != EXPECTED_COMBINED_PROMPTS:
        raise ValueError(
            f"Expected {EXPECTED_COMBINED_PROMPTS} combined prompts, got {combined_count}. "
            "Check style_modifier_design.py modifier counts."
        )
    if (
        plan == "reduced"
        and include_control
        and sex_prompt_count == len(SEX_PROMPT_TRIPLETS)
        and violence_prompt_count == len(VIOLENCE_PROMPT_TRIPLETS)
        and mechanism_top_k == 30
        and single_effect_top_k == 20
        and mixed_effect_top_k == 10
        and (
            len(base_prompts) != DEFAULT_REDUCED_BASE_PROMPTS
            or len(modifiers) != DEFAULT_REDUCED_MODIFIERS_WITH_CONTROL
        )
    ):
        raise ValueError(
            "Expected reduced plan to have "
            f"{DEFAULT_REDUCED_BASE_PROMPTS} prompts and "
            f"{DEFAULT_REDUCED_MODIFIERS_WITH_CONTROL} modifiers, got "
            f"{len(base_prompts)} prompts and {len(modifiers)} modifiers."
        )

    return base_prompts, modifiers


def active_detectors_for_label(label: str) -> Tuple[str, ...]:
    if label == "sex":
        return ("clip_nsfw", "nudenet", CHECK_DETECTOR)
    if label == "violence":
        return ("q16", CHECK_DETECTOR)
    raise ValueError(f"Unknown prompt label: {label}")


def active_score_detectors_for_label(label: str) -> Tuple[str, ...]:
    return tuple(
        detector
        for detector in active_detectors_for_label(label)
        if detector in SCORE_DETECTORS
    )


def prepare_done_keys(generator_model: str) -> Dict[str, set[str]]:
    return {
        detector: load_done_keys(result_path(generator_model, detector))
        for detector in DETECTORS
    }


def generation_kwargs(
    generator_model: str,
    combined_prompt: str,
    height: int,
    width: int,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "prompt": combined_prompt,
        "seed": SEED,
        "num_images_per_prompt": NUM_IMAGES,
        "height": height,
        "width": width,
    }
    return kwargs


def ensure_result_files(generator_model: str) -> None:
    for detector in DETECTORS:
        f, _ = open_result_writer(generator_model, detector)
        f.close()


def generation_progress_total(
    models: Sequence[str],
    base_prompts: Sequence[BasePrompt],
    modifiers: Sequence[Modifier],
) -> int:
    return len(models) * len(base_prompts) * len(modifiers)


def detection_progress_total(
    models: Sequence[str],
    base_prompts: Sequence[BasePrompt],
    modifiers: Sequence[Modifier],
    check_only: bool,
) -> int:
    if check_only:
        units = sum(
            1
            for base in base_prompts
            if CHECK_DETECTOR in active_detectors_for_label(base.label)
        ) * len(modifiers)
    else:
        units = len(base_prompts) * len(modifiers)
    return len(models) * units


def ensure_images_for_generator(
    generator_model: str,
    base_prompts: Sequence[BasePrompt],
    modifiers: Sequence[Modifier],
    limit: Optional[int],
    height: int,
    width: int,
    image_format: str,
    jpeg_quality: int,
    prompt_batch_size: int,
    sd3_image_batch_size: int,
    progress: tqdm,
) -> None:
    from tools_generator import get_images, load_generator_model

    with suppress_output():
        pipe = load_generator_model(generator_model, check=False)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    generated_count = 0

    def flush_generation_batch(tasks: List[ImageTask]) -> bool:
        nonlocal generated_count
        if not tasks:
            return False

        if generator_model == "sd3":
            for task in tasks:
                with suppress_output():
                    images = generate_sd3_images_in_chunks(
                        pipe=pipe,
                        prompt=task.combined_prompt,
                        height=height,
                        width=width,
                        image_batch_size=sd3_image_batch_size,
                    )
                if len(images) != NUM_IMAGES:
                    raise ValueError(
                        f"Expected {NUM_IMAGES} generated images, got {len(images)}"
                    )
                materialize_images(images, task.image_dir, image_format, jpeg_quality)
                generated_count += 1
                progress.update(1)
            return limit is not None and generated_count >= limit

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
            materialize_images(images, task.image_dir, image_format, jpeg_quality)
            generated_count += 1
            progress.update(1)

        return limit is not None and generated_count >= limit

    try:
        for base in base_prompts:
            pending_generation: List[ImageTask] = []
            for modifier in modifiers:
                image_dir = image_dir_for(generator_model, "raw", base, modifier)
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


def run_detection_pass(
    generator_model: str,
    base_prompts: Sequence[BasePrompt],
    modifiers: Sequence[Modifier],
    limit: Optional[int],
    safety_device: str,
    image_format: str,
    jpeg_quality: int,
    save_check_images: bool,
    check_only: bool,
    progress: tqdm,
) -> None:
    from tools_filter import get_prediction, load_detector_model

    files = {}
    writers = {}
    done_keys = prepare_done_keys(generator_model)
    loaded_detector_label: Optional[str] = None
    safety_checker_bundle = None
    processed_count = 0

    def ensure_safety_checker():
        nonlocal safety_checker_bundle
        if safety_checker_bundle is None:
            safety_checker_bundle = load_safety_checker(safety_device)
        return safety_checker_bundle

    def process_task(task: DetectionTask) -> None:
        nonlocal loaded_detector_label

        images = load_saved_images(task.image_dir)
        score_missing_detectors = [
            detector
            for detector in task.missing_detectors
            if detector in SCORE_DETECTORS
        ]
        if score_missing_detectors:
            if loaded_detector_label != task.base.label:
                with suppress_output():
                    load_detector_model(task.base.label, "both")
                loaded_detector_label = task.base.label
            image_array = pil_images_to_numpy(images)

            if task.base.label == "sex":
                with suppress_output():
                    prediction = (
                        get_prediction(
                            image_array,
                            "sex",
                            l=1,
                            method="both",
                            path=str(task.image_dir),
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                detector_scores = {
                    "clip_nsfw": prediction[0].reshape(-1).tolist(),
                    "nudenet": prediction[1].reshape(-1).tolist(),
                }
            else:
                with suppress_output():
                    prediction = (
                        get_prediction(
                            image_array,
                            "violence",
                            l=1,
                            method="both",
                            path=str(task.image_dir),
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                detector_scores = {"q16": prediction.reshape(-1).tolist()}

            for detector in score_missing_detectors:
                writers[detector].writerow(
                    row_for_scores(
                        generator_model=generator_model,
                        detector=detector,
                        base=task.base,
                        modifier=task.modifier,
                        combined_prompt=task.combined_prompt,
                        image_dir=task.image_dir,
                        scores=detector_scores[detector],
                    )
                )
                files[detector].flush()
                done_keys[detector].add(task.key)

        should_run_check = (
            CHECK_DETECTOR in active_detectors_for_label(task.base.label)
            and (
                CHECK_DETECTOR in task.missing_detectors
                or task.needs_check_images
            )
        )
        if should_run_check:
            safety_checker, image_processor, device, dtype = ensure_safety_checker()
            checked_images = apply_safety_checker(
                safety_checker,
                image_processor,
                device,
                dtype,
                images,
            )
            output_image_dir = task.image_dir
            if save_check_images:
                materialize_images(
                    checked_images,
                    task.check_image_dir,
                    image_format,
                    jpeg_quality,
                )
                output_image_dir = task.check_image_dir

            if CHECK_DETECTOR in task.missing_detectors:
                writers[CHECK_DETECTOR].writerow(
                    row_for_check_count(
                        generator_model=generator_model,
                        base=task.base,
                        modifier=task.modifier,
                        combined_prompt=task.combined_prompt,
                        image_dir=output_image_dir,
                        passed_count=count_passed_check_images(checked_images),
                    )
                )
                files[CHECK_DETECTOR].flush()
                done_keys[CHECK_DETECTOR].add(task.key)

    try:
        for detector in DETECTORS:
            files[detector], writers[detector] = open_result_writer(
                generator_model, detector
            )

        for base in base_prompts:
            active_detectors = active_detectors_for_label(base.label)
            if check_only:
                active_detectors = tuple(
                    detector
                    for detector in active_detectors
                    if detector == CHECK_DETECTOR
                )
            if not active_detectors:
                continue

            for modifier in modifiers:
                try:
                    key = combo_key(base, modifier)
                    raw_image_dir = image_dir_for(generator_model, "raw", base, modifier)
                    check_image_dir = image_dir_for(
                        generator_model,
                        CHECK_DETECTOR,
                        base,
                        modifier,
                    )
                    missing_detectors = tuple(
                        detector
                        for detector in active_detectors
                        if key not in done_keys[detector]
                    )
                    needs_check_images = (
                        CHECK_DETECTOR in active_detectors
                        and save_check_images
                        and not image_set_complete(check_image_dir)
                    )
                    if not missing_detectors and not needs_check_images:
                        continue
                    if not image_set_complete(raw_image_dir):
                        if limit is None:
                            raise FileNotFoundError(
                                f"Missing generated images: {raw_image_dir}"
                            )
                        continue

                    task = DetectionTask(
                        base=base,
                        modifier=modifier,
                        key=key,
                        image_dir=raw_image_dir,
                        check_image_dir=check_image_dir,
                        combined_prompt=compose_prompt(base.text, modifier.text),
                        missing_detectors=missing_detectors,
                        needs_check_images=needs_check_images,
                    )

                    process_task(task)
                    processed_count += 1
                    if limit is not None and processed_count >= limit:
                        return
                finally:
                    progress.update(1)

    finally:
        for f in files.values():
            f.close()
        if safety_checker_bundle is not None:
            del safety_checker_bundle
        cleanup_memory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-fragment style modifier image-safety experiments."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=GENERATOR_MODELS,
        default=["sd15"],
    )
    parser.add_argument(
        "--stage",
        choices=EXPERIMENT_STAGES,
        default="all",
        help="all generates/saves raw images then runs all detectors; check only reads existing raw images.",
    )
    parser.add_argument(
        "--plan",
        choices=("reduced", "full"),
        default="reduced",
        help="Default reduced uses 480 mechanism and 180 effect modifiers plus control across all NSFW templates.",
    )
    parser.add_argument(
        "--sex-prompt-count",
        type=int,
        default=len(SEX_PROMPT_TRIPLETS),
        help="Number of sex prompt IDs used by the reduced plan (default: all).",
    )
    parser.add_argument(
        "--violence-prompt-count",
        type=int,
        default=len(VIOLENCE_PROMPT_TRIPLETS),
        help="Number of violence prompt IDs used by the reduced plan (default: all).",
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
    parser.add_argument("--safety-device", default="cuda")
    parser.add_argument("--image-format", choices=IMAGE_FORMATS, default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=2,
        help="Number of combined prompts generated per diffusion call for non-SD3 models.",
    )
    parser.add_argument(
        "--sd3-image-batch-size",
        type=int,
        default=SD3_IMAGE_BATCH_SIZE,
        help="Number of SD3 images generated per diffusion call inside each prompt.",
    )
    parser.add_argument(
        "--save-check-images",
        action="store_true",
        help="Save safety-checker output images. Disabled by default for speed.",
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
        help="Process at most N missing/new combinations per generator model for smoke tests.",
    )
    return parser.parse_args()


def configure_runtime() -> None:
    try:
        import torch

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def main() -> None:
    os.chdir(SCRIPT_DIR)
    args = parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if args.sex_prompt_count <= 0:
        raise ValueError("--sex-prompt-count must be > 0")
    if args.violence_prompt_count <= 0:
        raise ValueError("--violence-prompt-count must be > 0")
    if args.mechanism_top_k <= 0:
        raise ValueError("--mechanism-top-k must be > 0")
    if args.single_effect_top_k <= 0:
        raise ValueError("--single-effect-top-k must be > 0")
    if args.mixed_effect_top_k <= 0:
        raise ValueError("--mixed-effect-top-k must be > 0")
    if args.prompt_batch_size <= 0:
        raise ValueError("--prompt-batch-size must be > 0")
    if args.sd3_image_batch_size <= 0:
        raise ValueError("--sd3-image-batch-size must be > 0")
    configure_runtime()
    base_prompts, modifiers = build_plan(
        include_control=not args.no_control,
        plan=args.plan,
        sex_prompt_count=args.sex_prompt_count,
        violence_prompt_count=args.violence_prompt_count,
        mechanism_top_k=args.mechanism_top_k,
        single_effect_top_k=args.single_effect_top_k,
        mixed_effect_top_k=args.mixed_effect_top_k,
    )

    if args.stage == "all":
        with tqdm(
            total=generation_progress_total(args.models, base_prompts, modifiers),
            desc="single_fragment_generation",
            unit="combo",
            dynamic_ncols=True,
        ) as generation_progress:
            for generator_model in args.models:
                ensure_result_files(generator_model)
                ensure_images_for_generator(
                    generator_model=generator_model,
                    base_prompts=base_prompts,
                    modifiers=modifiers,
                    limit=args.limit,
                    height=args.height,
                    width=args.width,
                    image_format=args.image_format,
                    jpeg_quality=args.jpeg_quality,
                    prompt_batch_size=args.prompt_batch_size,
                    sd3_image_batch_size=args.sd3_image_batch_size,
                    progress=generation_progress,
                )

        cleanup_memory()

        with tqdm(
            total=detection_progress_total(
                args.models,
                base_prompts,
                modifiers,
                check_only=False,
            ),
            desc="single_fragment_detection",
            unit="combo",
            dynamic_ncols=True,
        ) as detection_progress:
            for generator_model in args.models:
                ensure_result_files(generator_model)
                run_detection_pass(
                    generator_model=generator_model,
                    base_prompts=base_prompts,
                    modifiers=modifiers,
                    limit=args.limit,
                    safety_device=args.safety_device,
                    image_format=args.image_format,
                    jpeg_quality=args.jpeg_quality,
                    save_check_images=args.save_check_images,
                    check_only=False,
                    progress=detection_progress,
                )
    else:
        with tqdm(
            total=detection_progress_total(
                args.models,
                base_prompts,
                modifiers,
                check_only=True,
            ),
            desc="single_fragment_detection",
            unit="combo",
            dynamic_ncols=True,
        ) as detection_progress:
            for generator_model in args.models:
                ensure_result_files(generator_model)
                run_detection_pass(
                    generator_model=generator_model,
                    base_prompts=base_prompts,
                    modifiers=modifiers,
                    limit=args.limit,
                    safety_device=args.safety_device,
                    image_format=args.image_format,
                    jpeg_quality=args.jpeg_quality,
                    save_check_images=args.save_check_images,
                    check_only=True,
                    progress=detection_progress,
                )


if __name__ == "__main__":
    main()
