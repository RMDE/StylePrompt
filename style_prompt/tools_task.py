from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image
from tools_paths import PACKAGE_DIR, TOOLS_ROOT


PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_EXTRACTION_DIR = PROJECT_ROOT / "style_prompt" / "experiment_outputs" / "cvTask" / "extraction"
DEFAULT_SEGMENT_DIR = PROJECT_ROOT / "style_prompt" / "experiment_outputs" / "cvTask" / "segment"
DEFAULT_CLASSIFICATION_DIR = PROJECT_ROOT / "style_prompt" / "experiment_outputs" / "cvTask" / "classification"
DEFAULT_UNDERSTANDING_DIR = PROJECT_ROOT / "style_prompt" / "experiment_outputs" / "cvTask" / "understanding"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

YOLO11_WEIGHTS = {
    "n": TOOLS_ROOT / "Yolov11" / "yolo11n.pt",
    "m": TOOLS_ROOT / "Yolov11" / "yolo11m.pt",
    "x": TOOLS_ROOT / "Yolov11" / "yolo11x.pt",
}

DEIM_CONFIGS = {
    "n": TOOLS_ROOT / "DEIM" / "configs" / "deim_dfine" / "deim_hgnetv2_n_coco.yml",
    "s": TOOLS_ROOT / "DEIM" / "configs" / "deim_dfine" / "deim_hgnetv2_s_coco.yml",
    "m": TOOLS_ROOT / "DEIM" / "configs" / "deim_dfine" / "deim_hgnetv2_m_coco.yml",
    "l": TOOLS_ROOT / "DEIM" / "configs" / "deim_dfine" / "deim_hgnetv2_l_coco.yml",
    "x": TOOLS_ROOT / "DEIM" / "configs" / "deim_dfine" / "deim_hgnetv2_x_coco.yml",
}

DEIM_CHECKPOINTS = {
    "n": TOOLS_ROOT / "DEIM" / "checkpoints" / "deim_hgnetv2_n_coco.pth",
    "s": TOOLS_ROOT / "DEIM" / "checkpoints" / "deim_hgnetv2_s_coco.pth",
    "m": TOOLS_ROOT / "DEIM" / "checkpoints" / "deim_hgnetv2_m_coco.pth",
    "l": TOOLS_ROOT / "DEIM" / "checkpoints" / "deim_hgnetv2_l_coco.pth",
    "x": TOOLS_ROOT / "DEIM" / "checkpoints" / "deim_hgnetv2_x_coco.pth",
}

DEEPLAB_ROOT = TOOLS_ROOT / "DeepLabV3Plus-Pytorch"
SAM2_ROOT = TOOLS_ROOT / "SAM2"
CLASSIFICATION_ROOT = TOOLS_ROOT / "ImageClassification"
MAMBAVISION_ROOT = TOOLS_ROOT / "MambaVision"
CLIP_ROOT = TOOLS_ROOT / "CLIP"
LLAVA_ROOT = TOOLS_ROOT / "LLaVA-1.5"
QWEN3VL_ROOT = TOOLS_ROOT / "Qwen3-VL"

DEEPLAB_MODELS = {
    "mobilenet": "deeplabv3plus_mobilenet",
    "resnet50": "deeplabv3plus_resnet50",
    "resnet101": "deeplabv3plus_resnet101",
}

DEEPLAB_CHECKPOINTS = {
    "voc": {
        "mobilenet": DEEPLAB_ROOT / "checkpoints" / "best_deeplabv3plus_mobilenet_voc_os16.pth",
        "resnet50": DEEPLAB_ROOT / "checkpoints" / "best_deeplabv3plus_resnet50_voc_os16.pth",
        "resnet101": DEEPLAB_ROOT / "checkpoints" / "best_deeplabv3plus_resnet101_voc_os16.pth",
    },
    "cityscapes": {
        "mobilenet": DEEPLAB_ROOT / "checkpoints" / "best_deeplabv3plus_mobilenet_cityscapes_os16.pth",
    },
}

SAM2_CONFIGS = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

SAM2_CHECKPOINTS = {
    "tiny": SAM2_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt",
    "small": SAM2_ROOT / "checkpoints" / "sam2.1_hiera_small.pt",
    "base_plus": SAM2_ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt",
    "large": SAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt",
}

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Prompt templates use the VOC name "sofa", while COCO-based detectors such as
# YOLO11 and DEIM expose the same category as "couch".
DETECTION_CLASS_ALIASES = {
    "sofa": "couch",
}

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv monitor",
]

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]


@dataclass
class ObjectExtractor:
    name: str
    model: Any
    device: str
    input_size: int = 640


@dataclass
class Segmenter:
    name: str
    model: Any
    device: str
    dataset: str = ""
    class_names: List[str] = field(default_factory=list)
    input_size: Optional[int] = None


@dataclass
class ImageClassifier:
    name: str
    model: Any
    transform: Any
    class_names: List[str]
    device: str


@dataclass
class ClipModelWrapper:
    name: str
    model: Any
    processor: Any
    device: str
    dtype: Any = None


@dataclass
class VisualUnderstander:
    name: str
    model: Any
    processor: Any
    device: str
    dtype: Any = None


def load_object_extractor(
    model_type: str,
    model_size: str = "n",
    device: Optional[str] = None,
    model_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    input_size: int = 640,
) -> ObjectExtractor:
    model_type = _norm(model_type)
    if model_type in {"yolo11", "yolov11", "yolo"}:
        return _load_yolo11(model_size, device, model_path)
    if model_type == "deim":
        return _load_deim(model_size, device, config_path, checkpoint_path, input_size)
    raise ValueError("model_type must be 'yolo11' or 'deim'")


def extract_objects(
    extractor: ObjectExtractor,
    image_dir: Union[str, Path],
    target_classes: Union[str, Sequence[str]],
    conf: float = 0.25,
    batch_size: int = 8,
    output_json: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    image_paths = _list_images(image_dir)
    targets = _targets(target_classes)

    if extractor.name == "yolo11":
        results = _extract_yolo11(extractor, image_paths, targets, conf, batch_size)
    elif extractor.name == "deim":
        results = _extract_deim(extractor, image_paths, targets, conf, batch_size)
    else:
        raise ValueError(f"unsupported extractor: {extractor.name}")

    json_path = Path(output_json) if output_json else DEFAULT_EXTRACTION_DIR / f"{extractor.name}_boxes.json"
    _save_json(results, json_path)
    return results


def load_image_classifier(
    model_type: str,
    device: Optional[str] = None,
    model_name: str = "nvidia/MambaVision-T-1K",
) -> ImageClassifier:
    model_type = _norm(model_type)
    if model_type == "resnet50":
        return _load_resnet50_classifier(device)
    if model_type in {"mambavision", "mamba vision"}:
        return _load_mambavision_classifier(device, model_name)
    raise ValueError("model_type must be 'resnet50' or 'mambavision'")


def classify_images(
    classifier: ImageClassifier,
    image_dir: Union[str, Path],
    batch_size: int = 16,
    topk: int = 1,
    output_json: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if topk <= 0:
        raise ValueError("topk must be > 0")

    image_paths = _list_images(image_dir)
    results = _classify_images(classifier, image_paths, batch_size, topk)
    json_path = Path(output_json) if output_json else DEFAULT_CLASSIFICATION_DIR / f"{classifier.name}_predictions.json"
    _save_json(results, json_path)
    return results


def load_clip_model(
    model_name: str = "openai/clip-vit-base-patch32",
    device: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    local_files_only: Optional[bool] = None,
    revision: Optional[str] = None,
    use_safetensors: bool = True,
) -> ClipModelWrapper:
    return _load_clip_model(model_name, device, torch_dtype, local_files_only, revision, use_safetensors)


def clip_image_text_distance(
    clip_model: ClipModelWrapper,
    images: Any,
    texts: Any,
    batch_size: int = 64,
) -> Any:
    """
    Compute paired CLIP cosine distance between image tensor(s) and text(s).

    images accepts the list[PIL.Image.Image] returned by tools_generator.get_images().
    It also accepts [C,H,W], [H,W,C], [N,C,H,W], [N,H,W,C], or higher leading dims.
    texts can be a string, one text to broadcast, or nested texts matching image leading dims.
    The returned torch.Tensor has the same leading dims as images.
    """
    return _clip_image_text_distance(clip_model, images, texts, batch_size)


def load_visual_understander(
    model_type: str,
    device: Optional[str] = None,
    model_name: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    local_files_only: Optional[bool] = None,
) -> VisualUnderstander:
    model_type = _norm(model_type)
    if model_type in {"llava", "llava15", "llava 1.5", "llava-1.5"}:
        return _load_llava_understander(device, model_name or "llava-hf/llava-1.5-7b-hf", torch_dtype, local_files_only)
    if model_type in {"qwen3vl", "qwen3 vl", "qwen3-vl", "qwen vl"}:
        return _load_qwen3vl_understander(device, model_name or "Qwen/Qwen3-VL-2B-Instruct", torch_dtype, local_files_only)
    raise ValueError("model_type must be 'llava-1.5' or 'qwen3-vl'")


def get_visual_understanding(
    understander: VisualUnderstander,
    image_dir: Union[str, Path],
    questions: Any,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    output_json: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Return VLM answers with per-answer confidence.

    The confidence is the geometric mean of generated visible-token probabilities.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")

    image_paths = _list_images(image_dir)
    question_sets = _question_sets(questions, image_paths)
    results = _understand_images(understander, image_paths, question_sets, max_new_tokens, temperature, min_pixels, max_pixels)
    json_path = Path(output_json) if output_json else DEFAULT_UNDERSTANDING_DIR / f"{understander.name}_answers.json"
    _save_json(results, json_path)
    return results


def load_segmenter(
    model_type: str,
    model_size: str = "mobilenet",
    device: Optional[str] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    dataset: str = "voc",
    input_size: Optional[int] = 513,
    output_stride: int = 16,
) -> Segmenter:
    model_type = _norm(model_type)
    if model_type in {"deeplab", "deeplabv3", "deeplabv3plus", "deeplabv3+"}:
        return _load_deeplab(model_size, device, checkpoint_path, dataset, input_size, output_stride)
    if model_type in {"sam", "sam2"}:
        return _load_sam2(model_size, device, checkpoint_path, config_path)
    raise ValueError("model_type must be 'deeplabv3plus' or 'sam2'")


def segment_images(
    segmenter: Segmenter,
    image_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    output_json: Optional[Union[str, Path]] = None,
    batch_size: int = 4,
    max_masks: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Segment all images in image_dir.

    Output rule:
    - DeepLabV3+: only save one color semantic mask for each image.
    - SAM2: only save one merged color instance mask for each image.
    - Default output root: experiment_outputs/cvTask/segment.
    """
    image_paths = _list_images(image_dir)
    out_dir = Path(output_dir) if output_dir else DEFAULT_SEGMENT_DIR / segmenter.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if segmenter.name == "deeplabv3plus":
        results = _segment_deeplab(segmenter, image_paths, out_dir, batch_size)
    elif segmenter.name == "sam2":
        results = _segment_sam2(segmenter, image_paths, out_dir, max_masks)
    else:
        raise ValueError(f"unsupported segmenter: {segmenter.name}")

    json_path = Path(output_json) if output_json else DEFAULT_SEGMENT_DIR / f"{segmenter.name}_masks.json"
    _save_json(results, json_path)
    return results


# =========================
# Load object extractors
# =========================

def _load_yolo11(model_size: str, device: Optional[str], model_path: Optional[Union[str, Path]]) -> ObjectExtractor:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
    from ultralytics import YOLO

    weight = _must_exist(model_path or YOLO11_WEIGHTS[model_size])
    return ObjectExtractor("yolo11", YOLO(str(weight)), _device(device))


def _load_deim(
    model_size: str,
    device: Optional[str],
    config_path: Optional[Union[str, Path]],
    checkpoint_path: Optional[Union[str, Path]],
    input_size: int,
) -> ObjectExtractor:
    import torch
    import torch.nn as nn

    deim_root = _must_exist(TOOLS_ROOT / "DEIM")
    if str(deim_root) not in sys.path:
        sys.path.insert(0, str(deim_root))

    from engine.core import YAMLConfig

    config = _must_exist(config_path or DEIM_CONFIGS[model_size])
    checkpoint = _must_exist(checkpoint_path or DEIM_CHECKPOINTS[model_size])
    device_name = _device(device)

    cfg = YAMLConfig(str(config), resume=str(checkpoint))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.model.load_state_dict(_torch_state(torch, checkpoint))

    class DEIMWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images: Any, sizes: Any) -> Any:
            return self.postprocessor(self.model(images), sizes)

    model = DEIMWrapper().to(device_name).eval()
    return ObjectExtractor("deim", model, device_name, input_size)


# =========================
# Load segmenters
# =========================

def _load_deeplab(
    model_size: str,
    device: Optional[str],
    checkpoint_path: Optional[Union[str, Path]],
    dataset: str,
    input_size: Optional[int],
    output_stride: int,
) -> Segmenter:
    import importlib
    import torch

    dataset = dataset.lower()
    class_names = VOC_CLASSES if dataset == "voc" else CITYSCAPES_CLASSES
    model_name = DEEPLAB_MODELS[model_size]
    checkpoint = _must_exist(checkpoint_path or DEEPLAB_CHECKPOINTS[dataset][model_size])
    device_name = _device(device)

    deeplab_root = _must_exist(DEEPLAB_ROOT)
    if str(deeplab_root) not in sys.path:
        sys.path.insert(0, str(deeplab_root))

    network = importlib.import_module("network")
    model = getattr(network, model_name)(
        num_classes=len(class_names),
        output_stride=output_stride,
        pretrained_backbone=False,
    )
    model.load_state_dict(_strip_module_prefix(_torch_state(torch, checkpoint)), strict=False)
    model = model.to(device_name).eval()
    return Segmenter("deeplabv3plus", model, device_name, dataset, class_names, input_size)


def _load_sam2(
    model_size: str,
    device: Optional[str],
    checkpoint_path: Optional[Union[str, Path]],
    config_path: Optional[Union[str, Path]],
) -> Segmenter:
    sam_root = _must_exist(SAM2_ROOT)
    if str(sam_root) not in sys.path:
        sys.path.insert(0, str(sam_root))

    from sam2.build_sam import build_sam2

    model_size = _sam2_size(model_size)
    config = str(config_path or SAM2_CONFIGS[model_size])
    checkpoint = _must_exist(checkpoint_path or SAM2_CHECKPOINTS[model_size])
    device_name = _device(device)

    model = build_sam2(config, str(checkpoint), device=device_name, apply_postprocessing=True)
    return Segmenter("sam2", model, device_name)


# =========================
# Load classifiers
# =========================

def _load_resnet50_classifier(device: Optional[str]) -> ImageClassifier:
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    os.environ.setdefault("TORCH_HOME", str(CLASSIFICATION_ROOT))
    weights = ResNet50_Weights.DEFAULT
    device_name = _device(device)
    model = resnet50(weights=weights).to(device_name).eval()
    return ImageClassifier("resnet50", model, weights.transforms(), weights.meta["categories"], device_name)


def _load_mambavision_classifier(device: Optional[str], model_name: str) -> ImageClassifier:
    _clear_proxy_env()
    _set_hf_cache_env(MAMBAVISION_ROOT)

    from timm.data.transforms_factory import create_transform
    from transformers import AutoModelForImageClassification

    MAMBAVISION_ROOT.mkdir(parents=True, exist_ok=True)
    device_name = _device(device)
    source, local_only = _hf_source(model_name, MAMBAVISION_ROOT, None)
    model = AutoModelForImageClassification.from_pretrained(
        source,
        trust_remote_code=True,
        cache_dir=str(MAMBAVISION_ROOT),
        local_files_only=local_only,
    ).to(device_name).eval()
    cfg = model.config
    transform = create_transform(
        input_size=(3, 224, 224),
        is_training=False,
        mean=cfg.mean,
        std=cfg.std,
        crop_mode=cfg.crop_mode,
        crop_pct=cfg.crop_pct,
    )
    class_names = [cfg.id2label[i] for i in range(len(cfg.id2label))]
    return ImageClassifier("mambavision", model, transform, class_names, device_name)


# =========================
# Load CLIP
# =========================

def _load_clip_model(
    model_name: str,
    device: Optional[str],
    torch_dtype: Optional[str],
    local_files_only: Optional[bool],
    revision: Optional[str],
    use_safetensors: bool,
) -> ClipModelWrapper:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    CLIP_ROOT.mkdir(parents=True, exist_ok=True)
    device_name = _device(device)
    source, local_only = _hf_source(model_name, CLIP_ROOT, local_files_only)
    load_dtype = _torch_dtype(torch, torch_dtype, device_name)

    processor_kwargs: Dict[str, Any] = {
        "cache_dir": str(CLIP_ROOT),
        "local_files_only": local_only,
        "use_fast": False,
    }
    model_kwargs: Dict[str, Any] = {
        "cache_dir": str(CLIP_ROOT),
        "local_files_only": local_only,
    }
    if revision and source == model_name:
        processor_kwargs["revision"] = revision
        model_kwargs["revision"] = revision
    if use_safetensors:
        model_kwargs["use_safetensors"] = True
    if load_dtype != "auto":
        model_kwargs["torch_dtype"] = load_dtype

    processor = CLIPProcessor.from_pretrained(source, **processor_kwargs)
    model = CLIPModel.from_pretrained(source, **model_kwargs).to(device_name).eval()
    return ClipModelWrapper("clip", model, processor, device_name, _model_float_dtype(model))


# =========================
# Load visual understanders
# =========================

def _load_llava_understander(
    device: Optional[str],
    model_name: str,
    torch_dtype: Optional[str],
    local_files_only: Optional[bool],
) -> VisualUnderstander:
    _clear_proxy_env()
    _set_hf_cache_env(LLAVA_ROOT)

    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    LLAVA_ROOT.mkdir(parents=True, exist_ok=True)
    device_name = _device(device)
    load_dtype = _torch_dtype(torch, torch_dtype, device_name)
    source, local_only = _hf_source(model_name, LLAVA_ROOT, local_files_only)
    kwargs = {
        "cache_dir": str(LLAVA_ROOT),
        "low_cpu_mem_usage": True,
        "local_files_only": local_only,
    }
    if load_dtype is not None:
        kwargs["torch_dtype"] = load_dtype

    processor = AutoProcessor.from_pretrained(source, cache_dir=str(LLAVA_ROOT), local_files_only=local_only)
    model = LlavaForConditionalGeneration.from_pretrained(source, **kwargs).to(device_name).eval()
    return VisualUnderstander("llava15", model, processor, device_name, _model_float_dtype(model))


def _load_qwen3vl_understander(
    device: Optional[str],
    model_name: str,
    torch_dtype: Optional[str],
    local_files_only: Optional[bool],
) -> VisualUnderstander:
    _clear_proxy_env()
    _set_hf_cache_env(QWEN3VL_ROOT)

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise ImportError("Qwen3-VL requires transformers>=4.57.0 in the cvTask environment.") from exc

    QWEN3VL_ROOT.mkdir(parents=True, exist_ok=True)
    device_name = _device(device)
    source, local_only = _hf_source(model_name, QWEN3VL_ROOT, local_files_only)
    kwargs = {
        "cache_dir": str(QWEN3VL_ROOT),
        "local_files_only": local_only,
        "trust_remote_code": True,
    }
    if torch_dtype:
        kwargs["dtype"] = torch_dtype
    else:
        kwargs["dtype"] = "auto"

    processor = AutoProcessor.from_pretrained(source, cache_dir=str(QWEN3VL_ROOT), local_files_only=local_only, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(source, **kwargs).to(device_name).eval()
    return VisualUnderstander("qwen3vl", model, processor, device_name, _model_float_dtype(model))


# =========================
# Object extraction backends
# =========================

def _extract_yolo11(
    extractor: ObjectExtractor,
    image_paths: Sequence[Path],
    targets: Sequence[str],
    conf: float,
    batch_size: int,
) -> List[Dict[str, Any]]:
    names = {int(k): _norm(v) for k, v in extractor.model.names.items()}
    available_names = set(names.values())
    target_by_model_name = {}
    missing = []
    for target in targets:
        # 不同权重可能采用 VOC 的 ``sofa`` 或 COCO 的 ``couch``。
        # 优先使用模型中真实存在的原始名称，再尝试配置的类别别名。
        candidates = (target, DETECTION_CLASS_ALIASES.get(target, target))
        model_name = next(
            (candidate for candidate in candidates if candidate in available_names),
            None,
        )
        if model_name is None:
            missing.append(target)
        else:
            target_by_model_name[model_name] = target
    class_ids = [i for i, name in names.items() if name in target_by_model_name]
    if missing:
        raise ValueError(
            f"target classes not found in YOLO names: {missing}; "
            f"available classes: {sorted(available_names)}"
        )

    results: List[Dict[str, Any]] = []
    for batch in _chunks(image_paths, batch_size):
        preds = extractor.model.predict(
            source=[str(p) for p in batch],
            conf=conf,
            classes=class_ids,
            device=extractor.device,
            verbose=False,
        )
        for path, pred in zip(batch, preds):
            objects = []
            for b in ([] if pred.boxes is None else pred.boxes):
                objects.append({
                    "class": target_by_model_name[names[int(b.cls.item())]],
                    "box": _round_list(b.xyxy[0].detach().cpu().tolist()),
                    "confidence": round(float(b.conf.item()), 4),
                })
            results.append({"image": str(path), "objects": objects})
    return results


def _extract_deim(
    extractor: ObjectExtractor,
    image_paths: Sequence[Path],
    targets: Sequence[str],
    conf: float,
    batch_size: int,
) -> List[Dict[str, Any]]:
    import torch
    import torchvision.transforms as T

    preprocess = T.Compose([T.Resize((extractor.input_size, extractor.input_size)), T.ToTensor()])
    target_by_model_name = {
        DETECTION_CLASS_ALIASES.get(target, target): target
        for target in targets
    }
    results: List[Dict[str, Any]] = []

    for batch in _chunks(image_paths, batch_size):
        images = [Image.open(p).convert("RGB") for p in batch]
        tensors = torch.stack([preprocess(img) for img in images]).to(extractor.device)
        sizes = torch.tensor([[img.width, img.height] for img in images], device=extractor.device)

        with torch.no_grad():
            labels, boxes, scores = extractor.model(tensors, sizes)

        for i, path in enumerate(batch):
            objects = []
            for idx in scores[i].argsort(descending=True):
                score = float(scores[i][idx].item())
                if score < conf:
                    break
                label = COCO_NAMES[int(labels[i][idx].item())]
                model_name = _norm(label)
                if model_name not in target_by_model_name:
                    continue
                objects.append({
                    "class": target_by_model_name[model_name],
                    "box": _round_list(boxes[i][idx].detach().cpu().tolist()),
                    "confidence": round(score, 4),
                })
            results.append({"image": str(path), "objects": objects})
    return results


# =========================
# Segmentation backends
# =========================

def _segment_deeplab(
    segmenter: Segmenter,
    image_paths: Sequence[Path],
    output_dir: Path,
    batch_size: int,
) -> List[Dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T

    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess = T.Compose([
        T.Resize((segmenter.input_size, segmenter.input_size)) if segmenter.input_size else T.Lambda(lambda x: x),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    results: List[Dict[str, Any]] = []
    for batch in _chunks(image_paths, batch_size):
        images = [Image.open(p).convert("RGB") for p in batch]
        original_sizes = [img.size for img in images]
        tensors = torch.stack([preprocess(img) for img in images]).to(segmenter.device)

        with torch.no_grad():
            logits = _as_logits(segmenter.model(tensors))

        for idx, (path, size) in enumerate(zip(batch, original_sizes)):
            # Resize logits back to original image size before argmax.
            one_logits = F.interpolate(
                logits[idx:idx + 1],
                size=(size[1], size[0]),
                mode="bilinear",
                align_corners=False,
            )
            probs = F.softmax(one_logits, dim=1)
            conf_map, cls_map = probs.max(dim=1)
            topk_probs = probs.topk(k=min(2, probs.shape[1]), dim=1).values
            if topk_probs.shape[1] > 1:
                stability_map = topk_probs[:, 0] - topk_probs[:, 1]
            else:
                stability_map = topk_probs[:, 0]

            cls_map_np = cls_map[0].detach().cpu().numpy().astype("uint8")
            conf_map_np = conf_map[0].detach().cpu().numpy()
            stability_map_np = stability_map[0].detach().cpu().numpy()
            segment_mask_np = _semantic_segment_mask(cls_map_np, segmenter.class_names)

            stem = _safe_stem(path)
            color_mask_path = output_dir / f"{stem}_{segmenter.name}_color.png"
            Image.fromarray(_decode_deeplab_mask(cls_map_np, segmenter.dataset)).save(color_mask_path)

            results.append({
                "image": str(path),
                "color_mask": str(color_mask_path),
                "confidence": _masked_mean(conf_map_np, segment_mask_np),
                "stability_score": _masked_mean(stability_map_np, segment_mask_np),
                "unique_class_ids": [int(v) for v in sorted(set(cls_map_np.reshape(-1).tolist()))],
                "classes": _class_summary(cls_map_np, conf_map_np, segmenter.class_names, stability_map_np),
                "note": (
                    "DeepLab does not output SAM-style predicted IoU. Confidence is the average "
                    "top-class probability over predicted non-background semantic mask pixels; "
                    "stability_score is the average top-1/top-2 probability margin over those pixels."
                ),
            })
    return results


def _segment_sam2(
    segmenter: Segmenter,
    image_paths: Sequence[Path],
    output_dir: Path,
    max_masks: Optional[int],
) -> List[Dict[str, Any]]:
    import numpy as np
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    output_dir.mkdir(parents=True, exist_ok=True)

    generator = SAM2AutomaticMaskGenerator(segmenter.model, output_mode="binary_mask")
    results: List[Dict[str, Any]] = []

    for path in image_paths:
        image = np.array(Image.open(path).convert("RGB"))
        h, w = image.shape[:2]

        with torch.inference_mode():
            anns = generator.generate(image)

        anns = sorted(anns, key=lambda x: float(x.get("predicted_iou", 0.0)), reverse=True)
        if max_masks is not None:
            anns = anns[:max_masks]

        merged_map = np.zeros((h, w), dtype=np.uint16)
        merged_conf = np.zeros((h, w), dtype=np.float32)
        mask_infos = []

        for instance_id, ann in enumerate(anns, start=1):
            mask = np.asarray(ann["segmentation"], dtype=bool)
            score = float(ann.get("predicted_iou", 0.0))

            update = mask & (score >= merged_conf)
            merged_map[update] = instance_id
            merged_conf[update] = score

            mask_infos.append({
                "instance_id": instance_id,
                "box": _xywh_to_xyxy(ann["bbox"]),
                "area": int(ann["area"]),
                "ratio": round(float(ann["area"]) / float(h * w), 6),
                "confidence": round(score, 4),
                "stability_score": round(float(ann.get("stability_score", 0.0)), 4),
            })

        color_mask_path = output_dir / f"{_safe_stem(path)}_{segmenter.name}_color.png"
        Image.fromarray(_colorize_instance_map(merged_map)).save(color_mask_path)

        total_conf = float(merged_conf[merged_map > 0].mean()) if (merged_map > 0).any() else 0.0
        results.append({
            "image": str(path),
            "color_mask": str(color_mask_path),
            "confidence": round(total_conf, 4),
            "num_masks": len(mask_infos),
            "note": "SAM2 does not output semantic classes. Different colors indicate different segmented regions/instances.",
            "masks": mask_infos,
        })
    return results


# =========================
# Classification backends
# =========================

def _classify_images(
    classifier: ImageClassifier,
    image_paths: Sequence[Path],
    batch_size: int,
    topk: int,
) -> List[Dict[str, Any]]:
    import torch

    if classifier.name == "mambavision" and not classifier.device.startswith("cuda"):
        raise RuntimeError("MambaVision requires a CUDA device because mamba-ssm selective_scan does not run on CPU.")

    results: List[Dict[str, Any]] = []
    topk = min(topk, len(classifier.class_names))

    for batch in _chunks(image_paths, batch_size):
        images = [Image.open(path).convert("RGB") for path in batch]
        inputs = torch.stack([classifier.transform(image) for image in images]).to(classifier.device)

        with torch.no_grad():
            logits = _as_logits(classifier.model(inputs))
            probs = torch.softmax(logits, dim=1)
            scores, indices = probs.topk(topk, dim=1)

        for path, image_scores, image_indices in zip(batch, scores, indices):
            predictions = [
                {
                    "class": classifier.class_names[int(idx)],
                    "confidence": round(float(score), 4),
                }
                for score, idx in zip(image_scores.detach().cpu(), image_indices.detach().cpu())
            ]
            results.append({
                "image": str(path),
                "class": predictions[0]["class"],
                "confidence": predictions[0]["confidence"],
                "predictions": predictions,
            })
    return results


# =========================
# Visual understanding backends
# =========================

def _understand_images(
    understander: VisualUnderstander,
    image_paths: Sequence[Path],
    question_sets: Sequence[Sequence[str]],
    max_new_tokens: int,
    temperature: float,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for path, questions in zip(image_paths, question_sets):
        image = Image.open(path).convert("RGB") if understander.name == "llava15" else None
        answers = []
        try:
            for question in questions:
                if understander.name == "llava15":
                    answer, confidence = _ask_llava(understander, image, question, max_new_tokens, temperature)
                elif understander.name == "qwen3vl":
                    answer, confidence = _ask_qwen3vl(understander, path, question, max_new_tokens, temperature, min_pixels, max_pixels)
                else:
                    raise ValueError(f"unsupported understander: {understander.name}")
                answers.append({"question": question, "answer": answer, "confidence": confidence})
        finally:
            if image is not None:
                image.close()
        results.append({"image": str(path), "answers": answers})
    return results


def _ask_llava(
    understander: VisualUnderstander,
    image: Image.Image,
    question: str,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, float]:
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image"},
        ],
    }]
    prompt = understander.processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = understander.processor(images=image, text=prompt, return_tensors="pt")
    inputs = _inputs_to_device(inputs, understander.device, understander.dtype)
    return _generate_answer_with_confidence(understander, inputs, max_new_tokens, temperature)


def _ask_qwen3vl(
    understander: VisualUnderstander,
    image_path: Path,
    question: str,
    max_new_tokens: int,
    temperature: float,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
) -> tuple[str, float]:
    image_item: Dict[str, Any] = {"type": "image", "image": str(image_path)}
    if min_pixels is not None:
        image_item["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        image_item["max_pixels"] = int(max_pixels)

    messages = [{
        "role": "user",
        "content": [
            image_item,
            {"type": "text", "text": question},
        ],
    }]
    inputs = understander.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = _inputs_to_device(inputs, understander.device, understander.dtype)
    return _generate_answer_with_confidence(understander, inputs, max_new_tokens, temperature)


def _generate_answer_with_confidence(
    understander: VisualUnderstander,
    inputs: Any,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, float]:
    import torch

    generation_kwargs = _generation_kwargs(max_new_tokens, temperature)
    generation_kwargs.update({
        "return_dict_in_generate": True,
        "output_scores": True,
    })

    with torch.inference_mode():
        output = understander.model.generate(**inputs, **generation_kwargs)

    sequences = getattr(output, "sequences", output)
    answer = _decode_generated(understander.processor, sequences, inputs["input_ids"])
    confidence = _generated_confidence(understander.model, understander.processor, output, inputs["input_ids"])
    return answer, confidence


def _generated_confidence(model: Any, processor: Any, output: Any, input_ids: Any) -> float:
    import torch

    scores = getattr(output, "scores", None)
    sequences = getattr(output, "sequences", output)
    if not scores:
        return 0.0

    input_len = int(input_ids.shape[-1])
    generated_ids = sequences[:, input_len:input_len + len(scores)]
    if generated_ids.numel() == 0:
        return 0.0

    special_ids = _special_token_ids(processor)
    if hasattr(model, "compute_transition_scores"):
        beam_indices = getattr(output, "beam_indices", None)
        transition_scores = model.compute_transition_scores(
            sequences,
            scores,
            beam_indices=beam_indices,
            normalize_logits=True,
        )
        mask = _visible_token_mask(generated_ids, special_ids)
        selected = transition_scores[:, :generated_ids.shape[1]][mask]
        if selected.numel() > 0:
            confidence = selected.float().mean().exp()
            return round(float(confidence.detach().cpu().item()), 4)

    token_logprobs = []
    for step, step_scores in enumerate(scores[:generated_ids.shape[1]]):
        token_ids = generated_ids[:, step]
        logprobs = torch.log_softmax(step_scores.float(), dim=-1)
        selected = logprobs.gather(1, token_ids.unsqueeze(-1)).squeeze(-1)

        if special_ids:
            selected = selected[_visible_token_mask(token_ids, special_ids)]

        if selected.numel() > 0:
            token_logprobs.append(selected)

    if not token_logprobs:
        return 0.0

    confidence = torch.cat([values.reshape(-1) for values in token_logprobs]).mean().exp()
    return round(float(confidence.detach().cpu().item()), 4)


def _visible_token_mask(token_ids: Any, special_ids: set[int]) -> Any:
    import torch

    keep = torch.ones_like(token_ids, dtype=torch.bool)
    for special_id in special_ids:
        keep &= token_ids != special_id
    return keep


def _special_token_ids(processor: Any) -> set[int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    ids = set()
    for attr in ("all_special_ids", "bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is None:
            continue
        if isinstance(value, int):
            ids.add(value)
        else:
            try:
                ids.update(int(item) for item in value if item is not None)
            except TypeError:
                continue
    return ids



# =========================
# Small helpers
# =========================

def _clip_image_text_distance(
    clip_model: ClipModelWrapper,
    images: Any,
    texts: Any,
    batch_size: int,
) -> Any:
    import torch
    import torch.nn.functional as F

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    flat_images, leading_shape, channel_first = _flatten_clip_images(images)
    flat_texts = _flatten_clip_texts(texts, leading_shape)
    if len(flat_images) != len(flat_texts):
        raise ValueError("images and texts must contain the same number of items after flattening")

    distances = []
    for start in range(0, len(flat_texts), batch_size):
        end = min(start + batch_size, len(flat_texts))
        image_batch = [_to_clip_pil_image(img, channel_first) for img in flat_images[start:end]]
        text_batch = flat_texts[start:end]

        image_inputs = clip_model.processor(images=image_batch, return_tensors="pt")
        text_inputs = clip_model.processor(
            text=text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        image_inputs = _inputs_to_device(image_inputs, clip_model.device, clip_model.dtype)
        text_inputs = _inputs_to_device(text_inputs, clip_model.device)

        with torch.no_grad():
            image_features = clip_model.model.get_image_features(**image_inputs)
            text_features = clip_model.model.get_text_features(**text_inputs)
            image_features = F.normalize(image_features.float(), dim=-1)
            text_features = F.normalize(text_features.float(), dim=-1)
            distances.append(1.0 - (image_features * text_features).sum(dim=-1))

    return torch.cat(distances, dim=0).reshape(leading_shape).cpu()


def _flatten_clip_images(images: Any) -> tuple[Any, tuple[int, ...], Optional[bool]]:
    import torch

    if isinstance(images, Image.Image):
        return [images], (), None

    if isinstance(images, (list, tuple)) and _contains_pil_image(images):
        flat: List[Image.Image] = []
        leading_shape = _flatten_pil_images(images, flat)
        return flat, leading_shape, None

    tensor = images if isinstance(images, torch.Tensor) else torch.as_tensor(images)
    if tensor.ndim < 3:
        raise ValueError("images must have at least 3 dimensions")

    shape = tuple(int(v) for v in tensor.shape)
    if shape[-3] in {1, 3, 4}:
        leading_shape = shape[:-3]
        image_shape = shape[-3:]
        channel_first = True
    elif shape[-1] in {1, 3, 4}:
        leading_shape = shape[:-3]
        image_shape = shape[-3:]
        channel_first = False
    else:
        raise ValueError("images must be channel-first [...,C,H,W] or channel-last [...,H,W,C]")

    return tensor.reshape((-1, *image_shape)), leading_shape, channel_first


def _contains_pil_image(value: Any) -> bool:
    if isinstance(value, Image.Image):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_pil_image(item) for item in value)
    return False


def _flatten_pil_images(value: Any, out: List[Image.Image]) -> tuple[int, ...]:
    if isinstance(value, Image.Image):
        out.append(value)
        return ()

    if not isinstance(value, (list, tuple)):
        raise TypeError("PIL image inputs must be an image or a nested list/tuple of images")
    if not value:
        raise ValueError("image list must not be empty")

    child_shapes = [_flatten_pil_images(item, out) for item in value]
    first_shape = child_shapes[0]
    if any(shape != first_shape for shape in child_shapes):
        raise ValueError("nested image lists must have a rectangular shape")
    return (len(value), *first_shape)


def _flatten_clip_texts(texts: Any, leading_shape: Sequence[int]) -> List[str]:
    count = _shape_count(leading_shape)
    if isinstance(texts, str):
        return [texts] * count

    flat: List[str] = []
    _collect_texts(texts, flat)
    if len(flat) == 1 and count > 1:
        return flat * count
    if len(flat) != count:
        raise ValueError(f"texts contains {len(flat)} item(s), but images require {count}")
    return flat


def _collect_texts(value: Any, out: List[str]) -> None:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("text must not be empty")
        out.append(text)
        return
    try:
        items = list(value)
    except TypeError as exc:
        raise TypeError("texts must be a string or a nested sequence of strings") from exc
    for item in items:
        _collect_texts(item, out)


def _shape_count(shape: Sequence[int]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


def _to_clip_pil_image(image: Any, channel_first: Optional[bool]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    import torch

    tensor = image.detach().cpu()
    if channel_first:
        tensor = tensor.permute(1, 2, 0)

    channels = int(tensor.shape[-1])
    if channels == 1:
        tensor = tensor.repeat(1, 1, 3)
    elif channels == 4:
        tensor = tensor[..., :3]
    elif channels != 3:
        raise ValueError("CLIP image tensors must have 1, 3, or 4 channels")

    if tensor.is_floating_point():
        tensor = tensor.clamp(0, 1).mul(255)
    else:
        tensor = tensor.clamp(0, 255)
    array = tensor.to(torch.uint8).numpy()
    return Image.fromarray(array, mode="RGB")


def _as_logits(output: Any) -> Any:
    """Support plain tensor, dict-style output, or tuple/list output."""
    if isinstance(output, dict):
        if "out" in output:
            return output["out"]
        if "logits" in output:
            return output["logits"]
        raise ValueError(f"cannot find logits in model output keys: {list(output.keys())}")
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _list_images(image_dir: Union[str, Path]) -> List[Path]:
    root = _must_exist(image_dir)
    if not root.is_dir():
        raise NotADirectoryError(root)
    images = [p.resolve() for p in sorted(root.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise FileNotFoundError(f"no images found in {root}")
    return images


def _chunks(items: Sequence[Path], batch_size: int) -> List[Sequence[Path]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _question_sets(questions: Any, image_paths: Sequence[Path]) -> List[List[str]]:
    if isinstance(questions, str):
        return [[questions] for _ in image_paths]

    if isinstance(questions, dict):
        out = []
        for path in image_paths:
            value = questions.get(str(path)) or questions.get(path.name) or questions.get(path.stem)
            if value is None:
                raise KeyError(f"missing questions for image: {path.name}")
            out.append(_as_question_list(value))
        return out

    values = list(questions)
    if not values:
        raise ValueError("questions must not be empty")
    if all(isinstance(v, str) for v in values):
        return [[str(v) for v in values] for _ in image_paths]
    if len(values) != len(image_paths):
        raise ValueError("per-image questions must have the same length as image_dir images")
    return [_as_question_list(v) for v in values]


def _as_question_list(value: Any) -> List[str]:
    items = [value] if isinstance(value, str) else list(value)
    questions = [str(v).strip() for v in items if str(v).strip()]
    if not questions:
        raise ValueError("question list must not be empty")
    return questions


def _must_exist(path: Union[str, Path]) -> Path:
    path = Path(path).expanduser()
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path, PACKAGE_DIR / path, path]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(path)


def _device(device: Optional[str]) -> str:
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _norm(x: Any) -> str:
    return " ".join(str(x).lower().replace("_", " ").replace("-", " ").split())


def _targets(x: Union[str, Sequence[str]]) -> List[str]:
    values = [x] if isinstance(x, str) else list(x)
    return [_norm(v) for v in values]


def _torch_state(torch_module: Any, checkpoint: Path) -> Dict[str, Any]:
    try:
        ckpt = torch_module.load(str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch_module.load(str(checkpoint), map_location="cpu")
    except Exception:
        ckpt = torch_module.load(str(checkpoint), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "ema" in ckpt:
            return ckpt["ema"]["module"]
        if "model" in ckpt:
            return ckpt["model"]
        if "model_state" in ckpt:
            return ckpt["model_state"]
    return ckpt


def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    return {k.removeprefix("module."): v for k, v in state.items()}


def _round_list(values: Sequence[float]) -> List[float]:
    return [round(float(v), 2) for v in values]


def _torch_dtype(torch_module: Any, dtype_name: Optional[str], device: str) -> Any:
    if dtype_name is None:
        return torch_module.float16 if device.startswith("cuda") else "auto"
    if dtype_name == "auto":
        return "auto"
    if not hasattr(torch_module, dtype_name):
        raise ValueError("torch_dtype must be one of: auto, float16, bfloat16, float32")
    return getattr(torch_module, dtype_name)


def _model_float_dtype(model: Any) -> Any:
    try:
        for param in model.parameters():
            if param.is_floating_point():
                return param.dtype
    except Exception:
        return None
    return None


def _inputs_to_device(inputs: Any, device: str, dtype: Any = None) -> Any:
    import torch

    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            if dtype is not None and value.is_floating_point():
                inputs[key] = value.to(device=device, dtype=dtype)
            else:
                inputs[key] = value.to(device=device)
    return inputs


def _generation_kwargs(max_new_tokens: int, temperature: float) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        kwargs["temperature"] = temperature
    return kwargs


def _decode_generated(processor: Any, output: Any, input_ids: Any) -> str:
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, output)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def _resize_mask(mask: Any, size: Sequence[int]) -> Any:
    import numpy as np
    resample = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
    return np.array(Image.fromarray(mask).resize(tuple(size), resample=resample))


def _resize_conf(conf: Any, size: Sequence[int]) -> Any:
    import numpy as np
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    return np.array(Image.fromarray(conf.astype("float32")).resize(tuple(size), resample=resample))


def _semantic_segment_mask(mask: Any, names: Sequence[str]) -> Any:
    import numpy as np

    if names and _norm(names[0]) == "background":
        return np.asarray(mask) != 0
    return np.ones_like(mask, dtype=bool)


def _masked_mean(values: Any, mask: Any) -> float:
    import numpy as np

    region = np.asarray(mask, dtype=bool)
    if not region.any():
        return 0.0
    return round(float(np.asarray(values)[region].mean()), 4)


def _class_summary(mask: Any, conf: Any, names: Sequence[str], stability: Any = None) -> List[Dict[str, Any]]:
    import numpy as np
    total = int(mask.size)
    out = []
    for cls_id in np.unique(mask):
        cls_id = int(cls_id)
        region = mask == cls_id
        item = {
            "id": cls_id,
            "name": names[cls_id] if cls_id < len(names) else f"class_{cls_id}",
            "area": int(region.sum()),
            "ratio": round(float(region.sum()) / total, 6),
            "confidence": round(float(conf[region].mean()), 4),
        }
        if stability is not None:
            item["stability_score"] = round(float(stability[region].mean()), 4)
        out.append(item)
    return out


def _decode_deeplab_mask(mask: Any, dataset: str) -> Any:
    import numpy as np

    if dataset == "cityscapes":
        colors = np.array([
            (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
            (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
            (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
            (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
            (0, 80, 100), (0, 0, 230), (119, 11, 32),
        ], dtype=np.uint8)
        clipped = np.clip(mask, 0, len(colors) - 1)
        return colors[clipped]

    cmap = _voc_cmap()
    return cmap[mask]


def _voc_cmap() -> Any:
    import numpy as np

    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r = g = b = 0
        value = i
        for j in range(8):
            r |= ((value >> 0) & 1) << (7 - j)
            g |= ((value >> 1) & 1) << (7 - j)
            b |= ((value >> 2) & 1) << (7 - j)
            value >>= 3
        cmap[i] = np.array([r, g, b], dtype=np.uint8)
    return cmap


def _colorize_instance_map(instance_map: Any) -> Any:
    import numpy as np

    h, w = instance_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(instance_map)
    for idx in unique_ids:
        idx = int(idx)
        if idx == 0:
            continue
        rgb = _instance_color(idx)
        color[instance_map == idx] = rgb
    return color


def _instance_color(idx: int) -> List[int]:
    return [
        (idx * 37) % 256,
        (idx * 67) % 256,
        (idx * 97) % 256,
    ]


def _xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box]
    return [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)]


def _safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def _save_json(data: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _clear_proxy_env() -> None:
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ.pop(key, None)


def _set_hf_cache_env(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root / "hf_home")
    os.environ["HF_MODULES_CACHE"] = str(cache_root / "hf_modules")


def _hf_source(model_name: str, cache_root: Path, local_files_only: Optional[bool]) -> tuple[str, bool]:
    snapshot = _hf_cached_snapshot(model_name, cache_root)
    if snapshot and local_files_only is not False:
        return str(snapshot), True
    return model_name, bool(local_files_only)


def _hf_cached_snapshot(model_name: str, cache_root: Path) -> Optional[Path]:
    cache_name = "models--" + model_name.replace("/", "--")
    snapshots = cache_root / cache_name / "snapshots"
    candidates = sorted((p for p in snapshots.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for snapshot in candidates:
        if any(snapshot.glob("*.safetensors")) or any(snapshot.glob("pytorch_model*.bin")):
            return snapshot.resolve()
    return None


def _has_hf_model_cache(model_name: str, cache_root: Path = MAMBAVISION_ROOT) -> bool:
    return _hf_cached_snapshot(model_name, cache_root) is not None


def _sam2_size(size: str) -> str:
    aliases = {"t": "tiny", "s": "small", "b": "base_plus", "base": "base_plus", "l": "large"}
    return aliases.get(size.lower(), size.lower())

if __name__ == "__main__":
    det = load_object_extractor("deim", model_size="n", device="cuda:0")
    boxes = extract_objects(det, "style_prompt/img/img", ["cell phone"])
    print(boxes)
    det = load_object_extractor("yolo11", model_size="n", device="cuda:0")
    boxes = extract_objects(det, "style_prompt/img/img", ["cell phone"])
    print(boxes)

    # Example 2: DeepLabV3+ semantic segmentation
    seg = load_segmenter("deeplabv3plus", model_size="mobilenet", dataset="voc", device="cuda:0")
    masks = segment_images(seg, "style_prompt/img/img", output_dir="experiment_outputs/cvTask/segment/deeplab")
    print(masks)

    # Example 3: SAM2 automatic masks
    seg = load_segmenter("sam2", model_size="base_plus", device="cuda:0")
    masks = segment_images(seg, "style_prompt/img/img", output_dir="experiment_outputs/cvTask/segment/sam2", max_masks=20)
    print(masks)

    clf = load_image_classifier("resnet50", device="cuda:0")
    results = classify_images(clf, "style_prompt/img/img", batch_size=16, topk=3)
    print(results)

    clf = load_image_classifier("mambavision", device="cuda:0")
    results = classify_images(clf, "style_prompt/img/img", batch_size=10, topk=3)
    print(results)

    vlm = load_visual_understander("qwen3-vl",device="cuda:0",local_files_only=True,)
    results = get_visual_understanding(vlm,image_dir="style_prompt/img/img",questions=["What is in the photo? options: A. dog, B. sun, C. flower, D. person"],max_new_tokens=128,temperature=0.0,min_pixels=50176,max_pixels=262144,)
    print(results)

    vlm = load_visual_understander("llava-1.5",device="cuda:0",torch_dtype="float16",local_files_only=True,)
    results = get_visual_understanding(vlm,"style_prompt/img/img",["What is in the photo? options: A. dog, B. sun, C. flower, D. person"],max_new_tokens=128,)
    print(results)


    # clip=load_clip_model(local_files_only=False)
    # from tools_generator import load_generator_model, get_images
    #
    # pipe = load_generator_model('sd15')
    #
    # images = get_images(pipe,prompt="dog",seed=123,num_images_per_prompt=1,height=512,width=512)
    #
    # print(clip_image_text_distance(clip,images,"dog"))
