"""本地图像艺术程度评估，以及InstantStyle-Plus风格迁移。"""

import gc
import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path
import sys
import uuid

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from tools_paths import PACKAGE_DIR, TOOLS_ROOT, resolve_tools_asset_path


ARTSCORE_ROOT = TOOLS_ROOT / "ArtScore"
CLIPIQA_ROOT = TOOLS_ROOT / "CLIP-IQA"
CLIPIQA_CHECKPOINT = CLIPIQA_ROOT / "RN50.pt"
CLIP_ROOT = TOOLS_ROOT / "CLIP"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
TEXT_MODEL_ROOTS = {
    "blip": TOOLS_ROOT / "BLIP",
    "siglip2": TOOLS_ROOT / "SigLIP2",
}
PROMPTS = ("Real photo.", "Abstract photo.")
MODEL_TYPES = ("artscore", "clip-iqa", "blip", "siglip2")
DEFAULT_CHECKPOINT = ARTSCORE_ROOT / "ckpt" / (
    "loss@listMLE_model@resnet50_denseLayer@True_batch_size@16_lr@0.0001_dropout@0.5_E_8.pth"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SIMILARITY_TYPES = ("CLIPScore", "BLIP")


def _transfer_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _configure_transfer_pipe(pipe, device, offload="model"):
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    if offload == "none":
        pipe.to(device)
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
    else:
        pipe.enable_model_cpu_offload(device=device)
    return pipe


def _instantstyle_code():
    code = TOOLS_ROOT / "InstantStyle-Plus/code"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    return code


def _load_instantstyle_pipeline(assets, device, style_scale, content_scale, offload="model"):
    """共用的官方InstantStyle-Plus推理管线；不修改已安装包。"""
    from diffusers import ControlNetModel, DDIMScheduler
    from transformers import CLIPVisionModelWithProjection

    source = _instantstyle_code() / "pipeline_controlnet_sd_xl_img2img.py"
    spec = importlib.util.spec_from_file_location("_evaluation_instantstyle", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_requires_grad = module.set_requires_grad
    module.set_requires_grad = lambda model, value: original_requires_grad(model, value) if model is not None else None
    pipeline_class = module.StableDiffusionXLControlNetImg2ImgPipeline
    pipeline_class._optional_components = [*pipeline_class._optional_components, "clip_model"]
    control = ControlNetModel.from_pretrained(assets["tile"], torch_dtype=torch.float16, local_files_only=True)
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        assets["ip_adapter"], subfolder="models/image_encoder", torch_dtype=torch.float16, local_files_only=True)
    pipe = pipeline_class.from_pretrained(
        assets["sdxl"], controlnet=control, image_encoder=encoder, clip_model=None,
        torch_dtype=torch.float16, local_files_only=True, add_watermarker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter([str(assets["ip_adapter"])] * 2, subfolder=["sdxl_models"] * 2,
                         weight_name=["ip-adapter_sdxl_vit-h.safetensors"] * 2,
                         image_encoder_folder=None, local_files_only=True)
    pipe.set_ip_adapter_scale([content_scale, {"up": {"block_0": [0.0, style_scale, 0.0]}}])
    # 旧管线的动态upcast与CPU卸载钩子冲突，在注册钩子前固定VAE的FP32解码。
    vae = pipe.vae.to(dtype=torch.float32)
    original_decode = vae.decode
    def decode_with_matching_dtype(latent, *values, **kwargs):
        return original_decode(latent.to(dtype=vae.dtype), *values, **kwargs)
    vae.decode = decode_with_matching_dtype
    return _configure_transfer_pipe(pipe, device, offload)


def _load_instantstyle_inversion(assets, device, seed):
    _instantstyle_code()
    from src.config import RunConfig
    from src.eunms import Model_Type, Scheduler_Type
    from src.pipes.sdxl_inversion_pipeline import SDXLDDIMPipeline
    from src.schedulers.ddim_scheduler import MyDDIMScheduler

    pipe = SDXLDDIMPipeline.from_pretrained(
        assets["sdxl"], torch_dtype=torch.float16, local_files_only=True, add_watermarker=False)
    pipe.scheduler = MyDDIMScheduler.from_config(pipe.scheduler.config)
    pipe.cfg = RunConfig(model_type=Model_Type.SDXL, scheduler_type=Scheduler_Type.DDIM,
                         seed=seed, num_inference_steps=30, num_inversion_steps=30,
                         num_renoise_steps=1, perform_noise_correction=False)
    return _configure_transfer_pipe(pipe, device)


def _transfer_generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def _save_transfer_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    image.convert("RGB").save(temporary)
    temporary.replace(path)


def transfer_imgaes(base_prompts: str, modifiers: str, style_scale=1.6,
                    content_scale=0.15, control_scale=0.25, seed=2026) -> dict[str, str]:
    """用InstantStyle-Plus迁移风格，每次重新生成，不复用图像或反演缓存。

    base_prompts、modifiers均为单个字符串。三个scale依次为风格IP-Adapter、
    内容IP-Adapter、Tile ControlNet权重。所有阶段直接使用同一个seed。
    固定512×512、30步，保留当前标准SDXL及关闭额外CSD引导的设置。
    输出至experiment_outputs/transfer_imgs/{ref_imgs,transfer_imgs,ori_iomgs}/，
    返回三个同名PNG的路径字典；每次调用使用新的文件名，不写JSON或日志。
    """
    from diffusers import StableDiffusionPipeline

    if not isinstance(base_prompts, str) or not isinstance(modifiers, str):
        raise TypeError("base_prompts and modifiers must be strings")
    if not base_prompts.strip() or not modifiers.strip():
        raise ValueError("base_prompts and modifiers must not be empty")
    if any(not math.isfinite(v) or v < 0 for v in (style_scale, content_scale, control_scale)):
        raise ValueError("Style/content/control scales must be finite and nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not torch.cuda.is_available():
        raise RuntimeError("InstantStyle-Plus inference requires CUDA")
    device = f"cuda:{torch.cuda.current_device()}"
    config = json.loads((TOOLS_ROOT / "StyleTransfer/runtime_assets.json").read_text())["assets"]
    assets = {k: resolve_tools_asset_path(config[k]["path"])
              for k in ("sd15", "sdxl", "tile", "ip_adapter")}
    for path in assets.values():
        if not path.exists():
            raise FileNotFoundError(path)
    output = PACKAGE_DIR / "experiment_outputs/transfer_imgs"
    name = f"{seed}_{uuid.uuid4().hex}.png"
    paths = {kind: output / kind / name for kind in ("ref_imgs", "transfer_imgs", "ori_iomgs")}

    pipe = StableDiffusionPipeline.from_pretrained(
        assets["sd15"], torch_dtype=torch.float16, local_files_only=True,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    try:
        _configure_transfer_pipe(pipe, device)
        with torch.inference_mode():
            original = pipe(base_prompts, generator=_transfer_generator(device, seed),
                            num_inference_steps=30, guidance_scale=7.5, height=512, width=512).images[0]
            reference = pipe(f"{base_prompts}, {modifiers}", generator=_transfer_generator(device, seed),
                             num_inference_steps=30, guidance_scale=7.5, height=512, width=512).images[0]
        _save_transfer_image(original, paths["ori_iomgs"])
        _save_transfer_image(reference, paths["ref_imgs"])
    finally:
        del pipe
        _transfer_cleanup()

    pipe = _load_instantstyle_inversion(assets, device, seed)
    try:
        # ReNoise的CPU随机采样使用同一seed，并在结束后恢复调用方的CPU RNG状态。
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            result = pipe(image=original, prompt=base_prompts,
                          num_inversion_steps=30, num_inference_steps=30, strength=1.0,
                          denoising_start=0.0, guidance_scale=0.0, num_renoise_steps=1,
                          generator=_transfer_generator("cpu", seed))
            latent = result[0][0].detach().cpu()
            del result
    finally:
        del pipe
        _transfer_cleanup()

    pipe = _load_instantstyle_pipeline(assets, device, style_scale, content_scale)
    try:
        with torch.inference_mode():
            image = pipe(prompt=base_prompts, image=latent.to(device, torch.float16), control_image=original,
                         ip_adapter_image=[original, reference], generator=_transfer_generator(device, seed),
                         num_inference_steps=30, denoising_start=0.0001, guidance_scale=5.0,
                         controlnet_conditioning_scale=control_scale,
                         style_guidance_scale=0, content_guidance_scale=0,
                         height=512, width=512).images[0]
        _save_transfer_image(image, paths["transfer_imgs"])
    finally:
        del pipe
        _transfer_cleanup()
    return {kind: str(path) for kind, path in paths.items()}


def _similarity_type(type):
    if not isinstance(type, str):
        raise TypeError("type must be a string")
    normalized = type.strip().lower().replace("-", "").replace("_", "")
    if normalized not in {"clipscore", "blip"}:
        raise ValueError(f"type must be one of {SIMILARITY_TYPES}")
    return normalized


def _similarity_device():
    return f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"


def _clip_local_sources():
    """返回本地 CLIP 权重目录和处理器配置目录。"""
    cache = CLIP_ROOT / ("models--" + CLIP_MODEL_ID.replace("/", "--")) / "snapshots"
    snapshots = [path for path in cache.glob("*") if path.is_dir()]
    weight_roots = sorted(
        (path for path in snapshots if (path / "model.safetensors").is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    processor_roots = sorted(
        (
            path for path in snapshots
            if (path / "config.json").is_file()
            and (path / "preprocessor_config.json").is_file()
            and (path / "tokenizer.json").is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not weight_roots or not processor_roots:
        raise FileNotFoundError(f"incomplete local CLIP model cache: {cache}")
    return weight_roots[0], processor_roots[0]


@lru_cache(maxsize=None)
def _load_similarity_model(type, device):
    """按模型类型和设备缓存相似度模型，避免重复加载权重。"""
    if type == "clipscore":
        from transformers import CLIPConfig, CLIPModel, CLIPProcessor

        weight_root, processor_root = _clip_local_sources()
        config = CLIPConfig.from_pretrained(processor_root, local_files_only=True)
        processor = CLIPProcessor.from_pretrained(
            processor_root, local_files_only=True, use_fast=False
        )
        model = CLIPModel.from_pretrained(
            weight_root,
            config=config,
            local_files_only=True,
            use_safetensors=True,
        )
    else:
        from transformers import AutoProcessor, BlipForImageTextRetrieval

        root = TEXT_MODEL_ROOTS["blip"]
        processor = AutoProcessor.from_pretrained(root, local_files_only=True, use_fast=False)
        model = BlipForImageTextRetrieval.from_pretrained(
            root, local_files_only=True, use_safetensors=True
        )
    return model.to(device).eval(), processor


def _open_similarity_image(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, (str, Path)):
        path = Path(img)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            return image.convert("RGB")
    raise TypeError("img must be a PIL.Image.Image or an image file path")


@torch.inference_mode()
def similarity_calculate(cal_type, img, text) -> float:
    """计算单张图像与文本的相似度，返回四舍五入到两位小数的 0～100 分数。

    cal_type 支持 ``CLIPScore`` 和 ``BLIP``（不区分大小写，也接受连字符或下划线）。
    img 支持 PIL 图像或图像文件路径，text 必须为非空字符串。
    CLIPScore 为 ``max(cosine_similarity, 0) * 100``；BLIP 为图文匹配头中
    “匹配”类别的概率乘以 100。两种分数均为越高表示图文越相似。
    """
    model_type = _similarity_type(cal_type)
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = text.strip()
    if not text:
        raise ValueError("text must not be empty")

    device = _similarity_device()
    model, processor = _load_similarity_model(model_type, device)
    image = _open_similarity_image(img)

    if model_type == "clipscore":
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        text_inputs = processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(device)
        image_features = nn.functional.normalize(
            model.get_image_features(**image_inputs).float(), dim=-1
        )
        text_features = nn.functional.normalize(
            model.get_text_features(**text_inputs).float(), dim=-1
        )
        score = 100 * (image_features * text_features).sum(dim=-1).clamp(min=0).item()
    else:
        inputs = processor(
            images=image,
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        logits = model(**inputs, use_itm_head=True).itm_score.float()
        if logits.ndim != 2 or logits.shape != (1, 2):
            raise ValueError(f"unexpected BLIP ITM score shape: {tuple(logits.shape)}")
        score = 100 * logits.softmax(dim=-1)[0, 1].item()

    if not math.isfinite(score):
        raise ValueError("similarity score is not finite")
    return round(score, 2)


def load_art_model(type="artscore", device=None) -> nn.Module:
    """加载指定模型及预处理器；仅访问本地文件，不下载或修改环境。"""
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if type == "artscore":
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Linear(2048, 1000), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1000, 1)
        )
        model.load_state_dict(torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True))
    elif type == "clip-iqa":
        from transformers import CLIPTokenizerFast

        # 只加载仓库的模型核心，避免导入依赖旧版 MMCV 的 MMEditing 框架。
        source = CLIPIQA_ROOT / "mmedit" / "models" / "components" / "clip" / "model.py"
        spec = importlib.util.spec_from_file_location("_clip_iqa_model", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        checkpoint = torch.jit.load(str(CLIPIQA_CHECKPOINT), map_location="cpu")
        model = module.build_model(checkpoint.state_dict()).float()

        # 所有 CLIP 骨干共用分词器；复用已有本地缓存，固定英文提示词无需 ftfy。
        cache = CLIP_ROOT / "models--openai--clip-vit-base-patch32"
        tokenizers = sorted((cache / "snapshots").glob("*/tokenizer.json"))
        if not tokenizers:
            raise FileNotFoundError(f"CLIP tokenizer not found in {cache}")
        tokenizer = CLIPTokenizerFast.from_pretrained(tokenizers[0].parent, local_files_only=True)
        tokens = torch.zeros(2, 77, dtype=torch.long)
        for i, text in enumerate(PROMPTS):
            ids = tokenizer(text)["input_ids"]
            tokens[i, :len(ids)] = torch.tensor(ids)
        model.register_buffer("clip_iqa_tokens", tokens)
    elif type in TEXT_MODEL_ROOTS:
        from transformers import AutoModel, AutoProcessor, BlipForImageTextRetrieval

        root = TEXT_MODEL_ROOTS[type]
        model_class = BlipForImageTextRetrieval if type == "blip" else AutoModel
        model = model_class.from_pretrained(root, local_files_only=True, use_safetensors=True)
        model.art_processor = AutoProcessor.from_pretrained(
            root, local_files_only=True, use_fast=False
        )
    else:
        raise ValueError(f"type must be one of {MODEL_TYPES}")
    model.art_model_type = type
    model = model.to(device).eval()
    if type == "blip":
        # 固定提示词的几何界限 D；仅在加载时计算一次，随模型移动设备。
        with torch.no_grad():
            tokens = model.art_processor(text=list(PROMPTS), padding=True, return_tensors="pt").to(device)
            hidden = model.text_encoder(**tokens).last_hidden_state[:, 0, :]
            features = nn.functional.normalize(model.text_proj(hidden), dim=-1)
            distance = (features[1] - features[0]).norm()
        if not torch.isfinite(distance) or distance <= 0:
            raise ValueError("BLIP prompt features must have a finite, nonzero distance")
        model.register_buffer("art_text_distance", distance)
    return model


@torch.inference_mode()
def get_art_scores(type, image_dir, model) -> list[float]:
    """按文件名排序，返回分数列表 [score_1, ..., score_N]，四舍五入到两位小数。

    artscore 返回原始分数；其余模型返回 0～100 的相对抽象倾向分数。
    clip-iqa 衡量非写实程度，采用官方 RN50 和关闭图像位置编码的推理方式。
    blip 返回 50 * (1 + (s_abstract - s_real) / D)，D 为归一化文本特征的距离。
    该映射的几何理论范围为 [0, 100]，不保证实际图像覆盖整个区间。
    siglip2 按训练要求将提示词转小写，使用自带缩放的 logits 做双提示词 softmax。
    三者均为越高越抽象的相对匹配分数，并非校准的抽象程度，不能跨模型直接比较。
    """
    if type not in MODEL_TYPES:
        raise ValueError(f"type must be one of {MODEL_TYPES}")
    if type != model.art_model_type:
        raise ValueError("type must match the loaded model")
    image_paths = sorted(
        p for p in Path(image_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        raise FileNotFoundError(f"no images found in {image_dir}")
    if type == "artscore":
        transform = transforms.Compose([
            transforms.Resize(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif type == "clip-iqa":
        # 与官方 test_pipeline 一致：保持原始分辨率，仅转 RGB、缩放像素并归一化。
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711]),
        ])
    device = next(model.parameters()).device
    scores = []
    for path in image_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if type in TEXT_MODEL_ROOTS:
                # SigLIP2 训练时使用小写文本、固定长度 64；旧版 processor 不自动转小写。
                padding = {"padding": "max_length", "max_length": 64, "truncation": True}
                texts = [text.lower() for text in PROMPTS] if type == "siglip2" else list(PROMPTS)
                inputs = model.art_processor(
                    images=image, text=texts, return_tensors="pt",
                    **(padding if type == "siglip2" else {"padding": True}),
                ).to(device)
            else:
                tensor = transform(image).unsqueeze(0).to(device)
        if type in TEXT_MODEL_ROOTS:
            if type == "blip":
                logits = model(**inputs, use_itm_head=False).itm_score
                difference = logits[0, 1] - logits[0, 0]
                score = (50 * (1 + difference / model.art_text_distance)).clamp(0, 100).item()
            else:
                logits = model(**inputs).logits_per_image
                score = 100 * logits.softmax(dim=-1)[0, 1].item()
        elif type == "artscore":
            score = model(tensor).item()
        else:
            logits, _ = model(tensor, model.clip_iqa_tokens, pos_embedding=False)
            realism = logits.softmax(dim=-1)[0, 0].item()
            score = 100 * (1 - realism)
        scores.append(round(score, 2))
    return scores


if __name__ == "__main__":
    # for model_type in ("clip-iqa", "artscore", "blip", "siglip2"):
    #     print(f"================ {model_type} ================", flush=True)
    #     model = load_art_model(type=model_type)
    #     scores = get_art_scores(model_type, PACKAGE_DIR / "img/art", model)
    #     for i in range(0, len(scores), 5):
    #         print("[" + ", ".join(f"{score:.2f}" for score in scores[i:i + 5]) + "]")
    #     del model
    paths = transfer_imgaes(
        base_prompts="a golden retriever",
        modifiers="comic art",
        style_scale=1.7,
        content_scale=0.2,
        control_scale=0.4,
        seed=2026,
    )
