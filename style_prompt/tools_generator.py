"""本地图像生成与 APIMart 文生图统一接口。"""
import importlib.util
from pathlib import Path
import sys
from functools import lru_cache

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusion3Pipeline
from tools_paths import TOOLS_ROOT

GENERATOR_ROOT = TOOLS_ROOT / "Generators"
LOCAL_MODELS = {
    "sld": ("AIML-TUDA/stable-diffusion-safe", "sld"),
    "flux.1-dev": ("black-forest-labs/FLUX.1-dev", "flux.1-dev"),
    "flux-3-image-klein": ("black-forest-labs/flux-3-image-klein", "flux-3-image-klein"),
}
SLD_CONFIGS = {
    "sld-medium": dict(sld_warmup_steps=10, sld_guidance_scale=1000,
                       sld_threshold=0.01, sld_momentum_scale=0.3, sld_mom_beta=0.4),
    "sld-strong": dict(sld_warmup_steps=7, sld_guidance_scale=2000,
                       sld_threshold=0.025, sld_momentum_scale=0.5, sld_mom_beta=0.7),
    "sld-max": dict(sld_warmup_steps=0, sld_guidance_scale=5000,
                    sld_threshold=1.0, sld_momentum_scale=0.5, sld_mom_beta=0.7),
}


@lru_cache(maxsize=2)
def _load_post_safety_checker(kind):
    """后置检查器固定在CPU/FP32执行，避免与大生成模型争用显存。"""
    from transformers import CLIPImageProcessor, pipeline
    path = GENERATOR_ROOT / ("sd-safety-checker" if kind == "sd" else "flux-safety-checker")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"SC资源缺失：{path}；运行 prepare_generator_models.py --models {path.name}")
    if kind == "sd":
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
        model = StableDiffusionSafetyChecker.from_pretrained(
            str(path), local_files_only=True, torch_dtype=torch.float32).eval().to("cpu")
        processor = CLIPImageProcessor.from_pretrained(str(path), local_files_only=True)
        return model, processor
    if kind == "flux":
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(
            str(path), local_files_only=True, torch_dtype=torch.float32).eval()
        processor = AutoImageProcessor.from_pretrained(str(path), local_files_only=True)
        return pipeline("image-classification", model=model, image_processor=processor, device=-1)
    raise ValueError(f"Unsupported safety checker: {kind}")


@torch.inference_mode()
def _filter_generated_images(pipe, images):
    """在get_images的所有返回路径执行SC；命中时返回等尺寸黑图，保留图像数量。"""
    import numpy as np
    from PIL import Image
    kind = getattr(pipe, "_style_prompt_sc_kind", None)
    if kind not in ("sd", "flux"):
        return images
    checker = pipe._style_prompt_sc
    if kind == "sd":
        model, processor = checker
        pixels = processor(images=images, return_tensors="pt").pixel_values.to("cpu", torch.float32)
        # 原始图片尺寸可以不同；检查器仅需要逐图数组及统一CLIP输入。
        arrays = [np.array(image.convert("RGB"), copy=True) for image in images]
        _, flags = model(clip_input=pixels, images=arrays)
    else:
        flags = []
        for image in images:
            scores = checker(image, top_k=None)
            nsfw = [float(item["score"]) for item in scores if item["label"].lower() == "nsfw"]
            if len(nsfw) != 1:
                raise RuntimeError("FLUX安全检查器未返回唯一nsfw分数")
            if not np.isfinite(nsfw[0]) or not 0 <= nsfw[0] <= 1:
                raise RuntimeError("FLUX安全检查器返回无效分数")
            # BFL官方save_image使用严格小于0.85作为放行条件。
            flags.append(nsfw[0] >= 0.85)
    if len(flags) != len(images):
        raise RuntimeError("安全检查结果数量与图片数量不一致")
    pipe._style_prompt_nsfw_flags = [bool(flag) for flag in flags]
    return [Image.new("RGB", image.size, 0) if flag else image
            for image, flag in zip(images, flags)]


def _local_generator_path(key):
    path = GENERATOR_ROOT / LOCAL_MODELS[key][1]
    index = "modular_model_index.json" if key == "flux-3-image-klein" else "model_index.json"
    if not (path / index).is_file():
        raise FileNotFoundError(
            f"本地模型不完整：{path}。请先运行 prepare_generator_models.py --models {key}"
        )
    return path


def _sld_classes():
    # SLD 已从新版 Diffusers 移除，使用 tools 下保留的官方实现。
    code = GENERATOR_ROOT / "code" / "sld"
    init = code / "__init__.py"
    if not init.is_file():
        raise FileNotFoundError(f"SLD源码缺失：{init}；运行 prepare_generator_models.py --models sld")
    name = "_style_prompt_sld"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(code)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    module = sys.modules[name]
    return module.StableDiffusionPipelineSafe, module.SafeStableDiffusionSafetyChecker


def load_generator_model(model_type: str, device: str = "cuda", fp16: bool = True,
                         check: bool = False, cpu_offload: bool = False):
    """支持 sd15/sdxl/sd3、sld-medium/strong/max、flux.1-dev、flux-3-image-klein。

    新增模型仅从 tools/Generators 加载，缺文件直接报错。FLUX 使用 BF16
    （fp16=False 时 FP32）；check控制get_images生成后的SC，SLD安全引导始终启用。
    cpu_offload=True时FLUX.1-dev采用逐层CPU卸载，避免整个Transformer
    同时进入显存；其他模型按整组件卸载。卸载需要额外主存并降低速度。
    """
    model_type = model_type.lower().replace("_", "-")
    aliases = {"flux1-dev": "flux.1-dev", "flux-1-dev": "flux.1-dev",
               "flux3-klein": "flux-3-image-klein",
               "flux-3-image-klein-distilled": "flux-3-image-klein"}
    model_type = aliases.get(model_type, model_type)
    dtype = torch.float16 if fp16 else torch.float32
    is_sld = model_type in SLD_CONFIGS
    is_flux = model_type in ("flux.1-dev", "flux-3-image-klein")
    if model_type not in {"sd15", "sdxl", "sd3", *SLD_CONFIGS, "flux.1-dev", "flux-3-image-klein"}:
        raise ValueError(f"Unsupported model: {model_type}")
    sc_kind = ("flux" if is_flux else "sd") if check and not is_sld else None
    # 先加载检查器，确保缺少SC资源时不会先加载数十GB生成权重。
    sc = _load_post_safety_checker(sc_kind) if sc_kind else None

    if model_type == "sd15":
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=dtype, safety_checker=None)
    elif model_type == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype, use_safetensors=True)
    elif model_type == "sd3":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=dtype)
    elif is_sld:
        path = _local_generator_path("sld")
        pipeline_class, checker_class = _sld_classes()
        checker = None
        if check:
            checker = checker_class.from_pretrained(
                str(path / "safety_checker"), torch_dtype=dtype, local_files_only=True)
        pipe = pipeline_class.from_pretrained(
            str(path), torch_dtype=dtype, safety_checker=checker,
            requires_safety_checker=check, local_files_only=True)
    elif model_type == "flux.1-dev":
        from diffusers import FluxPipeline
        path = _local_generator_path(model_type)
        pipe = FluxPipeline.from_pretrained(
            str(path), torch_dtype=torch.bfloat16 if fp16 else torch.float32,
            local_files_only=True)
    elif model_type == "flux-3-image-klein":
        import diffusers
        if not hasattr(diffusers, "Flux3AutoBlocks"):
            raise ImportError(
                f"diffusers {diffusers.__version__} 没有FLUX.3管线。仅下载权重不能补齐实现；"
                "官方0.42.1实现还要求Python>=3.11和huggingface-hub>=1.32.0。"
                "当前环境未被修改，请使用支持Flux3AutoBlocks的独立运行环境。")
        path = _local_generator_path(model_type)
        pipe = diffusers.ModularPipeline.from_pretrained(str(path), local_files_only=True)
        # 显式重定向每个组件，避免model_index中的repo ID触发Hub缓存。
        for name in pipe.pretrained_component_names:
            spec = pipe.get_component_spec(name)
            spec.pretrained_model_name_or_path = str(path)
            spec.subfolder = name
            component = spec.load(torch_dtype=torch.bfloat16 if fp16 else torch.float32,
                                  local_files_only=True)
            pipe.update_components(**{name: component})
    else:
        raise ValueError(f"Unsupported model: {model_type}")

    if not is_sld and not is_flux:
        # SD系列统一在get_images中使用SD1.5检查器，防止原生SD1.5重复过滤。
        pipe.safety_checker = None
    if cpu_offload:
        if torch.device(device).type != "cuda":
            raise ValueError("cpu_offload需要CUDA执行设备")
        if model_type == "flux.1-dev":
            # BF16 Transformer本身已接近24GB；整组件卸载仍可能OOM，
            # 尤其同卡还需运行目标视觉模型及语义相似度模型时。
            # 必须在任何pipe.to(cuda)之前安装逐层卸载hook。
            pipe.enable_sequential_cpu_offload(device=device)
        else:
            pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    pipe._style_prompt_model_type = model_type
    pipe._style_prompt_device = str(device)
    pipe._style_prompt_sc_kind = sc_kind
    pipe._style_prompt_sc = sc
    pipe._style_prompt_nsfw_flags = None
    return pipe


def _generator_device(pipe):
    # CPU卸载时pipe.device可能是cpu，使用执行设备。
    device = getattr(pipe, "_execution_device", None)
    if device is None:
        device = getattr(pipe, "_style_prompt_device", None)
    return device if device is not None else pipe.device


def get_images(pipe, prompt, negative_prompt: str = "", seed: int = 42,
               num_images_per_prompt: int = 1, height: int = 512, width: int = 512,
               num_inference_steps=None, guidance_scale=None):
    """返回PIL图片列表。批量提示词各自从同一seed起始，便于修饰词比较。"""
    model_type = getattr(pipe, "_style_prompt_model_type", None)
    batch = isinstance(prompt, (list, tuple))
    prompts = list(prompt) if batch else [prompt]
    if not prompts or num_images_per_prompt < 1:
        raise ValueError("prompt和num_images_per_prompt必须非空/大于0")
    negatives = list(negative_prompt) if isinstance(negative_prompt, (list, tuple)) else [negative_prompt] * len(prompts)
    if len(negatives) != len(prompts):
        raise ValueError("negative_prompt数量必须与prompt一致")
    device = _generator_device(pipe)

    if model_type in ("flux.1-dev", "flux-3-image-klein"):
        # FLUX的latent为packed结构，不能使用SD的四维latent构造器。
        images = []
        klein = model_type == "flux-3-image-klein"
        if klein and (any(negatives) or guidance_scale is not None):
            raise ValueError("FLUX.3 Klein蒸馏版无CFG，不接受negative_prompt或guidance_scale")
        for text, negative in zip(prompts, negatives):
            gen = torch.Generator(device=device).manual_seed(seed)
            kwargs = dict(prompt=text, height=height, width=width, generator=gen,
                          num_inference_steps=num_inference_steps if num_inference_steps is not None else (16 if klein else 50))
            if klein:
                # 官方ModularPipeline每次只支持一条prompt、一张图片。
                for _ in range(num_images_per_prompt):
                    images.extend(pipe(**kwargs, output="images"))
            else:
                kwargs.update(num_images_per_prompt=num_images_per_prompt,
                              guidance_scale=3.5 if guidance_scale is None else guidance_scale,
                              max_sequence_length=512)
                if negative:
                    # FLUX.1 的negative_prompt只有true CFG启用时才生效。
                    import inspect
                    if "true_cfg_scale" not in inspect.signature(pipe.__call__).parameters:
                        raise ValueError("当前FluxPipeline不支持negative_prompt的true CFG")
                    kwargs.update(negative_prompt=negative, true_cfg_scale=2.0)
                images.extend(pipe(**kwargs).images)
        return _filter_generated_images(pipe, images)

    kwargs = dict(prompt=prompts if batch else prompt,
                  negative_prompt=negatives if batch else negatives[0],
                  height=height, width=width, num_images_per_prompt=num_images_per_prompt)
    if batch:
        kwargs["latents"] = make_prompt_batch_latents(
            pipe, len(prompts), num_images_per_prompt, height, width, seed)
    else:
        kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)
    if model_type == "sd3":
        kwargs.update(guidance_scale=7.0, num_inference_steps=28)
    if model_type in SLD_CONFIGS:
        if guidance_scale is not None and not guidance_scale > 1.0:
            raise ValueError("SLD的guidance_scale必须大于1，以保持安全引导启用")
        kwargs.update(SLD_CONFIGS[model_type], guidance_scale=10.0)
    if num_inference_steps is not None:
        kwargs["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        kwargs["guidance_scale"] = guidance_scale
    result = pipe(**kwargs)
    if model_type in SLD_CONFIGS:
        pipe._style_prompt_nsfw_flags = getattr(result, "nsfw_content_detected", None)
    return _filter_generated_images(pipe, result.images)


class APIModerationRejected(RuntimeError):
    """内容拒绝：stage 为 input/output/unknown，经验推断通过 inferred 标记。"""
    def __init__(self, message, stage, *, inferred=False, task_id=None, elapsed_seconds=None,
                 actual_time=None, billing_status=None, cost=None, credits_cost=None):
        super().__init__(message)
        self.stage = stage
        self.inferred = inferred
        self.api_status = {"input": "input_rejected", "output": "output_rejected",
                           "unknown": "content_rejected_unknown"}[stage]
        self.task_id = task_id
        self.elapsed_seconds = elapsed_seconds
        self.actual_time = actual_time
        self.billing_status = billing_status
        self.cost = cost
        self.credits_cost = credits_cost


def _classify_api_moderation(error, model=None, elapsed_seconds=None, http_status=None):
    """返回 (阶段, 是否推断)，非内容拒绝返回 (None, False)。不将耗时当作生成证据。"""
    import re
    if http_status is not None and http_status not in (200, 400, 403, 422):
        return None, False
    codes, messages = set(), []

    def collect(value):
        if isinstance(value, dict):
            for name in ("error", "errors", "code", "type", "reason", "message", "detail", "fail_reason"):
                if name in value:
                    collect(value[name])
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, str):
            codes.add(value.strip().lower().replace("-", "_").replace(" ", "_"))
            messages.append(" ".join(value.lower().split()))
    collect(error)
    text = " ".join(messages)
    operational = {"service_unavailable", "server_error", "internal_server_error", "authentication_error",
                   "invalid_api_key", "permission_error", "rate_limit_error", "insufficient_quota",
                   "insufficient_balance", "payment_required", "timeout", "invalid_parameter", "model_not_found",
                   "moderation_service_unavailable", "content_moderation_unavailable", "safety_checker_error"}
    # 用户对 banana2 的对照实验规则：仅匹配这段完整的异步任务通用错误。
    # 它也可能掩盖真实服务故障，因此始终保留 inferred=True，不推广到其他模型。
    banana_input_notice = (
        model in ("banana2", "gemini-3.1-flash-image-preview-official")
        and {"task_failed", "service_unavailable"}.issubset(codes)
        and "this model couldn't complete your request right now" in text
        and "please try again in a few minutes" in text
        and "any deducted quota has been refunded automatically" in text
        and not (codes & (operational - {"service_unavailable"}))
    )
    if banana_input_notice:
        return "input", True
    if codes & operational or re.search(
            r"\b(?:unavailable|timed? out|timeout|not configured|failed to connect)\b|"
            r"(?:审核|安全).{0,20}(?:服务不可用|服务异常|超时)", text):
        return None, False
    input_codes = {"prompt_blocked", "prompt_banned", "banned_prompt", "prompt_safety_violation",
                   "prompt_moderation_failed", "text_moderation_failed",
                   "input_content_blocked", "input_moderation_failed", "input_content_policy_violation"}
    output_codes = {"output_content_filter", "output_content_policy_violation", "image_safety_violation",
                    "output_moderation_blocked", "image_moderation_failed", "generated_image_blocked"}
    output_text = re.search(
        r"\b(?:generated|output) (?:image|content|response)\b.{0,100}"
        r"\b(?:blocked|filtered|rejected|violates?|violating)\b.{0,80}"
        r"\b(?:safety|content|policy|policies|moderation|nsfw)\b|"
        r"(?:生成的图像|生成图像|输出图像|输出图片).{0,50}(?:违规|有害|未通过.*审核|被.*拦截)", text)
    if codes & output_codes or output_text:
        return "output", False
    if codes & input_codes or re.search(
        r"\bbanned prompt\b|"
        r"\b(?:prompt|input)\b.{0,80}\b(?:blocked|rejected|flagged|banned|triggered|violates?|violating|against)\b"
        r".{0,100}\b(?:safety|content|polic(?:y|ies)|moderation|community|nsfw)\b|"
        r"\b(?:prompt|input)\b.{0,60}\bcontains?\b.{0,40}\b(?:banned words?|sensitive words?|prohibited content|unsafe content|nsfw)\b|"
        r"消息(?:中)?(?:包含|含有)(?:敏感|违禁)词|"
        r"(?:提示词|提示语|输入内容).{0,80}(?:违反|违背).{0,20}(?:社区|内容|安全)(?:政策|准则|规范)|"
        r"(?:提示词|提示语|输入内容).{0,80}(?:敏感词|违禁词|违规内容|审核未通过|未通过.*审核)", text):
        return "input", False
    content_codes = {"content_policy_violation", "content_filter", "content_filter_error", "content_blocked",
                     "content_moderation_blocked", "moderation_blocked", "safety_violation", "sensitive_content",
                     "prohibited_content"}
    content_text = re.search(
        r"failed content moderation|did not pass content safety review|"
        r"(?:rejected|blocked).{0,100}(?:safety system|content filter|content policy)|"
        r"(?:敏感内容|违规内容).{0,30}(?:拒绝|拦截)", text)
    if not (codes & content_codes or content_text):
        return None, False
    # GPT 的时长规则是用户指定的实验启发式，并非服务商确认的生成阶段。
    if model == "gpt-image-2.5-flare" and elapsed_seconds is not None:
        input_only_notice = all(part in text for part in (
            "did not pass content safety review", "no image was generated", "input only", "no output cost"))
        if input_only_notice and elapsed_seconds <= 8:
            return "input", True
        if elapsed_seconds > 10:
            return "output", True
    return "unknown", False


def _is_api_input_rejection(error, http_status=None):
    """兼容入口：仅明确的输入审核信息为 True，通用内容拒绝不猜测阶段。"""
    return _classify_api_moderation(error, http_status=http_status)[0] == "input"


def get_api_images(prompt, model: str = "midjourney", num_images_per_prompt: int = 1,
                   seed=42, *, api_key=None, base_url="https://api.apimart.ai/v1",
                   size="1:1", version=None, speed="relax", timeout=900,
                   poll_interval=5, request_timeout=60, strict_seed=False):
    """通过 APIMart 文生图，返回 None 或与 get_images 一致的 list[PIL.Image.Image]。

    prompt 支持字符串或字符串列表；结果按 prompt 顺序展平。Midjourney 部分返回时数量可少于请求值。
    文本拦截时本次调用返回 None（多提示词调用也不返回部分图片），不保存占位图。
    model 支持 midjourney6.1、midjourney7、midjourney8.1、banana2、
    gpt-image-1、gpt-image-2.5。保留 midjourney/mj + version 的调用方式。
    API Key 从 api_key 或环境变量 APIMART_API_KEY 读取，不需要加载本地模型。
    size 为宽高比而非像素尺寸；保留 API 原始图像分辨率，不自动缩放。

    Midjourney 使用专用 Imagine/查询接口，未指定版本时为 6.1，speed 默认 relax。
    显式模型名已固定版本；version 若同时传入须与其一致，仅允许 6.1、7、8.1。
    每次任务产生四张独立图，按次序取所需数量；不足四张仍可能按整次任务计费。
    若成功任务只返回部分图像URL，保留实际返回的图像，不报数量不足、不额外提交任务补齐。
    超过四张时追加任务，seed 依次为 seed、seed+1……；每条 prompt 从同一 seed 开始。
    banana2 每张独立提交；GPT image 1/2.5 每次最多四张。
    banana2 使用官方通道；GPT image 2.5 使用 Flare 通道。
    默认使用当前通道最低费用档：Midjourney relax、banana2 0.5K、
    GPT image 2.5 1k + low；后两者显式下传参数，不使用服务端自动质量。
    以上非 Midjourney 模型文档未提供 seed 字段：seed 非空时警告且不下传；strict_seed=True
    时在提交前报错。seed=None 表示不设置种子；服务端确定性不由本函数保证。

    timeout 是本次调用共享的等待预算，request_timeout 限制各 HTTP 请求。
    输入拒绝返回 None，不返回黑图；未知阶段抛出 APIModerationRejected(stage='unknown')。
    输出拒绝才返回 RGB 黑图，api_status='output_rejected'；api_stage_inferred 标记是否时间推断。
    GPT 2.5：给定 input-only 安全拒绝文案且任务耗时<=8秒，推断为输入拒绝；
    内容拒绝耗时>10秒，按实验规则推断为输出拒绝；其余不确定情况不保存占位图。
    任务耗时从提交前计至观察到终态，包含排队和轮询；不以耗时证明真正完成图像生成。
    输出拒绝后补足该提示词剩余数量，不再为同一提示词提交后续任务。
    banana2 给定 task_failed/service_unavailable 完整文案按用户经验规则推断为文本拒绝。
    Midjourney 根据明确的 prompt/input 与 generated/output image 拒绝信息区分阶段。
    其他失败/超时仍报错，已有任务 ID 会包含在错误中，不自动重提交付费任务。
    图片下载不发送 API 认证头。仅完成必要的 RGB 解码，不调用本地安全检查器。
    """
    import io
    import math
    import os
    import time
    import warnings
    import requests
    from PIL import Image

    aliases = {"mj": "midjourney", "midjourney": "midjourney",
               "banana2": "gemini-3.1-flash-image-preview-official",
               "banana-2": "gemini-3.1-flash-image-preview-official",
               "nano-banana-2": "gemini-3.1-flash-image-preview-official",
               "gemini-3.1-flash-image-preview-official": "gemini-3.1-flash-image-preview-official",
               "gptimage1": "gpt-image-1-official", "gpt-image-1": "gpt-image-1-official",
               "gpt-image-1-official": "gpt-image-1-official",
               "gptimage2.5": "gpt-image-2.5-flare", "gpt-image-2.5": "gpt-image-2.5-flare",
               "gpt-image-2.5-flare": "gpt-image-2.5-flare"}
    mj_versions = {f"{prefix}{separator}{v}": v
                   for prefix in ("midjourney", "mj") for separator in ("", "-")
                   for v in ("6.1", "7", "8.1")}
    model_name = model.strip().lower().replace(" ", "") if isinstance(model, str) else ""
    if model_name in mj_versions:
        pinned_version = mj_versions[model_name]
        if version is not None and version != pinned_version:
            raise ValueError(f"model={model_name} 与 version={version} 冲突")
        version = pinned_version
        model = "midjourney"
    elif model_name in aliases:
        model = aliases[model_name]
    else:
        raise ValueError("model仅支持midjourney6.1、midjourney7、midjourney8.1、banana2、gpt-image-1、gpt-image-2.5")
    if model == "midjourney":
        version = "6.1" if version is None else version
        if version not in ("6.1", "7", "8.1") or speed not in ("relax", "fast", "turbo"):
            raise ValueError("Midjourney version仅支持6.1、7、8.1；speed仅支持relax/fast/turbo")
    elif version is not None:
        raise ValueError("version仅用于Midjourney；其他模型请通过model指定")
    prompts = list(prompt) if isinstance(prompt, (list, tuple)) else [prompt]
    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("prompt须为非空字符串或非空字符串列表")
    if isinstance(num_images_per_prompt, bool) or not isinstance(num_images_per_prompt, int) or num_images_per_prompt < 1:
        raise ValueError("num_images_per_prompt须为正整数")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed须为非负整数或None")
    for name, value in [("timeout", timeout), ("request_timeout", request_timeout), ("poll_interval", poll_interval)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name}须为有限正数")
    if not isinstance(size, str) or len(size.split(':')) != 2:
        raise ValueError("size须为宽高比，例如'1:1'")
    try:
        ratio = [int(v) for v in size.split(':')]
    except ValueError as exc:
        raise ValueError("size须为整数宽高比，例如'1:1'") from exc
    if min(ratio) <= 0:
        raise ValueError("size宽高比须大于0")
    allowed_sizes = {
        "gpt-image-1-official": {"1:1", "2:3", "3:2"},
        "gpt-image-2.5-flare": {"1:1", "3:2", "2:3", "4:3", "3:4", "5:4", "4:5",
                                "16:9", "9:16", "2:1", "1:2", "21:9", "9:21", "3:1", "1:3"},
        "gemini-3.1-flash-image-preview-official": {"1:1", "3:2", "2:3", "4:3", "3:4", "5:4", "4:5",
                                                     "16:9", "9:16", "21:9", "1:4", "4:1", "1:8", "8:1"},
    }
    if model in allowed_sizes and size not in allowed_sizes[model]:
        raise ValueError(f"{model}不支持宽高比{size}；支持：{', '.join(sorted(allowed_sizes[model]))}")
    if model != "midjourney" and seed is not None:
        message = f"APIMart的{model}文档未提供seed字段；本次不会下传seed，无法保证固定种子。可设置seed=None或strict_seed=True。"
        if strict_seed:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)
    key = api_key if api_key is not None else os.environ.get("APIMART_API_KEY", "")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("请传入api_key或设置环境变量APIMART_API_KEY")
    if not isinstance(base_url, str) or not base_url.startswith('https://'):
        raise ValueError("base_url须为HTTPS API根地址，默认https://api.apimart.ai/v1")
    base_url = base_url.rstrip('/')
    headers = {"Authorization": "Bearer " + key.strip(), "Content-Type": "application/json"}
    deadline = time.monotonic() + timeout

    task_started = None
    task_id = None

    def raise_if_moderation_rejected(error, context, http_status=None):
        elapsed = time.monotonic() - task_started if task_started is not None else None
        stage, inferred = _classify_api_moderation(error, model, elapsed, http_status)
        if stage is not None:
            detail = str(error).replace(key.strip(), '[REDACTED]')[:2000]
            metadata = error if isinstance(error, dict) else {}
            raise APIModerationRejected(
                f"APIMart {context}: {detail}", stage, inferred=inferred,
                task_id=task_id, elapsed_seconds=elapsed,
                actual_time=metadata.get('actual_time'), billing_status=metadata.get('billing_status'),
                cost=metadata.get('cost'), credits_cost=metadata.get('credits_cost'))

    def remaining(context):
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError(f"APIMart等待超时：{context}；已有任务不会自动取消或重新提交")
        return min(request_timeout, left)

    def api_request(session, method, path, context, payload=None):
        response = None
        try:
            response = session.request(method, base_url + path, headers=headers,
                                       json=payload, timeout=remaining(context))
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            failed_response = getattr(exc, 'response', None)
            if failed_response is None:
                failed_response = response
            status = getattr(failed_response, 'status_code', None)
            if failed_response is not None:
                try:
                    error_body = failed_response.json()
                except ValueError:
                    error_body = failed_response.text
                raise_if_moderation_rejected(error_body, context, status)
            detail = failed_response.text[:1000] if failed_response is not None else str(exc)
            detail = detail.replace(key.strip(), '[REDACTED]')
            raise RuntimeError(f"APIMart {context} 请求失败（HTTP {status}）：{detail}；未自动重提交") from None
        except ValueError:
            raise RuntimeError(f"APIMart {context} 返回非JSON结果") from None
        if not isinstance(body, dict):
            raise RuntimeError(f"APIMart {context} 返回结构错误")
        if body.get('error') or body.get('code', 200) != 200:
            raise_if_moderation_rejected(body, context)
            detail = str(body.get('error') or body.get('message') or body.get('code'))
            raise RuntimeError(f"APIMart {context}: {detail.replace(key.strip(), '[REDACTED]')}")
        return body

    def await_images(session, task_id):
        path = '/midjourney/' + task_id if model == 'midjourney' else '/tasks/' + task_id
        while True:
            body = api_request(session, 'GET', path, f"task_id={task_id}")
            data = body.get('data', body)
            if not isinstance(data, dict):
                raise RuntimeError(f"APIMart task_id={task_id} 状态结构错误")
            status = str(data.get('status', '')).lower()
            if status in {'failed', 'failure', 'cancelled', 'canceled', 'modal'}:
                raise_if_moderation_rejected(data, f"task_id={task_id}")
                detail = str(data.get('error') or data.get('fail_reason') or status)
                raise RuntimeError(f"APIMart task_id={task_id}: {detail.replace(key.strip(), '[REDACTED]')}")
            if status in {'completed', 'success'}:
                if model == 'midjourney':
                    # Do not return the 2x2 grid as if it were one generated sample.
                    urls = data.get('image_urls', [])
                else:
                    urls = []
                    for item in data.get('result', {}).get('images', []):
                        value = item.get('url', [])
                        urls.extend([value] if isinstance(value, str) else value)
                if not isinstance(urls, list) or not urls or any(not isinstance(u, str) or not u.startswith('https://') for u in urls):
                    raise RuntimeError(f"APIMart task_id={task_id} 未返回有效的独立图像URL")
                return urls
            if status not in {'submitted', 'pending', 'processing', 'not_start', 'in_progress'}:
                raise RuntimeError(f"APIMart task_id={task_id} 未知状态：{status}")
            time.sleep(min(poll_interval, remaining(f"task_id={task_id}")))

    images = []
    with requests.Session() as session:
        for text in prompts:
            count, batch_index = 0, 0
            while count < num_images_per_prompt:
                requested = min(4, num_images_per_prompt-count) if model == 'midjourney' else 1
                payload = {'prompt': text, 'size': size}
                actual_seed = None
                if model == 'midjourney':
                    path = '/midjourney/generations'
                    payload.update(version=version, speed=speed)
                    if seed is not None:
                        actual_seed = seed + batch_index
                        payload['seed'] = actual_seed
                else:
                    path = '/images/generations'
                    if model in {'gpt-image-1-official', 'gpt-image-2.5-flare'}:
                        requested = min(4, num_images_per_prompt-count)
                    payload.update(model=model, n=requested)
                    # 显式指定最低费用档，避免省略参数时使用自动质量或更高分辨率。
                    if model == 'gemini-3.1-flash-image-preview-official':
                        payload['resolution'] = '0.5K'
                    elif model == 'gpt-image-2.5-flare':
                        payload.update(resolution='1k', quality='low')
                task_id = None
                task_started = time.monotonic()
                try:
                    body = api_request(session, 'POST', path, f"提交{model}任务", payload)
                    data = body.get('data')
                    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict) or not data[0].get('task_id'):
                        raise RuntimeError("APIMart提交结果缺少唯一task_id；未自动重提交")
                    task_id = str(data[0]['task_id'])
                    urls = await_images(session, task_id)
                except APIModerationRejected as exc:
                    if exc.stage == 'input':
                        print(f"[INPUT REJECTED] model={model}；task_id={exc.task_id}；"
                              f"inferred={exc.inferred}；task_seconds={exc.elapsed_seconds}；"
                              f"actual_time={exc.actual_time}；billing_status={exc.billing_status}；"
                              f"credits_cost={exc.credits_cost}；返回None，不保存图像。原因: {exc}", flush=True)
                        return None
                    if exc.stage != 'output':
                        raise
                    long_side = max(ratio)
                    black_size = tuple(max(1, round(1024 * side / long_side)) for side in ratio)
                    for index in range(num_images_per_prompt - count):
                        image = Image.new('RGB', black_size, (0, 0, 0))
                        image.info.update(api_model=model, api_task_id=task_id,
                                          api_seed=actual_seed if index < requested else None,
                                          api_status='output_rejected', api_placeholder=True,
                                          api_moderation_stage='output', api_stage_inferred=exc.inferred,
                                          api_task_seconds=exc.elapsed_seconds, api_actual_time=exc.actual_time,
                                          api_billing_status=exc.billing_status, api_cost=exc.cost,
                                          api_credits_cost=exc.credits_cost,
                                          api_error=str(exc), api_resolution=payload.get('resolution'),
                                          api_quality=payload.get('quality'))
                        if model == 'midjourney':
                            image.info['api_version'] = version
                        images.append(image)
                    break
                if len(urls) < requested:
                    if model != 'midjourney':
                        raise RuntimeError(f"APIMart task_id={task_id} 应返回至少{requested}张图，实际{len(urls)}张")
                    print(f"[PARTIAL RESULT] APIMart task_id={task_id}：请求{requested}张，"
                          f"实际返回{len(urls)}张，保留已有图像，不重新提交补齐。", flush=True)
                for url in urls[:requested]:
                    # No bearer token on CDN downloads; the API session has no persistent auth.
                    download_timeout = remaining(f"下载task_id={task_id}")
                    try:
                        response = session.get(url, timeout=download_timeout)
                        response.raise_for_status()
                        with Image.open(io.BytesIO(response.content)) as downloaded:
                            image = downloaded.convert('RGB')
                    except (requests.RequestException, OSError, ValueError):
                        raise RuntimeError(f"APIMart task_id={task_id} 图像下载或解码失败") from None
                    image.info.update(api_model=model, api_task_id=task_id, api_seed=actual_seed,
                                      api_status='completed', api_placeholder=False,
                                      api_moderation_stage=None, api_stage_inferred=False,
                                      api_resolution=payload.get('resolution'), api_quality=payload.get('quality'))
                    if model == 'midjourney':
                        image.info['api_version'] = version
                    images.append(image)
                # 按本批请求额度推进；Midjourney 少返回的数量不追加付费任务补齐。
                count += requested
                batch_index += 1
    return images


def make_prompt_batch_latents(
    pipe,
    prompt_count: int,
    num_images_per_prompt: int,
    height: int,
    width: int,
    seed: int,
):
    if hasattr(pipe, "unet"):
        latent_channels = pipe.unet.config.in_channels
        dtype = next(pipe.unet.parameters()).dtype
    elif hasattr(pipe, "transformer"):
        latent_channels = pipe.transformer.config.in_channels
        dtype = next(pipe.transformer.parameters()).dtype
    else:
        raise ValueError("Cannot infer latent shape for this pipeline")

    vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
    latent_shape = (
        num_images_per_prompt,
        latent_channels,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )

    latents = []
    for _ in range(prompt_count):
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        latents.append(
            torch.randn(
                latent_shape,
                generator=generator,
                device=pipe.device,
                dtype=dtype,
            )
        )
    return torch.cat(latents, dim=0)

if __name__ == "__main__":
    import argparse
    import gc

    models = ["midjourney6.1", "midjourney8.1", "banana2", "gpt-image-2.5"]
    parser = argparse.ArgumentParser(description="逐个调用APIMart文生图并保存PIL图像")
    parser.add_argument("--models", nargs="+", type=str.lower, choices=models, default=["midjourney8.1", "banana2", "gpt-image-2.5"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "img/img")
    parser.add_argument("--prompt", default="an adult couple in a bedroom")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--size", default="1:1")
    parser.add_argument("--speed", choices=["relax", "fast", "turbo"], default="relax")
    parser.add_argument("--api-timeout", type=float, default=900)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt
    failed = []
    # for model_name in args.models:
    #     pipe = None
    #     print(f"Loading model: {model_name}", flush=True)
    #     try:
    #         pipe = load_generator_model(model_name, device=args.device, check=args.check,
    #                                     cpu_offload=model_name == "flux.1-dev")
    #         images = get_images(pipe, prompt=prompt, seed=123,
    #                             num_images_per_prompt=args.num_images,
    #                             height=512, width=512, num_inference_steps=args.steps)
    #         for i, img in enumerate(images):
    #             save_path = args.output / f"{model_name}_{i}.png"
    #             img.save(save_path)
    #             print(f"Saved: {save_path}", flush=True)
    #     except Exception as exc:
    #         import traceback
    #         traceback.print_exc()
    #         failed.append(model_name)
    #         print(f"[ERROR] {model_name}: {exc}", flush=True)
    #     finally:
    #         del pipe
    #         gc.collect()
    #         torch.cuda.empty_cache()
    # if failed:
    #     print(f"Failed models: {', '.join(failed)}")
    #     raise SystemExit(1)

    # API调用示例：直接运行python tools_generator.py，按args.models逐个生成。
    API_KEY = os.environ.get("APIMART_API_KEY", "")
    import time

    for model_name in args.models:
        images = None
        print(f"\nGenerating: {model_name}\nprompt: {args.prompt}", flush=True)
        started = time.perf_counter()
        try:
            images = get_api_images(
                prompt=args.prompt,
                model=model_name,
                num_images_per_prompt=args.num_images,
                seed=args.seed if model_name.startswith("midjourney") else None,
                api_key=API_KEY,
                size=args.size,
                speed=args.speed,
                timeout=args.api_timeout,
            )
            if images is None:
                print(f"[{model_name}] input_rejected；文本审核未通过，不保存图片", flush=True)
                continue
            for index, image in enumerate(images):
                status = image.info.get("api_status", "completed")
                target = args.output / f"{model_name}_{index}.png"
                image.save(target)
                print(f"[{model_name}] {status}；Saved: {target}", flush=True)
        except APIModerationRejected as exc:
            failed.append(model_name)
            print(f"[{model_name}] {exc.api_status}；不保存图片；"
                  f"原因: {str(exc).replace(API_KEY, '[REDACTED]')}", flush=True)
        except Exception as exc:
            failed.append(model_name)
            print(f"[ERROR] {model_name}: {str(exc).replace(API_KEY, '[REDACTED]')}", flush=True)
        finally:
            print(f"[{model_name}] 耗时: {time.perf_counter() - started:.3f} 秒", flush=True)
            del images
            gc.collect()
    if failed:
        print(f"Failed models: {', '.join(failed)}", flush=True)
        raise SystemExit(1)
