from __future__ import annotations

import os

import gc
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Iterable

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

import tools_filter as tf
from tools_filter import (
    load_detector_model,
    get_prediction,
    load_classifier_model,
    text_classifier,
)

try:
    import torch
except ImportError:
    torch = None


# =========================
# 配置
# =========================
ROOT_DIR = os.environ.get("STYLEPROMPT_RAW_DATA_ROOT", "./raw_data")
JSONL_PATH = "./experiment_outputs/prompt_level_records_streaming_8w.jsonl"

# 临时目录放到 tools_filter.py 同级目录下，适配 Q16 的相对路径拼接逻辑
TOOLS_FILTER_DIR = Path(tf.__file__).resolve().parent
TEMP_ROOT = TOOLS_FILTER_DIR / "runtime_filter_tmp"

# 按显存情况自己调
BATCH_SIZE_Q16 = 64
BATCH_SIZE_CLIP = 32
BATCH_SIZE_NUDENET = 64
BATCH_SIZE_BERT = 128
BATCH_SIZE_LATENTGUARD = 64
BATCH_SIZE_SHIELDGEMMA = 8

SAVE_EVERY_N_BATCHES = 10
FAILED_IMAGE_LOG_PATH = "./experiment_outputs/failed_images.jsonl"
def init_failed_image_stats() -> Dict[str, Any]:
    return {
        "failed_sample_ids": set(),
        "failed_records": [],
    }


def record_failed_image(
    failed_stats: Dict[str, Any],
    record: Dict[str, Any],
    abs_path: str,
    error: Exception,
) -> None:
    sample_id = record.get("sample_id")
    if sample_id not in failed_stats["failed_sample_ids"]:
        failed_stats["failed_sample_ids"].add(sample_id)
        failed_stats["failed_records"].append({
            "sample_id": sample_id,
            "image_path": record.get("image_path"),
            "abs_image_path": abs_path,
            "error": repr(error),
        })


def save_failed_image_log(failed_stats: Dict[str, Any], path: str = FAILED_IMAGE_LOG_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for item in failed_stats["failed_records"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
def robust_open_image_rgb(path: str) -> Image.Image:
    """
    尽量鲁棒地打开图像并转成 RGB：
    1) PIL 正常读取
    2) PIL 截断容忍
    3) OpenCV 兜底
    """
    # 1) PIL 正常读取
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception:
        pass

    # 2) PIL 截断容忍
    old_flag = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(path) as img:
            img.load()
            return img.convert("RGB")
    except Exception:
        pass
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = old_flag

    # 3) OpenCV 兜底
    try:
        import cv2
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
    except Exception:
        pass

    raise OSError(f"无法读取或恢复图像: {path}")

# =========================
# 基本 IO
# =========================
def load_jsonl_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到文件: {path}")

    with p.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] 跳过非法 JSON 行: line={line_idx}")
    return records


def save_jsonl_atomic(records: List[Dict[str, Any]], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(p.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    tmp_path.replace(p)


# =========================
# 记录结构辅助
# =========================
def ensure_filter_schema(record: Dict[str, Any]) -> None:
    if "filter_output" not in record or not isinstance(record["filter_output"], dict):
        record["filter_output"] = {}

    for key in ["Q16", "CLIP", "NudeNet", "Bert", "latentguard", "shieldgemma"]:
        record["filter_output"].setdefault(key, None)

    if "nsfw_result" not in record or not isinstance(record["nsfw_result"], dict):
        record["nsfw_result"] = {"image": None, "text": None}
    else:
        record["nsfw_result"].setdefault("image", None)
        record["nsfw_result"].setdefault("text", None)


def resolve_abs_image_path(root_dir: str, image_path: str) -> str:
    return root_dir.rstrip("/") + image_path


def batched(items: List[int], batch_size: int) -> Iterable[List[int]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# =========================
# 图像批处理辅助
# =========================
def make_safe_token(text: Any, max_len: int = 24) -> str:
    """
    将任意内容转成适合放进文件名的短 token。
    """
    s = str(text) if text is not None else "none"
    s = s.strip()
    allowed = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_"}:
            allowed.append(ch)
        else:
            allowed.append("_")
    s = "".join(allowed)
    s = s.strip("_")
    if not s:
        s = "none"
    return s[:max_len]
def materialize_images_to_temp_dir(
    batch_records: List[Dict[str, Any]],
    batch_idxs: List[int],
    root_dir: str,
    failed_stats: Dict[str, Any],
    temp_root: Path = TEMP_ROOT,
) -> Tuple[str, List[int]]:
    """
    将当前 batch 的图片尽量鲁棒地转成 PNG，放入固定 temp 目录。
    自动跳过坏图，并返回真正保留下来的 records 索引。
    """
    temp_dir = temp_root / "temp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    kept_batch_idxs: List[int] = []

    for rec_idx in batch_idxs:
        record = batch_records[rec_idx]
        abs_path = resolve_abs_image_path(root_dir, record["image_path"])
        dst_path = temp_dir / f"{len(kept_batch_idxs):06d}.png"

        try:
            img = robust_open_image_rgb(abs_path)
            img.save(dst_path, format="PNG")
            kept_batch_idxs.append(rec_idx)
        except Exception as e:
            print(f"[WARN] 跳过坏图 rec_idx={rec_idx}, sample_id={record.get('sample_id')}, path={abs_path}, error={e}")
            record_failed_image(failed_stats, record, abs_path, e)

    rel = temp_dir.relative_to(TOOLS_FILTER_DIR)
    return "./" + rel.as_posix(), kept_batch_idxs


def load_images_as_numpy(
    abs_image_paths: List[str],
    target_size=(224, 224),
) -> np.ndarray:
    images = []
    for p in abs_image_paths:
        img = Image.open(p).convert("RGB")
        img = img.resize(target_size, Image.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 255.0
        images.append(arr)
    return np.stack(images, axis=0)


def normalize_single_detector_predictions(pred: Any, n: int) -> List[Any]:
    arr = np.asarray(pred)
    arr = np.squeeze(arr)

    if arr.ndim == 0:
        return [float(arr)] * n

    vals = arr.reshape(-1).tolist()

    if len(vals) == n:
        return vals

    if n == 1 and len(vals) >= 1:
        return [vals[0]]

    raise ValueError(f"无法解析单检测器输出形状: shape={arr.shape}, n={n}")


def cleanup_model_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================
# 各检测器批量填充
# =========================
def fill_q16(
    records: List[Dict[str, Any]],
    root_dir: str,
    jsonl_path: str,
    failed_stats: Dict[str, Any],
    batch_size: int = BATCH_SIZE_Q16,
) -> None:
    target = "Q16"
    pending = [i for i, r in enumerate(records) if r["filter_output"].get(target) is None]
    print(f"[INFO] {target} 待处理数量: {len(pending)}")
    if not pending:
        return

    load_detector_model(label="violence", method="both")

    for batch_no, batch_idxs in enumerate(batched(pending, batch_size), start=1):
        temp_dir_rel, kept_batch_idxs = materialize_images_to_temp_dir(
            batch_records=records,
            batch_idxs=batch_idxs,
            root_dir=root_dir,
            failed_stats=failed_stats,
        )

        if not kept_batch_idxs:
            print(f"[{target}] batch {batch_no} 全是坏图，跳过")
            continue

        pred = get_prediction(
            images_input=None,
            label="violence",
            l=len(kept_batch_idxs),
            path=temp_dir_rel,
        )

        pred = pred.detach().cpu().numpy()
        scores = normalize_single_detector_predictions(pred, len(kept_batch_idxs))

        for rec_idx, score in zip(kept_batch_idxs, scores):
            records[rec_idx]["filter_output"][target] = score

        print(f"[{target}] batch {batch_no} 完成，成功样本数={len(kept_batch_idxs)}")

        if batch_no % SAVE_EVERY_N_BATCHES == 0:
            save_jsonl_atomic(records, jsonl_path)
            print(f"[{target}] checkpoint 已保存")

        cleanup_model_memory()

    save_jsonl_atomic(records, jsonl_path)
    print(f"[{target}] 全部完成并保存")


def fill_clip(
    records: List[Dict[str, Any]],
    root_dir: str,
    jsonl_path: str,
    failed_stats: Dict[str, Any],
    batch_size: int = BATCH_SIZE_CLIP,
) -> None:
    target = "CLIP"
    pending = [i for i, r in enumerate(records) if r["filter_output"].get(target) is None]
    print(f"[INFO] {target} 待处理数量: {len(pending)}")
    if not pending:
        return

    load_detector_model(label="sex", method="clip")

    for batch_no, batch_idxs in enumerate(batched(pending, batch_size), start=1):
        valid_batch_idxs = []
        valid_images = []

        for rec_idx in batch_idxs:
            record = records[rec_idx]
            abs_path = resolve_abs_image_path(root_dir, record["image_path"])
            try:
                img = robust_open_image_rgb(abs_path)
                img = img.resize((224, 224), Image.BICUBIC)
                arr = np.asarray(img).astype(np.float32) / 255.0
                valid_batch_idxs.append(rec_idx)
                valid_images.append(arr)
            except Exception as e:
                print(f"[WARN] 跳过坏图 rec_idx={rec_idx}, sample_id={record.get('sample_id')}, path={abs_path}, error={e}")
                record_failed_image(failed_stats, record, abs_path, e)

        if not valid_batch_idxs:
            print(f"[{target}] batch {batch_no} 全是坏图，跳过")
            continue

        images_np = np.stack(valid_images, axis=0)

        pred = get_prediction(
            images_input=images_np,
            label="sex",
            l=len(valid_batch_idxs),
            method="clip",
        )

        pred = pred.detach().cpu().numpy()
        scores = normalize_single_detector_predictions(pred, len(valid_batch_idxs))

        for rec_idx, score in zip(valid_batch_idxs, scores):
            records[rec_idx]["filter_output"][target] = score

        print(f"[{target}] batch {batch_no} 完成，成功样本数={len(valid_batch_idxs)}")

        if batch_no % SAVE_EVERY_N_BATCHES == 0:
            save_jsonl_atomic(records, jsonl_path)
            print(f"[{target}] checkpoint 已保存")

        cleanup_model_memory()

    save_jsonl_atomic(records, jsonl_path)
    print(f"[{target}] 全部完成并保存")


def fill_nudenet(
    records: List[Dict[str, Any]],
    root_dir: str,
    jsonl_path: str,
    failed_stats: Dict[str, Any],
    batch_size: int = BATCH_SIZE_NUDENET,
) -> None:
    target = "NudeNet"
    pending = [i for i, r in enumerate(records) if r["filter_output"].get(target) is None]
    print(f"[INFO] {target} 待处理数量: {len(pending)}")
    if not pending:
        return

    load_detector_model(label="sex", method="nudenet")

    for batch_no, batch_idxs in enumerate(batched(pending, batch_size), start=1):
        temp_dir_rel, kept_batch_idxs = materialize_images_to_temp_dir(
            batch_records=records,
            batch_idxs=batch_idxs,
            root_dir=root_dir,
            failed_stats=failed_stats,
        )

        if not kept_batch_idxs:
            print(f"[{target}] batch {batch_no} 全是坏图，跳过")
            continue

        pred = get_prediction(
            images_input=temp_dir_rel,
            label="sex",
            l=len(kept_batch_idxs),
            method="nudenet",
        )

        pred = pred.detach().cpu().numpy()
        scores = normalize_single_detector_predictions(pred, len(kept_batch_idxs))

        for rec_idx, score in zip(kept_batch_idxs, scores):
            records[rec_idx]["filter_output"][target] = score

        print(f"[{target}] batch {batch_no} 完成，成功样本数={len(kept_batch_idxs)}")

        if batch_no % SAVE_EVERY_N_BATCHES == 0:
            save_jsonl_atomic(records, jsonl_path)
            print(f"[{target}] checkpoint 已保存")

        cleanup_model_memory()

    save_jsonl_atomic(records, jsonl_path)
    print(f"[{target}] 全部完成并保存")


def fill_text_detector(
    records: List[Dict[str, Any]],
    jsonl_path: str,
    target_key: str,
    method: str,
    batch_size: int,
) -> None:
    pending = [i for i, r in enumerate(records) if r["filter_output"].get(target_key) is None]
    print(f"[INFO] {target_key} 待处理数量: {len(pending)}")
    if not pending:
        return

    load_classifier_model(method=method)

    for batch_no, batch_idxs in enumerate(batched(pending, batch_size), start=1):
        prompts = [records[i]["raw_prompt"] for i in batch_idxs]
        scores = text_classifier(prompts, method=method)

        if len(scores) != len(batch_idxs):
            raise ValueError(f"{target_key} 输出长度不匹配: {len(scores)} vs {len(batch_idxs)}")

        for rec_idx, score in zip(batch_idxs, scores):
            records[rec_idx]["filter_output"][target_key] = score

        print(f"[{target_key}] batch {batch_no} 完成，样本数={len(batch_idxs)}")

        if batch_no % SAVE_EVERY_N_BATCHES == 0:
            save_jsonl_atomic(records, jsonl_path)
            print(f"[{target_key}] checkpoint 已保存")

        cleanup_model_memory()

    save_jsonl_atomic(records, jsonl_path)
    print(f"[{target_key}] 全部完成并保存")


# =========================
# 主流程
# =========================
def main():
    records = load_jsonl_records(JSONL_PATH)
    print(f"[INFO] 读取记录数: {len(records)}")

    for r in records:
        ensure_filter_schema(r)

    failed_stats = init_failed_image_stats()

    # 一次只跑一个检测器
    fill_q16(records, ROOT_DIR, JSONL_PATH, failed_stats, batch_size=BATCH_SIZE_Q16)
    fill_clip(records, ROOT_DIR, JSONL_PATH, failed_stats, batch_size=BATCH_SIZE_CLIP)
    fill_nudenet(records, ROOT_DIR, JSONL_PATH, failed_stats, batch_size=BATCH_SIZE_NUDENET)

    fill_text_detector(records, JSONL_PATH, target_key="Bert", method="bert", batch_size=BATCH_SIZE_BERT)
    fill_text_detector(records, JSONL_PATH, target_key="latentguard", method="latentguard", batch_size=BATCH_SIZE_LATENTGUARD)
    fill_text_detector(records, JSONL_PATH, target_key="shieldgemma", method="shieldgemma", batch_size=BATCH_SIZE_SHIELDGEMMA)

    save_failed_image_log(failed_stats, FAILED_IMAGE_LOG_PATH)

    print("[INFO] 全部检测器填充完成")
    print(f"[INFO] 读取失败的唯一图像数量: {len(failed_stats['failed_sample_ids'])}")
    print(f"[INFO] 失败图像明细已保存到: {FAILED_IMAGE_LOG_PATH}")


if __name__ == "__main__":
    main()
