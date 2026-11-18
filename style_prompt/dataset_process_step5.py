from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Union, Set

from tools_prompt import (
    build_online_llm_client,
    llm_score_effect_profiles_one_prompt,
    classify_granularity_one,
)

# =========================
# 配置
# =========================
INPUT_JSONL_PATH = "./experiment_outputs/fragment_level_records.jsonl"
OUTPUT_JSONL_PATH = "./experiment_outputs/fragment_level_records.jsonl"
FAILED_JSONL_PATH = "./experiment_outputs/failed_fragment_classification.jsonl"

MODEL_NAME = "gpt-4o"

# effect_profile 建议批量跑，节省 API 次数
BATCH_SIZE = 8
SAVE_EVERY_N_BATCHES = 20

EFFECT_PROFILE_FIELDS = [
    "artistic_context",
    "realism",
    "stylization",
    "atmospheric_intensity",
    "sensory_salience",
    "platform_association",
    "depiction_explicitness",
    "violence_graphicness",
]

VALID_GRANULARITY = {
    "token",
    "short_phrase",
    "long_phrase",
}


# =========================
# 基本 IO
# =========================
def load_jsonl_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

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


def save_jsonl_atomic(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    tmp.replace(p)


# =========================
# schema / 校验
# =========================
def is_valid_effect_profile(value: Any) -> bool:
    if not isinstance(value, dict):
        return False

    for field in EFFECT_PROFILE_FIELDS:
        if field not in value:
            return False
        x = value[field]
        if not isinstance(x, (int, float)):
            return False
        x = float(x)
        if x < 0.0 or x > 1.0:
            return False

    return True


def is_valid_granularity(value: Any) -> bool:
    return isinstance(value, str) and value in VALID_GRANULARITY


def ensure_fragment_score_schema(record: Dict[str, Any]) -> None:
    if "effect_profile" not in record or not is_valid_effect_profile(record.get("effect_profile")):
        record["effect_profile"] = None

    if "granularity" not in record or not is_valid_granularity(record.get("granularity")):
        record["granularity"] = None


def needs_scoring(record: Dict[str, Any]) -> bool:
    return (not is_valid_effect_profile(record.get("effect_profile"))) or (not is_valid_granularity(record.get("granularity")))


# =========================
# 断点续跑
# =========================
def load_processed_fragment_ids(path: Union[str, Path]) -> Set[str]:
    p = Path(path)
    if not p.exists():
        return set()

    processed: Set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                fid = item.get("fragment_id")
                if isinstance(fid, str) and fid and is_valid_effect_profile(item.get("effect_profile")) and is_valid_granularity(item.get("granularity")):
                    processed.add(fid)
            except Exception:
                continue
    return processed


def merge_input_with_existing(
    input_record: Dict[str, Any],
    existing_record: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    用输入记录作为主干，若已有旧输出，则把已算好的字段合并回来。
    """
    merged = dict(input_record)

    if existing_record is not None:
        if "effect_profile" in existing_record:
            merged["effect_profile"] = existing_record["effect_profile"]
        if "granularity" in existing_record:
            merged["granularity"] = existing_record["granularity"]

    ensure_fragment_score_schema(merged)
    return merged


# =========================
# 批处理
# =========================
def batched(items: List[int], batch_size: int) -> List[List[int]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def process_one_batch(
    records: List[Dict[str, Any]],
    batch_idxs: List[int],
    client,
    model_name: str,
) -> None:
    fragments = [records[i]["raw_descriptor"] for i in batch_idxs]

    profiles = llm_score_effect_profiles_one_prompt(
        fragments=fragments,
        client=client,
        model=model_name,
    )

    if len(profiles) != len(batch_idxs):
        raise ValueError(f"effect_profile 返回长度不一致: {len(profiles)} vs {len(batch_idxs)}")

    for rec_idx, fragment, profile in zip(batch_idxs, fragments, profiles):
        records[rec_idx]["effect_profile"] = profile
        records[rec_idx]["granularity"] = classify_granularity_one(fragment)


def process_all_records(
    input_jsonl_path: str,
    output_jsonl_path: str,
    failed_jsonl_path: str,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    save_every_n_batches: int = SAVE_EVERY_N_BATCHES,
) -> None:
    input_records = load_jsonl_records(input_jsonl_path)
    print(f"[INFO] 读取 fragment-level 记录数: {len(input_records)}")

    existing_output_records: List[Dict[str, Any]] = []
    if Path(output_jsonl_path).exists():
        existing_output_records = load_jsonl_records(output_jsonl_path)

    failed_records: List[Dict[str, Any]] = []
    if Path(failed_jsonl_path).exists():
        failed_records = load_jsonl_records(failed_jsonl_path)

    existing_map: Dict[str, Dict[str, Any]] = {}
    for item in existing_output_records:
        fid = item.get("fragment_id")
        if isinstance(fid, str) and fid:
            existing_map[fid] = item

    merged_records: List[Dict[str, Any]] = []
    for record in input_records:
        fid = record.get("fragment_id")
        existing_record = existing_map.get(fid) if isinstance(fid, str) else None
        merged = merge_input_with_existing(record, existing_record)
        merged_records.append(merged)

    processed_fragment_ids = load_processed_fragment_ids(output_jsonl_path)
    pending_idxs: List[int] = []

    for idx, record in enumerate(merged_records):
        fid = record.get("fragment_id")
        if not isinstance(fid, str) or not fid.strip():
            print(f"[WARN] idx={idx} 缺少有效 fragment_id，跳过")
            continue
        if fid in processed_fragment_ids:
            continue
        if needs_scoring(record):
            pending_idxs.append(idx)

    print(f"[INFO] 待处理 fragment 数量: {len(pending_idxs)}")
    if not pending_idxs:
        print("[INFO] 没有新的 fragment 需要处理")
        return

    client = build_online_llm_client()
    batches = batched(pending_idxs, batch_size)

    success_count = 0
    fail_count = 0

    for batch_no, batch_idxs in enumerate(batches, start=1):
        try:
            process_one_batch(
                records=merged_records,
                batch_idxs=batch_idxs,
                client=client,
                model_name=model_name,
            )
            success_count += len(batch_idxs)
            print(f"[DONE] batch {batch_no}/{len(batches)} 完成，处理 {len(batch_idxs)} 条")

        except Exception as e:
            print(f"[WARN] batch {batch_no}/{len(batches)} 批量处理失败，转为逐条回退。error={repr(e)}")
            for rec_idx in batch_idxs:
                record = merged_records[rec_idx]
                try:
                    process_one_batch(
                        records=merged_records,
                        batch_idxs=[rec_idx],
                        client=client,
                        model_name=model_name,
                    )
                    success_count += 1
                    print(f"[DONE] fragment_id={record.get('fragment_id')} 单条处理成功")
                except Exception as sub_e:
                    fail_count += 1
                    failed_records.append({
                        "fragment_id": record.get("fragment_id"),
                        "sample_id": record.get("sample_id"),
                        "raw_descriptor": record.get("raw_descriptor"),
                        "error": repr(sub_e),
                    })
                    print(f"[WARN] fragment_id={record.get('fragment_id')} 单条处理失败，error={repr(sub_e)}")

        if batch_no % save_every_n_batches == 0:
            save_jsonl_atomic(merged_records, output_jsonl_path)
            save_jsonl_atomic(failed_records, failed_jsonl_path)
            print(f"[INFO] checkpoint 已保存：batch {batch_no}/{len(batches)}")

    save_jsonl_atomic(merged_records, output_jsonl_path)
    save_jsonl_atomic(failed_records, failed_jsonl_path)

    print("[INFO] fragment effect_profile + granularity 处理完成")
    print(f"[INFO] 成功处理数量: {success_count}")
    print(f"[INFO] 失败数量: {fail_count}")
    print(f"[INFO] 输出文件: {output_jsonl_path}")
    print(f"[INFO] 失败日志: {failed_jsonl_path}")


def main() -> None:
    process_all_records(
        input_jsonl_path=INPUT_JSONL_PATH,
        output_jsonl_path=OUTPUT_JSONL_PATH,
        failed_jsonl_path=FAILED_JSONL_PATH,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        save_every_n_batches=SAVE_EVERY_N_BATCHES,
    )


if __name__ == "__main__":
    main()