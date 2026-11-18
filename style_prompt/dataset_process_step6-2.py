#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从原始 fragment_level_records.jsonl 直接执行：
Step 1: 按 fragment_id 去重
Step 2: 按 raw_descriptor 精确分组，并对重复 raw_descriptor 进行合并

输入：
    ./experiment_outputs/fragment_level_records.jsonl

输出：
    ./experiment_outputs/dedup_step2_descriptor_merge/
    ├── raw_descriptor_no_duplicate.jsonl
    ├── raw_descriptor_duplicate_original.jsonl
    ├── raw_descriptor_duplicate_merged.jsonl
    └── fragment_level_records.descriptor_recombined.jsonl

处理逻辑：
    1. 先按 fragment_id 去重：
       - 同一个 fragment_id 只保留第一次出现的记录
       - 后续重复 fragment_id 不进入 raw_descriptor 分组

    2. 再按 raw_descriptor 分组：
       - 完全没有重复的 raw_descriptor：原样写入 raw_descriptor_no_duplicate.jsonl
       - 有重复的 raw_descriptor：所有原始记录原样写入 raw_descriptor_duplicate_original.jsonl
       - 有重复的 raw_descriptor：合并后写入 raw_descriptor_duplicate_merged.jsonl

    3. 最后重组：
       - no_duplicate + duplicate_merged
       - 输出 fragment_level_records.descriptor_recombined.jsonl

合并规则：
    sample_id:
        全部保留为 list

    effect_profile:
        每个维度分别取众数

    其他字段：
        取众数

    众数平票：
        取最早出现的值
"""

import os
import re
import json
from collections import defaultdict, Counter
from typing import Any, Dict, List


# =========================
# 配置区域
# =========================

INPUT_JSONL_PATH = "./experiment_outputs/fragment_level_records_step6_1.jsonl"

OUTPUT_DIR = "./experiment_outputs/dedup_step2_descriptor_merge"

NO_DUPLICATE_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "raw_descriptor_no_duplicate.jsonl"
)

DUPLICATE_ORIGINAL_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "raw_descriptor_duplicate_original.jsonl"
)

DUPLICATE_MERGED_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "raw_descriptor_duplicate_merged.jsonl"
)

RECOMBINED_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "fragment_level_records.descriptor_recombined.jsonl"
)

ADD_MERGE_META = True
STRICT_FRAGMENT_ID = True
STRICT_RAW_DESCRIPTOR = True


# =========================
# 工具函数
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def normalize_descriptor(text: Any) -> str:
    """
    用于 raw_descriptor 精确分组的轻量归一化。
    由于前置清洗已经完成，这里只做：
    strip + 合并多余空格 + casefold。
    """
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.casefold()
    return text


def normalize_display_text(text: Any) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def get_fragment_id(record: Dict[str, Any], line_no: int) -> str:
    fragment_id = record.get("fragment_id", None)

    if fragment_id is None or str(fragment_id).strip() == "":
        msg = f"Line {line_no}: missing or empty fragment_id"
        if STRICT_FRAGMENT_ID:
            raise ValueError(msg)
        return ""

    return str(fragment_id).strip()


def get_raw_descriptor(record: Dict[str, Any], line_no: int) -> str:
    raw_descriptor = record.get("raw_descriptor", None)

    if raw_descriptor is None or str(raw_descriptor).strip() == "":
        msg = f"Line {line_no}: missing or empty raw_descriptor"
        if STRICT_RAW_DESCRIPTOR:
            raise ValueError(msg)
        return ""

    return normalize_display_text(raw_descriptor)


def value_to_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def ordered_unique(values: List[Any]) -> List[Any]:
    seen = set()
    result = []

    for value in values:
        key = value_to_key(value)
        if key not in seen:
            seen.add(key)
            result.append(value)

    return result


def mode_value(values: List[Any]) -> Any:
    """
    取众数。
    如果多个值频次相同，返回最早出现的值。
    """
    counter = Counter()
    first_value = {}

    for value in values:
        key = value_to_key(value)
        counter[key] += 1

        if key not in first_value:
            first_value[key] = value

    if not counter:
        return None

    best_key = None
    best_count = -1

    for key, count in counter.items():
        if count > best_count:
            best_key = key
            best_count = count

    return first_value[best_key]


def count_values(values: List[Any]) -> Dict[str, int]:
    counter = Counter()

    for value in values:
        key = value_to_key(value)
        counter[key] += 1

    result = {}

    for key, count in counter.items():
        try:
            loaded = json.loads(key)
            display_key = str(loaded)
        except Exception:
            display_key = key

        result[display_key] = count

    return result


def merge_effect_profile(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    effect_profile 内部每个维度分别取众数。
    """
    dim_to_values = defaultdict(list)

    for record in records:
        effect_profile = record.get("effect_profile", {})

        if not isinstance(effect_profile, dict):
            continue

        for dim, value in effect_profile.items():
            dim_to_values[dim].append(value)

    merged_profile = {}

    for dim, values in dim_to_values.items():
        merged_profile[dim] = mode_value(values)

    return merged_profile


def count_effect_profile(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    dim_to_values = defaultdict(list)

    for record in records:
        effect_profile = record.get("effect_profile", {})

        if not isinstance(effect_profile, dict):
            continue

        for dim, value in effect_profile.items():
            dim_to_values[dim].append(value)

    result = {}

    for dim, values in dim_to_values.items():
        result[dim] = count_values(values)

    return result


# =========================
# Step 1 + Step 2 读取分组
# =========================

def load_from_original_and_group_by_raw_descriptor(input_path: str):
    """
    直接从原始 JSONL 读取：
    1. 先按 fragment_id 去重
    2. 再按 raw_descriptor 分组
    """
    groups = defaultdict(list)
    line_no_groups = defaultdict(list)

    seen_fragment_ids = set()

    total_lines = 0
    empty_lines = 0
    valid_json_records = 0
    invalid_json_lines = 0

    fragment_unique_records = 0
    duplicate_fragment_records = 0

    with open(input_path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            total_lines += 1

            raw_line = line.rstrip("\n")

            if not raw_line.strip():
                empty_lines += 1
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as e:
                invalid_json_lines += 1
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e

            valid_json_records += 1

            fragment_id = get_fragment_id(record, line_no)

            # Step 1: fragment_id 去重
            if fragment_id in seen_fragment_ids:
                duplicate_fragment_records += 1
                continue

            seen_fragment_ids.add(fragment_id)
            fragment_unique_records += 1

            # Step 2: raw_descriptor 分组
            raw_descriptor = get_raw_descriptor(record, line_no)
            descriptor_key = normalize_descriptor(raw_descriptor)

            groups[descriptor_key].append(record)
            line_no_groups[descriptor_key].append(line_no)

    stats = {
        "input_path": input_path,
        "total_lines": total_lines,
        "empty_lines": empty_lines,
        "valid_json_records": valid_json_records,
        "invalid_json_lines": invalid_json_lines,
        "fragment_unique_records": fragment_unique_records,
        "duplicate_fragment_records_removed": duplicate_fragment_records,
        "unique_raw_descriptor_count_after_fragment_dedup": len(groups),
    }

    return groups, line_no_groups, stats


# =========================
# 合并重复 raw_descriptor 组
# =========================

def merge_duplicate_group(
    descriptor_key: str,
    records: List[Dict[str, Any]],
    line_nos: List[int],
) -> Dict[str, Any]:
    """
    合并同一个 raw_descriptor 下的多条记录。

    合并规则：
    - sample_id：全部保留为 list
    - effect_profile：内部每个维度取众数
    - 其他字段：取众数
    """
    all_keys = set()
    for record in records:
        all_keys.update(record.keys())

    merged = {}

    # 1. sample_id 全部保留
    sample_ids = ordered_unique([
        record.get("sample_id")
        for record in records
        if record.get("sample_id") is not None
    ])
    merged["sample_id"] = sample_ids

    # 2. effect_profile 每个维度分别取众数
    if "effect_profile" in all_keys:
        merged["effect_profile"] = merge_effect_profile(records)

    # 3. 其他字段取众数
    special_keys = {
        "sample_id",
        "effect_profile",
    }

    for key in sorted(all_keys):
        if key in special_keys:
            continue

        values = [
            record.get(key)
            for record in records
            if key in record
        ]

        merged[key] = mode_value(values)

    # 4. 添加合并元信息
    if ADD_MERGE_META:
        raw_descriptor_values = [
            normalize_display_text(record.get("raw_descriptor", ""))
            for record in records
        ]

        mechanism_category_values = [
            record.get("mechanism_category")
            for record in records
            if "mechanism_category" in record
        ]

        granularity_values = [
            record.get("granularity")
            for record in records
            if "granularity" in record
        ]

        fragment_id_values = [
            record.get("fragment_id")
            for record in records
            if "fragment_id" in record
        ]

        merged["_merge_meta"] = {
            "merged": True,
            "reason": "duplicate_raw_descriptor_merged",
            "descriptor_key": descriptor_key,
            "record_count": len(records),
            "source_line_nos_in_original_file": line_nos,

            "sample_ids": sample_ids,
            "fragment_ids": ordered_unique(fragment_id_values),

            "raw_descriptor_counts": count_values(raw_descriptor_values),
            "mechanism_category_counts": count_values(mechanism_category_values),
            "granularity_counts": count_values(granularity_values),
            "fragment_id_counts": count_values(fragment_id_values),
            "effect_profile_value_counts": count_effect_profile(records),

            "merge_policy": {
                "sample_id": "keep_all_unique_values_as_list",
                "effect_profile": "mode_value_per_dimension",
                "other_fields": "mode_value",
                "tie_break": "first_occurrence",
            },
        }

    return merged


# =========================
# 主流程
# =========================

def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 80)
    print("Step 1 + Step 2: 从原始文件读取，fragment_id 去重后按 raw_descriptor 聚合")
    print("=" * 80)
    print(f"输入文件: {INPUT_JSONL_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")

    groups, line_no_groups, stats = load_from_original_and_group_by_raw_descriptor(
        INPUT_JSONL_PATH
    )

    no_duplicate_records = []
    duplicate_original_records = []
    duplicate_merged_records = []

    no_duplicate_count = 0
    duplicate_group_count = 0
    duplicate_original_record_count = 0
    merged_record_count = 0

    for descriptor_key, records in groups.items():
        line_nos = line_no_groups[descriptor_key]

        if len(records) == 1:
            no_duplicate_records.append(records[0])
            no_duplicate_count += 1
        else:
            duplicate_original_records.extend(records)

            merged_record = merge_duplicate_group(
                descriptor_key=descriptor_key,
                records=records,
                line_nos=line_nos,
            )

            duplicate_merged_records.append(merged_record)

            duplicate_group_count += 1
            duplicate_original_record_count += len(records)
            merged_record_count += 1

    # 1. 完全没有重复 raw_descriptor 的记录
    with open(NO_DUPLICATE_OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for record in no_duplicate_records:
            fout.write(json_dumps(record) + "\n")

    # 2. 重复 raw_descriptor 的所有原始记录，原样保留
    with open(DUPLICATE_ORIGINAL_OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for record in duplicate_original_records:
            fout.write(json_dumps(record) + "\n")

    # 3. 重复 raw_descriptor 的合并结果
    with open(DUPLICATE_MERGED_OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for record in duplicate_merged_records:
            fout.write(json_dumps(record) + "\n")

    # 4. 重组后的最终文件
    with open(RECOMBINED_OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for record in no_duplicate_records:
            fout.write(json_dumps(record) + "\n")

        for record in duplicate_merged_records:
            fout.write(json_dumps(record) + "\n")

    print("=" * 80)
    print("处理完成")
    print("=" * 80)

    for key, value in stats.items():
        print(f"{key}: {value}")

    print("-" * 80)
    print(f"完全没有重复的 raw_descriptor 记录数: {no_duplicate_count}")
    print(f"存在重复的 raw_descriptor 组数: {duplicate_group_count}")
    print(f"重复 raw_descriptor 原始记录总数: {duplicate_original_record_count}")
    print(f"重复组被合并后的记录数: {merged_record_count}")
    print(f"重组后最终记录数: {no_duplicate_count + merged_record_count}")

    print("-" * 80)
    print("输出文件：")
    print(f"1. 完全没有重复的记录: {NO_DUPLICATE_OUTPUT_PATH}")
    print(f"2. 重复 raw_descriptor 原始记录: {DUPLICATE_ORIGINAL_OUTPUT_PATH}")
    print(f"3. 重复 raw_descriptor 合并结果: {DUPLICATE_MERGED_OUTPUT_PATH}")
    print(f"4. 重组后的最终文件: {RECOMBINED_OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()