from __future__ import annotations

import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

# =========================
# 配置
# =========================
INPUT_JSONL_PATH = "./experiment_outputs/fragment_level_records.jsonl"
OUTPUT_JSONL_PATH = "./experiment_outputs/fragment_level_records.jsonl"

INVALID_JSONL_PATH = "./experiment_outputs/fragment_level_invalid_step1_enhanced.jsonl"
SUMMARY_JSON_PATH = "./experiment_outputs/fragment_level_step1_enhanced_summary.json"

MECHANISM_CATEGORIES = {
    "reference_anchor",
    "rendering_medium",
    "representational_regime",
    "affective_atmosphere",
    "spatial_composition",
    "perceptual_cue",
    "contextual_frame",
    "platform_context",
}

VALID_GRANULARITY = {
    "token",
    "short_phrase",
    "long_phrase",
}

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

# 脏数据过滤阈值
MAX_DESCRIPTOR_CHARS = 300
MAX_DESCRIPTOR_TOKENS = 40


# =========================
# 基本 IO
# =========================
def save_jsonl_atomic(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(p.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    tmp_path.replace(p)


def save_json_atomic(obj: Any, path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(p.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    tmp_path.replace(p)


# =========================
# 文本规范化
# =========================
def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_quotes_dashes(text: str) -> str:
    replacements = {
        "“": "\"",
        "”": "\"",
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "−": "-",
        "…": "...",
        "\u00a0": " ",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    return text


def normalize_raw_descriptor(text: str) -> str:
    """
    加强版第一步：
    1. unicode 规范化
    2. html 反转义
    3. 特殊引号/破折号归一
    4. 全部转小写
    5. 把 _ 替换成空格
    6. 去掉多余空格
    """
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = normalize_quotes_dashes(text)
    text = text.replace("_", " ")
    text = text.lower()
    text = normalize_spaces(text)
    return text


def tokenize_descriptor(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", text)


def looks_like_url(text: str) -> bool:
    return bool(re.search(r"(https?://|www\.)", text, flags=re.I))


def looks_like_html(text: str) -> bool:
    return bool(re.search(r"<[^>]+>", text))


def has_any_alnum(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def is_obviously_dirty_descriptor(text: str) -> Optional[str]:
    """
    返回 None 表示看起来正常；
    返回字符串表示判定为脏数据的原因。
    """
    if not text:
        return "empty_descriptor"

    if len(text) > MAX_DESCRIPTOR_CHARS:
        return "too_long_chars"

    tokens = tokenize_descriptor(text)
    if len(tokens) == 0:
        return "no_valid_tokens"

    if len(tokens) > MAX_DESCRIPTOR_TOKENS:
        return "too_long_tokens"

    if not has_any_alnum(text):
        return "no_alnum"

    if looks_like_url(text):
        return "looks_like_url"

    if looks_like_html(text):
        return "looks_like_html"

    if re.fullmatch(r"[_\-\.\,\:\;\|\+\=\*/\\\s]+", text):
        return "only_punctuation"

    if re.search(r"(.)\1{10,}", text):
        return "excessive_repeated_chars"

    return None


# =========================
# 字段校验
# =========================
def validate_effect_profile(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, dict):
        return None

    clean_profile: Dict[str, float] = {}

    for field in EFFECT_PROFILE_FIELDS:
        if field not in value:
            return None

        x = value[field]
        if not isinstance(x, (int, float)):
            return None

        x = float(x)
        if x < 0.0 or x > 1.0:
            return None

        clean_profile[field] = x

    return clean_profile


def validate_and_clean_record(record: Dict[str, Any]) -> (Optional[Dict[str, Any]], List[str]):
    """
    返回：
        clean_record, issues
    不合法时 clean_record 为 None
    """
    issues: List[str] = []

    required_fields = [
        "sample_id",
        "fragment_id",
        "raw_descriptor",
        "mechanism_category",
        "effect_profile",
        "granularity",
    ]

    for field in required_fields:
        if field not in record:
            return None, [f"missing_field:{field}"]

    sample_id = record["sample_id"]
    fragment_id = record["fragment_id"]
    raw_descriptor = record["raw_descriptor"]
    mechanism_category = record["mechanism_category"]
    effect_profile = record["effect_profile"]
    granularity = record["granularity"]

    if not isinstance(sample_id, str) or not sample_id.strip():
        return None, ["invalid_sample_id"]

    if not isinstance(fragment_id, str) or not fragment_id.strip():
        return None, ["invalid_fragment_id"]

    if not isinstance(raw_descriptor, str):
        return None, ["invalid_raw_descriptor_type"]

    raw_descriptor_clean = normalize_raw_descriptor(raw_descriptor)
    dirty_reason = is_obviously_dirty_descriptor(raw_descriptor_clean)
    if dirty_reason is not None:
        return None, [dirty_reason]

    if not isinstance(mechanism_category, str) or mechanism_category not in MECHANISM_CATEGORIES:
        return None, ["invalid_mechanism_category"]

    clean_profile = validate_effect_profile(effect_profile)
    if clean_profile is None:
        return None, ["invalid_effect_profile"]

    if not isinstance(granularity, str) or granularity not in VALID_GRANULARITY:
        return None, ["invalid_granularity"]

    clean_record = dict(record)
    clean_record["sample_id"] = sample_id.strip()
    clean_record["fragment_id"] = fragment_id.strip()
    clean_record["raw_descriptor"] = raw_descriptor_clean
    clean_record["effect_profile"] = clean_profile
    clean_record["mechanism_category"] = mechanism_category
    clean_record["granularity"] = granularity

    return clean_record, issues


# =========================
# 主流程
# =========================
def main() -> None:
    input_path = Path(INPUT_JSONL_PATH)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {INPUT_JSONL_PATH}")

    valid_records: List[Dict[str, Any]] = []
    invalid_records: List[Dict[str, Any]] = []

    total_lines = 0
    json_parse_fail_count = 0
    field_validation_fail_count = 0
    issue_counter = Counter()

    with input_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            total_lines += 1
            line = line.strip()

            if not line:
                invalid_records.append({
                    "line_index": line_idx,
                    "error_type": "empty_line",
                    "raw_line": line,
                })
                field_validation_fail_count += 1
                issue_counter["empty_line"] += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_records.append({
                    "line_index": line_idx,
                    "error_type": "json_decode_error",
                    "raw_line": line,
                })
                json_parse_fail_count += 1
                issue_counter["json_decode_error"] += 1
                continue

            clean_record, issues = validate_and_clean_record(record)
            if clean_record is None:
                invalid_records.append({
                    "line_index": line_idx,
                    "error_type": "field_validation_error",
                    "issues": issues,
                    "record": record,
                })
                field_validation_fail_count += 1
                for issue in issues:
                    issue_counter[issue] += 1
                continue

            valid_records.append(clean_record)

    save_jsonl_atomic(valid_records, OUTPUT_JSONL_PATH)
    save_jsonl_atomic(invalid_records, INVALID_JSONL_PATH)

    summary = {
        "input_path": INPUT_JSONL_PATH,
        "output_path": OUTPUT_JSONL_PATH,
        "invalid_output_path": INVALID_JSONL_PATH,
        "total_lines": total_lines,
        "valid_records": len(valid_records),
        "invalid_records": len(invalid_records),
        "json_parse_fail_count": json_parse_fail_count,
        "field_validation_fail_count": field_validation_fail_count,
        "issue_counter": dict(issue_counter),
        "max_descriptor_chars": MAX_DESCRIPTOR_CHARS,
        "max_descriptor_tokens": MAX_DESCRIPTOR_TOKENS,
    }
    save_json_atomic(summary, SUMMARY_JSON_PATH)

    print("[INFO] 加强版第一步清洗完成")
    print(f"[INFO] 总行数: {total_lines}")
    print(f"[INFO] 合法记录数: {len(valid_records)}")
    print(f"[INFO] 非法记录数: {len(invalid_records)}")
    print(f"[INFO] JSON解析失败数: {json_parse_fail_count}")
    print(f"[INFO] 字段校验失败数: {field_validation_fail_count}")
    print(f"[INFO] 清洗后输出: {OUTPUT_JSONL_PATH}")
    print(f"[INFO] 非法记录输出: {INVALID_JSONL_PATH}")
    print(f"[INFO] 摘要输出: {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    main()