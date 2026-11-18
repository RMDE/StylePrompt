from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Union

INPUT_JSONL_PATH = "./experiment_outputs/fragment_level_records.jsonl"
OUTPUT_JSONL_PATH = "./experiment_outputs/fragment_level_records.jsonl"
REMOVED_JSONL_PATH = "./experiment_outputs/removed_meaningless_fragments.jsonl"
SUMMARY_JSON_PATH = "./experiment_outputs/remove_meaningless_fragments_summary.json"


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


def load_jsonl_records(path: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(path))

    records = []
    with p.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_idx}: {e}") from e
    return records


def is_meaningless_fragment(text: str) -> bool:
    """
    删除规则：

    1. 不含任何英文字母 -> 删除
    2. 只有极少数字母，且整体主要由数字/符号构成 -> 删除

    删除：
        1
        1.2
        16:9
        ---
        #2024
        90°
        x1
        v2
        a-3
        b/12
        z.9

    保留：
        f2.8
        35mm
        iso 100
        4k wallpaper
        x-ray
        bw photo
        vray render
    """
    if not isinstance(text, str):
        return False

    text = text.strip().lower()
    if not text:
        return True

    letters = re.findall(r"[a-z]", text)
    alnum = re.findall(r"[a-z0-9]", text)

    # 没有字母，直接删
    if len(letters) == 0:
        return True

    # 只有 1 个字母且整体很短：删
    if len(letters) == 1 and len(alnum) <= 4:
        # 保留常见有意义相机参数形式，如 f2.8
        whitelist_patterns = [
            r"^[a-z]\d+\.\d+$",   # f2.8
            r"^[a-z]\d+[a-z]+$",  # 4k? 这里不会匹配，因为以数字开头；保留做扩展
        ]
        for pat in whitelist_patterns:
            if re.fullmatch(pat, text):
                return False
        return True

    # 只有 2 个字母、整体很短、且没有空格：通常也是噪声
    if len(letters) == 2 and len(alnum) <= 4 and " " not in text:
        whitelist_patterns = [
            r"^\d+[a-z]{2,}$",      # 35mm
            r"^[a-z]\d+\.\d+$",     # f2.8
            r"^[a-z]+-[a-z]+$",     # x-ray
            r"^[a-z]{2,}\d+$",      # vray2
        ]
        for pat in whitelist_patterns:
            if re.fullmatch(pat, text):
                return False
        return True

    return False


def main() -> None:
    records = load_jsonl_records(INPUT_JSONL_PATH)

    kept_records = []
    removed_records = []

    for rec in records:
        desc = rec.get("raw_descriptor", "")
        if is_meaningless_fragment(desc):
            removed_records.append(rec)
        else:
            kept_records.append(rec)

    save_jsonl_atomic(kept_records, OUTPUT_JSONL_PATH)
    save_jsonl_atomic(removed_records, REMOVED_JSONL_PATH)

    summary = {
        "input_path": INPUT_JSONL_PATH,
        "output_path": OUTPUT_JSONL_PATH,
        "removed_output_path": REMOVED_JSONL_PATH,
        "original_count": len(records),
        "kept_count": len(kept_records),
        "removed_count": len(removed_records),
        "rule": "remove fragments with no alphabetic letters, or fragments with very few letters and dominated by numbers/symbols",
    }
    save_json_atomic(summary, SUMMARY_JSON_PATH)

    print("[INFO] 无意义 fragment 过滤完成")
    print(f"[INFO] 原始记录数: {len(records)}")
    print(f"[INFO] 保留记录数: {len(kept_records)}")
    print(f"[INFO] 删除记录数: {len(removed_records)}")
    print(f"[INFO] 删除记录保存到: {REMOVED_JSONL_PATH}")
    print(f"[INFO] 摘要保存到: {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    main()