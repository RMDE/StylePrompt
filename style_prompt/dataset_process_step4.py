from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Union, Set

from tools_prompt import (
    build_online_llm_client,
    llm_classify_mechanism_categories_one_prompt,
)

JSONL_PATH = "./experiment_outputs/prompt_level_records_streaming_8w.jsonl"
FRAGMENT_SOURCE_FIELD = "style_fragments"

OUTPUT_DIR = Path("./experiment_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_FRAGMENTS_TXT_PATH = OUTPUT_DIR / "all_fragments.txt"
FRAGMENT_LEVEL_JSONL_PATH = OUTPUT_DIR / "fragment_level_records.jsonl"
FAILED_PROMPTS_JSONL_PATH = OUTPUT_DIR / "failed_fragment_classification.jsonl"

MODEL_NAME = "gpt-4o"
SAVE_EVERY_N_PROMPTS = 100


def load_jsonl_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl_atomic(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp.replace(p)


def save_lines(path: Union[str, Path], items: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(x + "\n")


def normalize_fragment_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []

    out: List[str] = []
    seen = set()
    for x in value:
        if not isinstance(x, str):
            continue
        x = x.strip()
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def make_fragment_id(sample_id: str, frag_idx: int, raw_descriptor: str) -> str:
    """
    fragment_id 是 fragment 的唯一标识。
    这里用：
        prompt的sample_id + fragment在该prompt中的序号 + fragment内容
    共同生成稳定唯一 id
    """
    base = f"{sample_id}::{frag_idx}::{raw_descriptor}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def build_fragment_level_record(
    sample_id: str,
    fragment_id: str,
    raw_descriptor: str,
    mechanism_category: str,
) -> Dict[str, Any]:
    """
    sample_id: 所属 prompt 的 sample_id
    fragment_id: 当前 fragment 的唯一识别号
    raw_descriptor: 当前 fragment 的原始文本内容
    """
    return {
        "sample_id": sample_id,
        "fragment_id": fragment_id,
        "raw_descriptor": raw_descriptor,
        "mechanism_category": mechanism_category,
    }


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
                if isinstance(fid, str) and fid:
                    processed.add(fid)
            except Exception:
                continue
    return processed


def collect_all_fragments_txt(
    prompt_records: List[Dict[str, Any]],
    fragment_field: str,
    out_path: Union[str, Path]
) -> None:
    all_fragments: List[str] = []
    for record in prompt_records:
        fragments = normalize_fragment_list(record.get(fragment_field))
        all_fragments.extend(fragments)
    save_lines(out_path, all_fragments)


def process_prompt_records(
    prompt_records: List[Dict[str, Any]],
    fragment_field: str,
    output_jsonl_path: Union[str, Path],
    failed_jsonl_path: Union[str, Path],
    model_name: str = MODEL_NAME,
    save_every_n_prompts: int = SAVE_EVERY_N_PROMPTS,
) -> None:
    client = build_online_llm_client()

    existing_records: List[Dict[str, Any]] = []
    if Path(output_jsonl_path).exists():
        existing_records = load_jsonl_records(str(output_jsonl_path))

    processed_fragment_ids = load_processed_fragment_ids(output_jsonl_path)

    failed_records: List[Dict[str, Any]] = []
    if Path(failed_jsonl_path).exists():
        failed_records = load_jsonl_records(str(failed_jsonl_path))

    fragment_level_records = existing_records[:]

    total_prompts = len(prompt_records)
    new_fragment_count = 0

    for prompt_idx, record in enumerate(prompt_records, start=1):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            print(f"[WARN] prompt {prompt_idx} 缺少有效 sample_id，跳过")
            continue

        fragments = normalize_fragment_list(record.get(fragment_field))
        if not fragments:
            continue

        fragment_ids = [
            make_fragment_id(sample_id=sample_id, frag_idx=i, raw_descriptor=frag)
            for i, frag in enumerate(fragments)
        ]

        if all(fid in processed_fragment_ids for fid in fragment_ids):
            print(f"[SKIP] prompt {prompt_idx}/{total_prompts} sample_id={sample_id} 已处理，跳过")
            continue

        try:
            categories = llm_classify_mechanism_categories_one_prompt(
                fragments=fragments,
                client=client,
                model=model_name,
            )

            if len(categories) != len(fragments):
                raise ValueError(f"分类结果长度不一致: {len(categories)} vs {len(fragments)}")

            added_this_prompt = 0
            for frag_idx, (frag, fid, cat) in enumerate(zip(fragments, fragment_ids, categories)):
                if fid in processed_fragment_ids:
                    continue

                item = build_fragment_level_record(
                    sample_id=sample_id,
                    fragment_id=fid,
                    raw_descriptor=frag,
                    mechanism_category=cat,
                )
                fragment_level_records.append(item)
                processed_fragment_ids.add(fid)
                new_fragment_count += 1
                added_this_prompt += 1

            print(f"[DONE] prompt {prompt_idx}/{total_prompts} sample_id={sample_id} 新增 {added_this_prompt} 条 fragment")

        except Exception as e:
            failed_records.append({
                "sample_id": sample_id,
                "prompt_index": prompt_idx,
                "fragments": fragments,
                "error": repr(e),
            })
            print(f"[WARN] prompt {prompt_idx}/{total_prompts} 分类失败，sample_id={sample_id}，error={repr(e)}")

        if prompt_idx % save_every_n_prompts == 0:
            save_jsonl_atomic(fragment_level_records, output_jsonl_path)
            save_jsonl_atomic(failed_records, failed_jsonl_path)
            print(f"[INFO] checkpoint 已保存：prompt_idx={prompt_idx}")

    save_jsonl_atomic(fragment_level_records, output_jsonl_path)
    save_jsonl_atomic(failed_records, failed_jsonl_path)

    print(f"[INFO] fragment-level 处理完成，新增 fragment 数量: {new_fragment_count}")
    print(f"[INFO] 输出文件: {output_jsonl_path}")
    print(f"[INFO] 失败日志: {failed_jsonl_path}")


def main() -> None:
    prompt_records = load_jsonl_records(JSONL_PATH)
    print(f"[INFO] 读取 prompt-level 记录数: {len(prompt_records)}")

    collect_all_fragments_txt(
        prompt_records=prompt_records,
        fragment_field=FRAGMENT_SOURCE_FIELD,
        out_path=ALL_FRAGMENTS_TXT_PATH,
    )
    print(f"[INFO] 所有 fragments 已保存到: {ALL_FRAGMENTS_TXT_PATH}")

    process_prompt_records(
        prompt_records=prompt_records,
        fragment_field=FRAGMENT_SOURCE_FIELD,
        output_jsonl_path=FRAGMENT_LEVEL_JSONL_PATH,
        failed_jsonl_path=FAILED_PROMPTS_JSONL_PATH,
        model_name=MODEL_NAME,
        save_every_n_prompts=SAVE_EVERY_N_PROMPTS,
    )


if __name__ == "__main__":
    main()