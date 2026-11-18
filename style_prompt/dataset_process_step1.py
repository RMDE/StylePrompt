# 对采集到的原始数据集进行预处理（逐条处理 / 逐条落盘）
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import json
import random
from pathlib import Path
from typing import List, Dict, Iterator, Optional, Any, Tuple, Set

from tools_prompt import *


# =========================
# 1. 提取基础记录
# =========================
PLATFORM_DIRS = [
    "civitai_crawler",
    "Ideogram_crawler",
    "lexica_crawler",
    "prompthero_crawler",
]
def find_master_jsonl_files(root_dir: str) -> List[str]:
    """
    递归查找 root_dir 下所有 records/manifests/master.jsonl 文件。
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root_dir}")

    matched = []
    for p in root.rglob("master.jsonl"):
        norm_parts = [x.lower() for x in p.parts]
        if "records" in norm_parts and "manifests" in norm_parts:
            matched.append(str(p.resolve()))

    return sorted(matched)
def find_master_jsonl_files_by_platform(
    root_dir: str,
    platform_dirs: List[str],
) -> Dict[str, List[str]]:
    """
    在 root_dir 下，按平台子目录分别查找 records/manifests/master.jsonl。
    返回：
        {
            "civitai_crawler": [...],
            "Ideogram_crawler": [...],
            ...
        }
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root_dir}")

    result: Dict[str, List[str]] = {p: [] for p in platform_dirs}

    for p in root.rglob("master.jsonl"):
        try:
            rel_parts = p.resolve().relative_to(root.resolve()).parts
        except Exception:
            continue

        if len(rel_parts) < 1:
            continue

        platform_dir = rel_parts[0]
        if platform_dir not in result:
            continue

        norm_parts = [x.lower() for x in p.parts]
        if "records" in norm_parts and "manifests" in norm_parts:
            result[platform_dir].append(str(p.resolve()))

    for k in result:
        result[k] = sorted(result[k])

    return result
def extract_prompt_records_by_platform(
    root_dir: str,
    platform_dirs: List[str],
) -> Dict[str, List[Dict]]:
    """
    按平台目录分别提取记录。
    返回：
        {
            "civitai_crawler": [record1, record2, ...],
            ...
        }
    """
    platform_to_files = find_master_jsonl_files_by_platform(root_dir, platform_dirs)

    platform_to_records: Dict[str, List[Dict]] = {p: [] for p in platform_dirs}

    for platform_dir, master_files in platform_to_files.items():
        for master_file in master_files:
            for item in iter_records_from_master_jsonl(master_file, root_dir):
                platform_to_records[platform_dir].append(item)

    return platform_to_records
def sample_records_per_platform(
    platform_to_records: Dict[str, List[Dict]],
    sample_size: int = 20000,
    sample_mode: str = "random",
    random_seed: int = 42,
) -> List[Dict]:
    """
    对每个平台分别：
    1. 去重
    2. 过滤空 prompt
    3. 按 sample_mode 抽 sample_size 条
    4. 最后合并

    sample_mode:
        - "head": 取前 sample_size 条
        - "random": 随机抽 sample_size 条

    若某个平台不足 sample_size，则保留全部并打印提示。
    """
    rng = random.Random(random_seed)
    final_records: List[Dict] = []

    for platform_dir, records in platform_to_records.items():
        print(f"[INFO] 平台 {platform_dir} 原始记录数: {len(records)}")

        records = deduplicate_records(records)
        print(f"[INFO] 平台 {platform_dir} 去重后记录数: {len(records)}")

        records = filter_valid_records(records)
        print(f"[INFO] 平台 {platform_dir} 过滤空 prompt 后记录数: {len(records)}")

        if len(records) <= sample_size:
            print(f"[WARN] 平台 {platform_dir} 记录不足 {sample_size}，将保留全部 {len(records)} 条")
            sampled = records
        else:
            if sample_mode == "head":
                sampled = records[:sample_size]
                print(f"[INFO] 平台 {platform_dir} 取前 {sample_size} 条")
            elif sample_mode == "random":
                sampled = rng.sample(records, sample_size)
                print(f"[INFO] 平台 {platform_dir} 随机抽取 {sample_size} 条")
            else:
                raise ValueError(f"不支持的 sample_mode: {sample_mode}")

        final_records.extend(sampled)

    print(f"[INFO] 四个平台合并后总记录数: {len(final_records)}")
    return final_records
def make_relative_image_path(image_local_path: str, root_dir: str) -> str:
    """
    将 image_local_path 转成相对于 root_dir 的相对路径，
    并强制前面加一个 '/'，便于后续用 root_dir + image_path 还原。
    """
    if not image_local_path:
        return ""

    root = Path(root_dir).resolve()
    anchor_name = root.name

    raw = image_local_path.replace("\\", "/").strip()
    parts = [p for p in raw.split("/") if p]

    rel_path = ""

    if anchor_name in parts:
        idx = parts.index(anchor_name)
        rel_parts = parts[idx + 1:]
        rel_path = "/".join(rel_parts)
    else:
        try:
            p = Path(image_local_path).resolve()
            rel = p.relative_to(root)
            rel_path = rel.as_posix()
        except Exception:
            rel_path = Path(raw).name

    if not rel_path.startswith("/"):
        rel_path = "/" + rel_path

    return rel_path


def parse_one_record(record: Dict, root_dir: str) -> Dict:
    """
    从原始记录中提取基础字段。
    """
    return {
        "sample_id": record.get("image_hash"),
        "raw_prompt": record.get("raw_prompt"),
        "image_path": make_relative_image_path(record.get("image_local_path") or "", root_dir),
        "model": record.get("model"),
        "platform": record.get("source_platform"),
    }


def iter_records_from_master_jsonl(master_jsonl_path: str, root_dir: str) -> Iterator[Dict]:
    """
    按行读取单个 master.jsonl，并逐条返回提取后的结果。
    跳过空行和非法 JSON 行。
    """
    path = Path(master_jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {master_jsonl_path}")

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] 跳过非法 JSON: {master_jsonl_path} 第 {line_idx} 行")
                continue

            yield parse_one_record(record, root_dir)


def extract_all_prompt_records(root_dir: str) -> List[Dict]:
    """
    递归查找所有 master.jsonl 并提取基础记录。
    """
    master_files = find_master_jsonl_files(root_dir)
    all_records: List[Dict] = []

    for master_file in master_files:
        for item in iter_records_from_master_jsonl(master_file, root_dir):
            all_records.append(item)

    return all_records


def deduplicate_records(records: List[Dict]) -> List[Dict]:
    """
    按 sample_id 去重。
    """
    seen = set()
    deduped = []

    for item in records:
        sid = item.get("sample_id")
        if sid and sid not in seen:
            seen.add(sid)
            deduped.append(item)

    return deduped


def filter_valid_records(records: List[Dict]) -> List[Dict]:
    """
    过滤掉没有 raw_prompt 的记录。
    """
    valid = []
    for item in records:
        raw_prompt = item.get("raw_prompt")
        if isinstance(raw_prompt, str) and raw_prompt.strip():
            valid.append(item)
    return valid


def select_records_for_trial(
    records: List[Dict],
    max_records: Optional[int] = None,
    sample_mode: str = "head",
    start_idx: int = 0,
    random_seed: int = 42,
) -> List[Dict]:
    """
    从 records 中选择一部分记录。
    """
    if max_records is None or max_records >= len(records):
        return records

    if max_records <= 0:
        return []

    if sample_mode == "head":
        return records[:max_records]

    elif sample_mode == "random":
        rng = random.Random(random_seed)
        indices = list(range(len(records)))
        rng.shuffle(indices)
        chosen = indices[:max_records]
        chosen.sort()
        return [records[i] for i in chosen]

    elif sample_mode == "slice":
        end_idx = min(start_idx + max_records, len(records))
        return records[start_idx:end_idx]

    else:
        raise ValueError(f"不支持的 sample_mode: {sample_mode}")


# =========================
# 2. 保存 / 断点续跑
# =========================

def append_record_to_jsonl(record: Dict[str, Any], output_path: str) -> None:
    """
    追加写入单条 jsonl 记录。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_records_to_jsonl(records: List[Dict[str, Any]], output_path: str) -> None:
    """
    整体写入 jsonl。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_processed_sample_ids(output_path: str) -> Set[str]:
    """
    从已存在的输出文件中读取已处理 sample_id，用于断点续跑。
    """
    path = Path(output_path)
    if not path.exists():
        return set()

    processed = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                sid = item.get("sample_id")
                if sid:
                    processed.add(sid)
            except Exception:
                continue
    return processed


# =========================
# 3. prompt-level 风格统计
# =========================

def count_tokens_from_texts(texts: List[str]) -> int:
    """
    使用 tokenize_prompts 规则统计 token 总数。
    """
    tokenized = tokenize_prompts(texts)
    return sum(len(tokens) for tokens in tokenized)


def build_prompt_style_statistics_one(
    prompt: str,
    final_frag_list: List[str],
    style_frags: List[str],
) -> Dict[str, Any]:
    """
    对单条 prompt 生成 prompt-level 风格统计字段。
    """
    num_total_tokens = count_tokens_from_texts([prompt])
    num_style_tokens = count_tokens_from_texts(style_frags)

    style_token_ratio = (
        num_style_tokens / num_total_tokens if num_total_tokens > 0 else 0.0
    )

    num_total_fragments = len(final_frag_list)
    num_style_fragments = len(style_frags)

    style_fragment_ratio = (
        num_style_fragments / num_total_fragments if num_total_fragments > 0 else 0.0
    )

    return {
        "num_total_tokens": num_total_tokens,
        "num_style_tokens": num_style_tokens,
        "style_token_ratio": style_token_ratio,
        "num_total_fragments": num_total_fragments,
        "num_style_fragments": num_style_fragments,
        "style_fragments": style_frags,
        "style_fragment_ratio": style_fragment_ratio,
    }


def merge_prompt_level_record(
    base_record: Dict[str, Any],
    prompt_style_stat: Dict[str, Any],
    filter_result: Dict[str, Any],
    similarity: float,
) -> Dict[str, Any]:
    """
    将基础记录、prompt-level 风格统计、filter 结果、最终 similarity
    merge 成单条记录。
    """
    return {
        "sample_id": base_record.get("sample_id"),
        "raw_prompt": base_record.get("raw_prompt"),
        "image_path": base_record.get("image_path"),
        "model": base_record.get("model"),
        "platform": base_record.get("platform"),

        "similarity": similarity,
        "num_total_tokens": prompt_style_stat.get("num_total_tokens"),
        "num_style_tokens": prompt_style_stat.get("num_style_tokens"),
        "style_token_ratio": prompt_style_stat.get("style_token_ratio"),
        "num_total_fragments": prompt_style_stat.get("num_total_fragments"),
        "num_style_fragments": prompt_style_stat.get("num_style_fragments"),
        "style_fragments": prompt_style_stat.get("style_fragments"),
        "style_fragment_ratio": prompt_style_stat.get("style_fragment_ratio"),

        "filter_output": filter_result.get("filter_output"),
        "nsfw_result": filter_result.get("nsfw_result"),
    }


# =========================
# 4. filter_output / nsfw_result
# 当前版本：不做任何检测，全部置为 None
# =========================

def build_filter_output_one() -> Dict[str, Any]:
    """
    当前版本不进行任何 image/text detection。
    为了保持输出字段结构稳定，统一返回 None。
    """
    return {
        "filter_output": {
            "Q16": None,
            "CLIP": None,
            "NudeNet": None,
            "Bert": None,
            "latentguard": None,
            "shieldgemma": None,
        },
        "nsfw_result": {
            "image": None,
            "text": None,
        }
    }


# =========================
# 5. 单条 prompt 的 fragment 划分（带阈值重试）
# =========================

def finalize_one_prompt_until_threshold(
    prompt: str,
    parser_results_one_prompt: List[Dict[str, Any]],
    client,
    model_name: str = "gpt-4o",
    similarity_threshold: float = 0.9,
    similarity_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    max_retries: Optional[int] = 5,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str], float, int]:
    """
    对单条 prompt 做 fragment 划分：
    - 先跑一次
    - 若语义相似度 < similarity_threshold，则继续重试
    - 直到相似度 >= similarity_threshold，或达到 max_retries
    """
    retry_count = 0

    final_result_one_prompt = llm_finalize_prompts(
        parser_results_2d=[parser_results_one_prompt],
        client=client,
        model=model_name
    )[0]

    final_frag_list_one_prompt = flatten_final_fragments([final_result_one_prompt])[0]

    best_similarity = compute_semantic_similarity_list(
        original_prompts=[prompt],
        fragment_lists_2d=[final_frag_list_one_prompt],
        model_name=similarity_model_name
    )[0]

    if verbose:
        print(f"[INIT] similarity = {best_similarity:.4f}")

    while best_similarity < similarity_threshold:
        if max_retries is not None and retry_count >= max_retries:
            if verbose:
                print(f"[STOP] 达到最大重试次数 max_retries={max_retries}，当前 best_similarity={best_similarity:.4f}")
            break

        retry_count += 1

        new_result_one_prompt = llm_finalize_prompts(
            parser_results_2d=[parser_results_one_prompt],
            client=client,
            model=model_name
        )[0]

        new_frag_list_one_prompt = flatten_final_fragments([new_result_one_prompt])[0]

        new_similarity = compute_semantic_similarity_list(
            original_prompts=[prompt],
            fragment_lists_2d=[new_frag_list_one_prompt],
            model_name=similarity_model_name
        )[0]

        if verbose:
            print(f"[RETRY {retry_count}] similarity = {new_similarity:.4f}")

        if new_similarity > best_similarity:
            final_result_one_prompt = new_result_one_prompt
            final_frag_list_one_prompt = new_frag_list_one_prompt
            best_similarity = new_similarity

        if best_similarity >= similarity_threshold:
            if verbose:
                print(f"[PASS] similarity 达标: {best_similarity:.4f}")
            break

    return final_result_one_prompt, final_frag_list_one_prompt, best_similarity, retry_count


# =========================
# 6. 单条记录完整处理
# =========================
_thread_local = threading.local()


def get_worker_resources() -> Tuple[Any, Any]:
    """
    为每个线程创建独立的 parser / client，避免线程间共享状态。
    注意：这里建议 build_constituency_parser() 使用 CPU 版本。
    """
    if not hasattr(_thread_local, "parser"):
        _thread_local.parser = build_constituency_parser(False)
    if not hasattr(_thread_local, "client"):
        _thread_local.client = build_online_llm_client()
    return _thread_local.parser, _thread_local.client


def process_one_record_worker(
    record: Dict[str, Any],
    root_dir: str,
    model_name: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    线程池 worker 调用入口。
    """
    parser, client = get_worker_resources()
    return process_one_record(
        record=record,
        root_dir=root_dir,
        parser=parser,
        client=client,
        model_name=model_name,
    )

def process_one_record(
    record: Dict[str, Any],
    root_dir: str,
    parser,
    client,
    model_name: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    对单条记录完成 prompt-level 全流程处理：
    - fragment 划分
    - style 识别
    - prompt-level 统计
    - merge
    """
    prompt = record["raw_prompt"]

    # 1. parser 候选
    coarse_chunks = coarse_split_prompts([prompt])
    candidate_results = parser_candidates_for_prompts(coarse_chunks, parser)

    # 2. fragment 划分（带相似度阈值重试）
    final_result_one_prompt, final_frag_list_one_prompt, best_similarity, retry_count = (
        finalize_one_prompt_until_threshold(
            prompt=prompt,
            parser_results_one_prompt=candidate_results[0],
            client=client,
            model_name=model_name,
            similarity_threshold=0.88,
            similarity_model_name="sentence-transformers/all-mpnet-base-v2",
            max_retries=0,
            verbose=True,
        )
    )

    # 3. 区分 content/style fragments
    _, style_2d = llm_split_content_and_style_fragments(
        fragments_2d=[final_frag_list_one_prompt],
        client=client,
        model=model_name
    )
    style_frags = style_2d[0]

    # 4. prompt-level 统计
    prompt_style_stat = build_prompt_style_statistics_one(
        prompt=prompt,
        final_frag_list=final_frag_list_one_prompt,
        style_frags=style_frags,
    )

    # 5. 当前版本不做 filter，只填默认空值
    filter_result = build_filter_output_one()

    # 6. merge
    merged_record = merge_prompt_level_record(
        base_record=record,
        prompt_style_stat=prompt_style_stat,
        filter_result=filter_result,
        similarity=best_similarity,
    )

    return merged_record


# =========================
# 7. 逐条处理并逐条写入
# =========================

def run_prompt_level_pipeline_streaming(
    root_dir: str,
    output_path: str,
    model_name: str = "gpt-4o",
    per_platform_sample_size: int = 20000,
    per_platform_sample_mode: str = "random",
    random_seed: int = 42,
    max_workers: int = 8,
    max_in_flight: int = 32,
) -> None:
    """
    并行逐条处理并逐条追加保存：
    - worker 线程并发处理样本
    - 主线程负责写文件
    - 支持断点续跑
    - max_workers: 并发 worker 数
    - max_in_flight: 同时在途任务上限，防止一次性提交过多 future
    """
    # 1. 提取基础记录
    platform_to_records = extract_prompt_records_by_platform(
        root_dir=root_dir,
        platform_dirs=PLATFORM_DIRS,
    )

    records = sample_records_per_platform(
        platform_to_records=platform_to_records,
        sample_size=per_platform_sample_size,
        sample_mode=per_platform_sample_mode,
        random_seed=random_seed,
    )

    print(f"[INFO] 本次待处理记录数: {len(records)}")

    # 2. 断点续跑
    processed_ids = load_processed_sample_ids(output_path)
    print(f"[INFO] 已存在输出中的 sample_id 数量: {len(processed_ids)}")

    # 只保留未处理样本
    pending_records = [r for r in records if r.get("sample_id") not in processed_ids]
    print(f"[INFO] 本次实际待处理样本数: {len(pending_records)}")

    if not pending_records:
        print("[INFO] 没有新的样本需要处理")
        return

    success_count = 0
    fail_count = 0

    # 3. 并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_meta = {}
        next_idx = 0
        total = len(pending_records)

        def submit_more():
            nonlocal next_idx
            while next_idx < total and len(future_to_meta) < max_in_flight:
                record = pending_records[next_idx]
                sid = record.get("sample_id")
                future = executor.submit(
                    process_one_record_worker,
                    record,
                    root_dir,
                    model_name,
                )
                future_to_meta[future] = {
                    "sample_id": sid,
                    "record_idx": next_idx + 1,
                    "total": total,
                }
                next_idx += 1

        submit_more()

        while future_to_meta:
            done, _ = wait(
                future_to_meta.keys(),
                return_when=FIRST_COMPLETED
            )

            for future in done:
                meta = future_to_meta.pop(future)
                sid = meta["sample_id"]
                idx = meta["record_idx"]
                total_num = meta["total"]

                print(f"\n[COLLECT] {idx}/{total_num} sample_id={sid}")

                try:
                    merged_record = future.result()

                    # 主线程统一写文件
                    append_record_to_jsonl(merged_record, output_path)
                    processed_ids.add(sid)
                    success_count += 1

                    print(f"[DONE] sample_id={sid} 已追加写入")

                except Exception as e:
                    fail_count += 1
                    print(f"[ERROR] sample_id={sid} 处理失败: {repr(e)}")

            # 补充提交新任务
            submit_more()

    print("\n[SUMMARY]")
    print(f"[INFO] 成功写入: {success_count}")
    print(f"[INFO] 失败数量: {fail_count}")
    print(f"[INFO] 输出文件: {output_path}")


# =========================
# 8. main
# =========================

if __name__ == "__main__":
    root_dir = os.environ.get("STYLEPROMPT_RAW_DATA_ROOT", "./raw_data")
    output_path = "./experiment_outputs/prompt_level_records_streaming_8w.jsonl"

    run_prompt_level_pipeline_streaming(
        root_dir=root_dir,
        output_path=output_path,
        model_name="gpt-4o",
        per_platform_sample_size=20000,
        per_platform_sample_mode= "random",
        random_seed=42,
        max_workers=32,
        max_in_flight=96,
    )

    print("\n[INFO] 已保存:")
    print(output_path)