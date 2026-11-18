#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3: raw_descriptor 高保守语义去重

目标：
    即使漏合并，也尽可能避免错合并。

输入：
    ./experiment_outputs/dedup_step2_descriptor_merge/
    fragment_level_records.descriptor_recombined.jsonl

输出目录：
    ./experiment_outputs/dedup_step3_semantic_only/

输出文件：
    1. semantic_no_similarity.jsonl
       完全无相似：单点簇，未与任何 descriptor 达到自动合并阈值。

    2. semantic_similar_groups.jsonl
       相似组：score >= AUTO_MERGE_THRESHOLD 的语义相似簇。

    3. semantic_combined_mode_recombined.jsonl
       组合组：完全无相似 + 每个相似组的合并众数 / canonical。

    4. semantic_dedup_stats.json
       统计信息。
"""

import os
import re
import json
from collections import defaultdict, Counter
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# ============================================================
# 配置区域
# ============================================================

INPUT_JSONL_PATH = (
    "./experiment_outputs/dedup_step2_descriptor_merge/"
    "fragment_level_records.descriptor_recombined.jsonl"
)

OUTPUT_DIR = "./experiment_outputs/dedup_step3_semantic_only"

MODEL_NAME = "BAAI/bge-large-en-v1.5"

# ============================================================
# 新阈值设定
# ============================================================
# 高保守自动合并阈值：
# score >= 0.87 才认为两个 raw_descriptor 可自动合并。
# 该阈值对应前面分析中的 fpr≤0.1%档位，适合“宁可漏合并，不要错合并”。
AUTO_MERGE_THRESHOLD = 0.8

# 不再使用灰区文件。
# 如后续想保留人工复核灰区，可单独加 REVIEW_THRESHOLD = 0.8705 或 0.8325。
USE_GRAY_ZONE = False

# 近邻搜索参数
# 为避免漏召回，建议比原来的 200 稍微提高。
TOP_K = 300

# FAISS HNSW 参数
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 512

ENCODE_BATCH_SIZE = 256
SEARCH_BATCH_SIZE = 8192

# None 表示自动选择；也可以设为 "cuda" 或 "cpu"
DEVICE = None

# 是否给最终组合文件添加 _semantic_meta
ADD_SEMANTIC_META_TO_FINAL = False

# 如果 raw_descriptor 缺失，是否直接报错
STRICT_RAW_DESCRIPTOR = True

# 调试用；None 表示处理全量
MAX_RECORDS = None


# ============================================================
# 输出文件
# ============================================================

SEMANTIC_NO_SIMILARITY_PATH = os.path.join(
    OUTPUT_DIR,
    "semantic_no_similarity.jsonl"
)

SEMANTIC_SIMILAR_GROUPS_PATH = os.path.join(
    OUTPUT_DIR,
    "semantic_similar_groups.jsonl"
)

SEMANTIC_COMBINED_MODE_PATH = os.path.join(
    OUTPUT_DIR,
    "semantic_combined_mode_recombined.jsonl"
)

SEMANTIC_STATS_PATH = os.path.join(
    OUTPUT_DIR,
    "semantic_dedup_stats.json"
)


# ============================================================
# 基础函数
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def normalize_display_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def get_raw_descriptor(record: Dict[str, Any], line_no: int) -> str:
    raw_descriptor = record.get("raw_descriptor", None)

    if raw_descriptor is None or str(raw_descriptor).strip() == "":
        msg = f"Line {line_no}: missing or empty raw_descriptor"
        if STRICT_RAW_DESCRIPTOR:
            raise ValueError(msg)
        return ""

    return normalize_display_text(raw_descriptor)


def with_meta(record: Dict[str, Any], key: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    out[key] = meta
    return out


def get_exact_record_count(item: Dict[str, Any]) -> int:
    """
    如果前一步 raw 重复合并时保留了 _merge_meta.record_count，
    则优先使用它作为“众数/代表项”选择依据。
    """
    meta = item.get("_merge_meta", {})
    if isinstance(meta, dict):
        try:
            return int(meta.get("record_count", 1))
        except Exception:
            return 1
    return 1


def get_descriptor_text(item: Dict[str, Any]) -> str:
    return normalize_display_text(item.get("raw_descriptor", ""))


def to_sample_id_list(value: Any) -> List[Any]:
    """输出文件中 sample_id 始终为去重 list。"""
    return ordered_unique(flatten_sample_ids_from_value(value))


def minimal_output_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    只保留最终数据集需要的 6 个字段。

    输出格式严格为：
    {
      "sample_id": [...],
      "fragment_id": ...,
      "raw_descriptor": ...,
      "mechanism_category": ...,
      "effect_profile": {...},
      "granularity": ...
    }
    """
    return {
        "sample_id": to_sample_id_list(record.get("sample_id")),
        "fragment_id": record.get("fragment_id"),
        "raw_descriptor": normalize_display_text(record.get("raw_descriptor", "")),
        "mechanism_category": record.get("mechanism_category"),
        "effect_profile": record.get("effect_profile", {}),
        "granularity": record.get("granularity"),
    }


# ============================================================
# 语义簇记录合并策略
# ============================================================

def value_to_key(value: Any) -> str:
    """将任意 JSON-like value 转成可比较 key。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def ordered_unique(values: List[Any]) -> List[Any]:
    """保留首次出现顺序的去重。"""
    seen = set()
    result = []
    for value in values:
        key = value_to_key(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def is_number_like(value: Any) -> bool:
    """bool 虽然是 int 子类，但这里不视为可平均数值。"""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def flatten_sample_ids_from_value(value: Any) -> List[Any]:
    """sample_id 可能是单值，也可能在前一步中已经是 list。"""
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for x in value:
            if x is None:
                continue
            if isinstance(x, list):
                result.extend(flatten_sample_ids_from_value(x))
            else:
                result.append(x)
        return result
    return [value]


def mode_or_average_tied(values: List[Any]) -> Any:
    """
    对字段值取众数。

    规则：
    1. 唯一众数：返回该众数；
    2. 多个众数且都是数值：返回这些众数的平均值；
    3. 多个众数且是 dict：逐 key 递归合并；
    4. 多个众数但无法平均：返回首次出现的众数，保证确定性。

    注意：字符串、类别字段没有数学平均含义，因此平票时无法真正“取平均”，
    只能稳定回退到首次出现的众数。
    """
    clean_values = [v for v in values if v is not None]
    if not clean_values:
        return None

    counter = Counter(value_to_key(v) for v in clean_values)
    max_count = max(counter.values())
    mode_keys = {k for k, c in counter.items() if c == max_count}

    first_value_by_key = {}
    for v in clean_values:
        k = value_to_key(v)
        if k not in first_value_by_key:
            first_value_by_key[k] = v

    modes = [first_value_by_key[k] for k in mode_keys]

    if len(modes) == 1:
        return modes[0]

    # 多个众数均为数值时，取平均。
    if all(is_number_like(v) for v in modes):
        return float(np.mean([float(v) for v in modes]))

    # 多个众数均为 dict 时，对每个子字段递归合并。
    if all(isinstance(v, dict) for v in modes):
        return merge_dict_values(modes)

    # 其他类型无法平均，取最早出现的众数，保证结果稳定。
    for v in clean_values:
        if value_to_key(v) in mode_keys:
            return v

    return clean_values[0]


def merge_dict_values(dict_values: List[Dict[str, Any]]) -> Dict[str, Any]:
    """合并 dict 字段，例如 effect_profile：每个维度分别取众数，多众数数值取平均。"""
    key_to_values = defaultdict(list)
    for d in dict_values:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            key_to_values[k].append(v)

    merged = {}
    for k in sorted(key_to_values.keys()):
        merged[k] = mode_or_average_tied(key_to_values[k])
    return merged


def count_values_for_meta(values: List[Any]) -> Dict[str, int]:
    """用于元信息记录字段值分布。"""
    counter = Counter(value_to_key(v) for v in values if v is not None)
    result = {}
    for k, c in counter.items():
        try:
            display_key = str(json.loads(k))
        except Exception:
            display_key = k
        result[display_key] = int(c)
    return result


def choose_simplest_raw_descriptor(records: List[Dict[str, Any]]) -> str:
    """raw_descriptor 取最简洁表达：长度最短；若长度相同，按字典序稳定选择。"""
    descriptors = [get_descriptor_text(r) for r in records if get_descriptor_text(r)]
    if not descriptors:
        return ""
    return sorted(descriptors, key=lambda x: (len(x), x.casefold(), x))[0]


def choose_fragment_id_for_cluster(records: List[Dict[str, Any]]) -> Any:
    """
    fragment_id 只保留一个。
    优先选 raw_descriptor 最简洁的那条记录对应的 fragment_id；若缺失，则回退到第一个非空值。
    """
    sorted_records = sorted(
        records,
        key=lambda r: (
            len(get_descriptor_text(r)),
            get_descriptor_text(r).casefold(),
            get_descriptor_text(r),
        )
    )
    for r in sorted_records:
        fid = r.get("fragment_id")
        if fid is not None and str(fid).strip() != "":
            return fid
    return None


def merge_semantic_cluster_records(
    records: List[Dict[str, Any]],
    cluster_id: int,
    member_indices: List[int],
) -> Dict[str, Any]:
    """
    将一个语义相似簇合并成一条最终数据记录。

    只输出 6 个字段：
    - sample_id：合并为去重 list；
    - fragment_id：只保留一个，取最简洁 raw_descriptor 所在记录的 fragment_id；
    - raw_descriptor：取最简洁表达；
    - mechanism_category：取众数；非数值多众数时取首次众数；
    - effect_profile：每个维度取众数；若多个数值众数，则取平均；
    - granularity：取众数；非数值多众数时取首次众数。

    不输出 _merge_meta、_semantic_meta、cluster_id、member_indices 等额外内容。
    """
    if not records:
        raise ValueError("merge_semantic_cluster_records received empty records")

    merged: Dict[str, Any] = {}

    sample_ids = []
    for r in records:
        sample_ids.extend(flatten_sample_ids_from_value(r.get("sample_id")))
    merged["sample_id"] = ordered_unique(sample_ids)

    merged["fragment_id"] = choose_fragment_id_for_cluster(records)
    merged["raw_descriptor"] = choose_simplest_raw_descriptor(records)

    mechanism_values = [
        r.get("mechanism_category")
        for r in records
        if "mechanism_category" in r and r.get("mechanism_category") is not None
    ]
    merged["mechanism_category"] = mode_or_average_tied(mechanism_values)

    effect_profiles = [
        r.get("effect_profile")
        for r in records
        if isinstance(r.get("effect_profile"), dict)
    ]
    merged["effect_profile"] = merge_dict_values(effect_profiles) if effect_profiles else {}

    granularity_values = [
        r.get("granularity")
        for r in records
        if "granularity" in r and r.get("granularity") is not None
    ]
    merged["granularity"] = mode_or_average_tied(granularity_values)

    return merged


# ============================================================
# 读取输入
# ============================================================

def load_items(input_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    items = []

    stats = {
        "input_path": input_path,
        "total_lines": 0,
        "empty_lines": 0,
        "valid_records": 0,
        "invalid_json_lines": 0,
        "missing_raw_descriptor": 0,
        "semantic_input_item_count": 0,
    }

    print("=" * 100)
    print("读取 raw 去重后的输入文件")
    print("=" * 100)
    print(f"输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(tqdm(fin, desc="Reading input JSONL"), 1):
            stats["total_lines"] += 1

            if MAX_RECORDS is not None and len(items) >= MAX_RECORDS:
                break

            raw_line = line.rstrip("\n")

            if not raw_line.strip():
                stats["empty_lines"] += 1
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as e:
                stats["invalid_json_lines"] += 1
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e

            raw_descriptor = record.get("raw_descriptor", None)
            if raw_descriptor is None or str(raw_descriptor).strip() == "":
                stats["missing_raw_descriptor"] += 1
                if STRICT_RAW_DESCRIPTOR:
                    raise ValueError(f"Line {line_no}: missing raw_descriptor")
                continue

            items.append(record)
            stats["valid_records"] += 1

    stats["semantic_input_item_count"] = len(items)

    print(f"读取完成，语义去重输入记录数: {len(items)}")

    return items, stats


# ============================================================
# Embedding + FAISS
# ============================================================

def load_faiss():
    try:
        import faiss
        return faiss
    except ImportError as e:
        raise ImportError("缺少 faiss，请先安装：pip install faiss-cpu") from e


def encode_descriptors(items: List[Dict[str, Any]]) -> np.ndarray:
    descriptors = [get_descriptor_text(item) for item in items]

    print("\n" + "=" * 100)
    print("编码 raw_descriptor")
    print("=" * 100)
    print(f"模型: {MODEL_NAME}")
    print(f"descriptor 数量: {len(descriptors)}")

    if DEVICE is None:
        model = SentenceTransformer(MODEL_NAME)
    else:
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    embeddings = model.encode(
        descriptors,
        batch_size=ENCODE_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return np.asarray(embeddings, dtype="float32")


def build_hnsw_index(embeddings: np.ndarray):
    faiss = load_faiss()
    dim = embeddings.shape[1]

    try:
        index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    except TypeError:
        index = faiss.IndexHNSWFlat(dim, HNSW_M)
        index.metric_type = faiss.METRIC_INNER_PRODUCT

    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH

    print("\n构建 FAISS HNSW 索引...")
    index.add(embeddings)
    print(f"索引构建完成，向量数量: {index.ntotal}")

    return index


def search_neighbors(index, embeddings: np.ndarray):
    n = embeddings.shape[0]
    k = min(TOP_K, n)

    all_scores = np.empty((n, k), dtype="float32")
    all_indices = np.empty((n, k), dtype="int64")

    print("\n" + "=" * 100)
    print("近邻搜索")
    print("=" * 100)
    print(f"TOP_K: {k}")

    for start in tqdm(range(0, n, SEARCH_BATCH_SIZE), desc="FAISS search"):
        end = min(start + SEARCH_BATCH_SIZE, n)
        scores, indices = index.search(embeddings[start:end], k)
        all_scores[start:end] = scores
        all_indices[start:end] = indices

    return all_scores, all_indices


# ============================================================
# canonical / 众数选择
# ============================================================

def choose_canonical_order(items: List[Dict[str, Any]]) -> List[int]:
    """
    canonical 选择优先级：
    1. raw 精确重复合并次数更多的优先，也就是更像“众数”
    2. descriptor 更短优先
    3. 字典序优先

    这样相似组最终保留的是“合并众数/代表表达”。
    """
    return sorted(
        range(len(items)),
        key=lambda i: (
            -get_exact_record_count(items[i]),
            len(get_descriptor_text(items[i])),
            get_descriptor_text(items[i])
        )
    )


def choose_cluster_mode_item(
    members: List[int],
    items: List[Dict[str, Any]],
) -> int:
    """
    在一个语义相似簇内部选择合并众数 / 代表项。

    优先级：
    1. _merge_meta.record_count 最大
    2. raw_descriptor 更短
    3. 字典序
    """
    return sorted(
        members,
        key=lambda i: (
            -get_exact_record_count(items[i]),
            len(get_descriptor_text(items[i])),
            get_descriptor_text(items[i])
        )
    )[0]


# ============================================================
# 语义自动合并聚类
# ============================================================

def semantic_auto_cluster(
    items: List[Dict[str, Any]],
    embeddings: np.ndarray,
    neighbor_scores: np.ndarray,
    neighbor_indices: np.ndarray,
):
    """
    score >= AUTO_MERGE_THRESHOLD 的 descriptor 自动合并。

    采用 canonical-greedy：
    - 避免 A≈B、B≈C 造成 A 和 C 被链式误合并
    - 每个 member 必须与 cluster canonical 达到自动合并阈值
    """
    n = len(items)

    assigned_cluster = np.full(n, -1, dtype=np.int64)
    score_to_canonical = np.zeros(n, dtype="float32")
    canonical_indices: List[int] = []

    order = choose_canonical_order(items)

    def score_to_cluster_canonical(item_idx: int, cluster_id: int) -> float:
        canonical_idx = canonical_indices[cluster_id]
        return float(np.dot(embeddings[item_idx], embeddings[canonical_idx]))

    print("\n" + "=" * 100)
    print("语义自动合并聚类")
    print("=" * 100)
    print(f"自动合并阈值: {AUTO_MERGE_THRESHOLD}")

    for idx in tqdm(order, desc="Semantic auto clustering"):
        if assigned_cluster[idx] != -1:
            continue

        # 尝试挂到已有 cluster
        best_cluster = -1
        best_score = -1.0

        for score, j in zip(neighbor_scores[idx], neighbor_indices[idx]):
            j = int(j)
            score = float(score)

            if j < 0 or j == idx:
                continue

            if score < AUTO_MERGE_THRESHOLD:
                continue

            j_cluster = int(assigned_cluster[j])
            if j_cluster == -1:
                continue

            canonical_score = score_to_cluster_canonical(idx, j_cluster)

            if canonical_score >= AUTO_MERGE_THRESHOLD and canonical_score > best_score:
                best_cluster = j_cluster
                best_score = canonical_score

        if best_cluster != -1:
            assigned_cluster[idx] = best_cluster
            score_to_canonical[idx] = best_score
            continue

        # 创建新 cluster
        cluster_id = len(canonical_indices)
        canonical_indices.append(idx)
        assigned_cluster[idx] = cluster_id
        score_to_canonical[idx] = 1.0

        # 将当前 canonical 的高置信近邻挂进来
        for score, j in zip(neighbor_scores[idx], neighbor_indices[idx]):
            j = int(j)
            score = float(score)

            if j < 0 or j == idx:
                continue

            if score >= AUTO_MERGE_THRESHOLD and assigned_cluster[j] == -1:
                assigned_cluster[j] = cluster_id
                score_to_canonical[j] = score

    clusters = defaultdict(list)
    for item_idx, cluster_id in enumerate(assigned_cluster):
        clusters[int(cluster_id)].append(int(item_idx))

    # 聚类完成后，重新用“合并众数”校正 canonical
    # 这样最终输出代表项更符合“合并众数”的语义。
    corrected_canonical_indices = []
    for cluster_id in range(len(canonical_indices)):
        members = clusters[cluster_id]
        mode_idx = choose_cluster_mode_item(members, items)
        corrected_canonical_indices.append(mode_idx)

    # 重新计算每个成员到最终众数项的分数
    corrected_score_to_canonical = np.zeros(n, dtype="float32")
    for cluster_id, members in clusters.items():
        canonical_idx = corrected_canonical_indices[cluster_id]
        for m in members:
            corrected_score_to_canonical[m] = float(
                np.dot(embeddings[m], embeddings[canonical_idx])
            )

    return {
        "assigned_cluster": assigned_cluster,
        "score_to_canonical": corrected_score_to_canonical,
        "canonical_indices": corrected_canonical_indices,
        "clusters": dict(clusters)
    }


# ============================================================
# 输出辅助
# ============================================================

def make_item_summary(
    items: List[Dict[str, Any]],
    item_idx: int,
    cluster_result: Dict[str, Any],
) -> Dict[str, Any]:
    assigned_cluster = cluster_result["assigned_cluster"]
    score_to_canonical = cluster_result["score_to_canonical"]
    canonical_indices = cluster_result["canonical_indices"]

    cluster_id = int(assigned_cluster[item_idx])
    canonical_idx = int(canonical_indices[cluster_id])

    item = items[item_idx]

    return {
        "item_index": item_idx,
        "cluster_id": cluster_id,
        "is_canonical_mode": bool(item_idx == canonical_idx),
        "score_to_canonical_mode": float(score_to_canonical[item_idx]),
        "raw_descriptor": get_descriptor_text(item),
        "sample_id": item.get("sample_id"),
        "fragment_id": item.get("fragment_id"),
        "mechanism_category": item.get("mechanism_category"),
        "granularity": item.get("granularity"),
        "exact_record_count": get_exact_record_count(item),
        "record": item
    }


# ============================================================
# 输出 1：完全无相似
# ============================================================

def write_no_similarity_items(
    items: List[Dict[str, Any]],
    cluster_result: Dict[str, Any],
) -> Tuple[Set[int], int]:
    """
    输出完全无相似记录。

    格式要求：只保留 6 个字段，并将 sample_id 规范为 list；
    不添加 _semantic_meta 或任何额外字段。
    """
    clusters = cluster_result["clusters"]

    no_similarity_indices = set()
    count = 0

    with open(SEMANTIC_NO_SIMILARITY_PATH, "w", encoding="utf-8") as fout:
        for cluster_id in sorted(clusters.keys()):
            members = clusters[cluster_id]

            if len(members) != 1:
                continue

            item_idx = int(members[0])
            no_similarity_indices.add(item_idx)

            fout.write(json_dumps(minimal_output_record(items[item_idx])) + "\n")
            count += 1

    print(f"完全无相似文件已输出: {SEMANTIC_NO_SIMILARITY_PATH}")
    print(f"完全无相似记录数量: {count}")

    return no_similarity_indices, count


# ============================================================
# 输出 2：相似组
# ============================================================

def write_similar_groups(
    items: List[Dict[str, Any]],
    cluster_result: Dict[str, Any],
) -> Tuple[Set[int], int]:
    """
    输出相似组，简化格式：

    每一行是一个 JSON list，只保留 raw_descriptor，例如：

    ["oil painting","oil paint artwork","oil painting style"]

    说明：
    - 只输出 cluster_size > 1 的相似组；
    - 每组一行；
    - 列表第一个元素是 canonical_mode / 合并众数项；
    - 后续元素按 record_count、长度、字典序排序；
    - 不再输出 cluster_id、score、record、sample_id 等详细字段。
    """
    clusters = cluster_result["clusters"]
    canonical_indices = cluster_result["canonical_indices"]

    similar_item_indices = set()
    similar_group_count = 0

    with open(SEMANTIC_SIMILAR_GROUPS_PATH, "w", encoding="utf-8") as fout:
        for cluster_id in sorted(clusters.keys()):
            members = clusters[cluster_id]

            # 单点簇不是相似组，不输出
            if len(members) <= 1:
                continue

            similar_group_count += 1

            canonical_idx = int(canonical_indices[cluster_id])

            for m in members:
                similar_item_indices.add(m)

            # canonical / 众数放在第一位
            sorted_members = sorted(
                members,
                key=lambda x: (
                    x != canonical_idx,
                    -get_exact_record_count(items[x]),
                    len(get_descriptor_text(items[x])),
                    get_descriptor_text(items[x])
                )
            )

            raw_descriptor_list = [
                get_descriptor_text(items[m])
                for m in sorted_members
            ]

            # 可选：如果担心组内 raw_descriptor 有重复，可以打开下面几行去重
            # seen = set()
            # raw_descriptor_list = [
            #     x for x in raw_descriptor_list
            #     if not (x in seen or seen.add(x))
            # ]

            fout.write(json_dumps(raw_descriptor_list) + "\n")

    print(f"相似组文件已输出: {SEMANTIC_SIMILAR_GROUPS_PATH}")
    print(f"相似组数量: {similar_group_count}")

    return similar_item_indices, similar_group_count


# ============================================================
# 输出 3：组合组 = 完全无相似 + 合并众数
# ============================================================

def write_combined_mode_recombined(
    items: List[Dict[str, Any]],
    cluster_result: Dict[str, Any],
) -> int:
    """
    输出最终组合文件：
    - 单点簇：保留原记录，但 sample_id 统一转换为 list；
    - 相似组：将组内所有记录合并为一条新记录。

    最终每行严格只包含 6 个字段：
    sample_id, fragment_id, raw_descriptor, mechanism_category, effect_profile, granularity。
    不输出任何额外 meta 信息。
    """
    clusters = cluster_result["clusters"]

    kept_count = 0

    with open(SEMANTIC_COMBINED_MODE_PATH, "w", encoding="utf-8") as fout:
        for cluster_id in sorted(clusters.keys()):
            members = sorted([int(m) for m in clusters[cluster_id]])

            if len(members) == 1:
                out_record = minimal_output_record(items[members[0]])
            else:
                records = [items[m] for m in members]
                out_record = merge_semantic_cluster_records(
                    records=records,
                    cluster_id=int(cluster_id),
                    member_indices=members,
                )

            fout.write(json_dumps(out_record) + "\n")
            kept_count += 1

    print(f"组合组文件已输出: {SEMANTIC_COMBINED_MODE_PATH}")
    print(f"组合组最终保留记录数量: {kept_count}")

    return kept_count


# ============================================================
# 统计
# ============================================================

def write_stats(
    base_stats: Dict[str, Any],
    items: List[Dict[str, Any]],
    cluster_result: Dict[str, Any],
    no_similarity_count: int,
    similar_group_count: int,
    combined_count: int,
):
    clusters = cluster_result["clusters"]
    cluster_sizes = [len(v) for v in clusters.values()]

    similar_item_count = sum(len(v) for v in clusters.values() if len(v) > 1)
    no_similarity_cluster_count = sum(1 for v in clusters.values() if len(v) == 1)

    stats = dict(base_stats)
    stats.update({
        "model_name": MODEL_NAME,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "top_k": TOP_K,
        "hnsw_m": HNSW_M,
        "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
        "hnsw_ef_search": HNSW_EF_SEARCH,

        "semantic_input_item_count": len(items),
        "semantic_cluster_count": len(clusters),

        "no_similarity_item_count": no_similarity_count,
        "no_similarity_cluster_count": no_similarity_cluster_count,

        "similar_group_count": similar_group_count,
        "similar_involved_item_count": similar_item_count,

        "combined_mode_recombined_item_count": combined_count,

        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "mean_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,

        "output_files": {
            "semantic_no_similarity": SEMANTIC_NO_SIMILARITY_PATH,
            "semantic_similar_groups": SEMANTIC_SIMILAR_GROUPS_PATH,
            "semantic_combined_mode_recombined": SEMANTIC_COMBINED_MODE_PATH,
            "semantic_stats": SEMANTIC_STATS_PATH
        }
    })

    with open(SEMANTIC_STATS_PATH, "w", encoding="utf-8") as fout:
        json.dump(stats, fout, ensure_ascii=False, indent=2)

    print(f"统计文件已输出: {SEMANTIC_STATS_PATH}")


# ============================================================
# 主流程
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 100)
    print("Step 3: 高保守 raw_descriptor 语义去重")
    print("=" * 100)
    print(f"输入文件: {INPUT_JSONL_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"模型: {MODEL_NAME}")
    print(f"自动合并阈值: {AUTO_MERGE_THRESHOLD}")
    print("输出类型: 完全无相似 / 相似组 / 组合组")

    items, base_stats = load_items(INPUT_JSONL_PATH)

    if not items:
        raise RuntimeError("输入文件中没有可用记录，请检查 INPUT_JSONL_PATH。")

    embeddings = encode_descriptors(items)

    index = build_hnsw_index(embeddings)

    neighbor_scores, neighbor_indices = search_neighbors(index, embeddings)

    cluster_result = semantic_auto_cluster(
        items=items,
        embeddings=embeddings,
        neighbor_scores=neighbor_scores,
        neighbor_indices=neighbor_indices
    )

    no_similarity_indices, no_similarity_count = write_no_similarity_items(
        items=items,
        cluster_result=cluster_result
    )

    similar_item_indices, similar_group_count = write_similar_groups(
        items=items,
        cluster_result=cluster_result
    )

    combined_count = write_combined_mode_recombined(
        items=items,
        cluster_result=cluster_result
    )

    write_stats(
        base_stats=base_stats,
        items=items,
        cluster_result=cluster_result,
        no_similarity_count=no_similarity_count,
        similar_group_count=similar_group_count,
        combined_count=combined_count
    )

    print("\n" + "=" * 100)
    print("语义去重完成")
    print("=" * 100)
    print("输出文件：")
    print(f"1. 完全无相似: {SEMANTIC_NO_SIMILARITY_PATH}")
    print(f"2. 相似组: {SEMANTIC_SIMILAR_GROUPS_PATH}")
    print(f"3. 组合组: {SEMANTIC_COMBINED_MODE_PATH}")
    print(f"4. 统计文件: {SEMANTIC_STATS_PATH}")

    print("\n统计摘要：")
    print(f"输入记录数: {len(items)}")
    print(f"完全无相似记录数: {no_similarity_count}")
    print(f"相似组数量: {similar_group_count}")
    print(f"组合组最终保留记录数: {combined_count}")


if __name__ == "__main__":
    main()