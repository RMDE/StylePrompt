# """
# 大规模风格词语义相似度评估脚本
#
# 功能：
# 1. 从 fragment_level_records.jsonl 中读取 raw_descriptor
# 2. 随机抽取 1000 个风格词
# 3. 每 5 个风格词为一组，共 200 组
# 4. 对两个模型分别评估：
#    - Same-style similarity：同一风格词在不同模板下的语义保持程度
#    - Cross-style similarity：同一模板下不同风格词之间的相似度
# 5. 每组生成一张合并图：
#    - 圆点：same-style
#    - 三角：cross-style
#    - 实线：same-style average
#    - 虚线：cross-style average
#    - 数字：separation_score = same_style_avg - cross_style_avg
#
# 最终输出：
# semantic_similarity_large_scale/
# ├── sampled_descriptors.csv
# ├── template_mapping.txt
# ├── sentence-transformers__all-mpnet-base-v2/
# │   ├── figures/
# │   │   ├── group_000.png
# │   │   ├── group_001.png
# │   │   └── ...
# │   ├── same_points.csv
# │   ├── cross_points.csv
# │   ├── group_template_summary.csv
# │   └── aggregate_template_ranking.csv
# └── BAAI__bge-large-en-v1.5/
#     ├── figures/
#     ├── same_points.csv
#     ├── cross_points.csv
#     ├── group_template_summary.csv
#     └── aggregate_template_ranking.csv
# """
#
# import os
# import re
# import csv
# import json
# import random
# import statistics
# from pathlib import Path
# from collections import defaultdict
# from typing import List, Dict, Tuple
#
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from sentence_transformers import SentenceTransformer
#
#
# try:
#     from tqdm import tqdm
# except ImportError:
#     def tqdm(x, **kwargs):
#         return x
#
#
# # ==================== 配置区域 ====================
#
# INPUT_JSONL_PATH = "./dataset_experiments/fragment_level_records.jsonl"
# OUTPUT_DIR = "./semantic_similarity_large_scale"
#
# MODEL_NAMES = [
#     "sentence-transformers/all-mpnet-base-v2",
#     "BAAI/bge-large-en-v1.5",
# ]
#
# SENTENCE_TEMPLATES = [
#     "{style}",
#     "in the style of {style}",
#     "rendered in {style} style",
#     "depicted in a {style} style",
#     "illustrated in {style} style",
#     "with a {style} aesthetic",
#     "in a {style} visual style",
#     "using a {style} rendering style",
#     "with the appearance of {style}",
#     "in the manner of {style}",
#     "styled as {style}",
#     "presented in {style} style",
#     "portrayed in a {style} style",
#     "visualized in {style} style",
#     "with {style} artistic treatment",
#     "following a {style} style",
#     "adopting a {style} visual appearance",
#     "given a {style} look",
#     "transformed into {style} style",
#     "with stylistic features of {style}",
#     "as a {style}-style image",
# ]
#
# SAMPLE_SIZE = 1000
# GROUP_SIZE = 5
# RANDOM_SEED = 42
#
# # True：对 raw_descriptor 去重后再随机采样，适合分析不同风格词之间的语义边界
# # False：保留数据集频次分布后采样，适合分析真实数据分布
# DEDUP_DESCRIPTORS = True
#
# LOWERCASE_DESCRIPTOR = False
# MIN_DESCRIPTOR_CHARS = 1
#
# BATCH_SIZE = 128
# FIG_DPI = 220
# SHOW_STYLE_MAPPING_ON_FIG = True
# ANNOTATE_SEPARATION_SCORE = True
#
#
# # ==================== 基础工具函数 ====================
#
# def ensure_dir(path: str):
#     os.makedirs(path, exist_ok=True)
#
#
# def safe_name(name: str) -> str:
#     """将模型名转换成安全目录名。"""
#     return re.sub(r"[^a-zA-Z0-9_.-]+", "__", name)
#
#
# def normalize_descriptor(text: str) -> str:
#     """对 raw_descriptor 做轻量清洗。"""
#     text = str(text).strip()
#     text = re.sub(r"\s+", " ", text)
#
#     if LOWERCASE_DESCRIPTOR:
#         text = text.lower()
#
#     return text
#
#
# def truncate_text(text: str, max_len: int = 42) -> str:
#     text = str(text)
#     if len(text) <= max_len:
#         return text
#     return text[:max_len - 3] + "..."
#
#
# def read_raw_descriptors(jsonl_path: str) -> List[str]:
#     """从 jsonl 文件中读取 raw_descriptor。"""
#     descriptors = []
#     bad_json_count = 0
#     missing_count = 0
#
#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         for line_idx, line in enumerate(f, 1):
#             line = line.strip()
#             if not line:
#                 continue
#
#             try:
#                 obj = json.loads(line)
#             except json.JSONDecodeError:
#                 bad_json_count += 1
#                 continue
#
#             desc = obj.get("raw_descriptor", None)
#
#             if desc is None:
#                 missing_count += 1
#                 continue
#
#             desc = normalize_descriptor(desc)
#
#             if len(desc) < MIN_DESCRIPTOR_CHARS:
#                 continue
#
#             descriptors.append(desc)
#
#     print(f"读取完成: {jsonl_path}")
#     print(f"  有效 raw_descriptor 数量: {len(descriptors)}")
#     print(f"  JSON 解析失败行数: {bad_json_count}")
#     print(f"  缺失 raw_descriptor 行数: {missing_count}")
#
#     return descriptors
#
#
# def deduplicate_descriptors(descriptors: List[str]) -> List[str]:
#     """按 casefold 后的文本去重，但保留首次出现的原始形式。"""
#     seen = set()
#     unique = []
#
#     for desc in descriptors:
#         key = desc.casefold().strip()
#         if key not in seen:
#             seen.add(key)
#             unique.append(desc)
#
#     return unique
#
#
# def sample_descriptors(
#     descriptors: List[str],
#     sample_size: int,
#     seed: int,
#     dedup: bool = True
# ) -> List[str]:
#     """随机采样指定数量的 descriptor。"""
#     if dedup:
#         before = len(descriptors)
#         descriptors = deduplicate_descriptors(descriptors)
#         after = len(descriptors)
#         print(f"去重后 descriptor 数量: {after} / {before}")
#
#     if len(descriptors) < sample_size:
#         raise ValueError(
#             f"可用 descriptor 数量不足：当前 {len(descriptors)}，需要 {sample_size}。"
#         )
#
#     random.seed(seed)
#     sampled = random.sample(descriptors, sample_size)
#
#     return sampled
#
#
# def split_into_groups(items: List[str], group_size: int) -> List[List[str]]:
#     """将列表按 group_size 切分。"""
#     assert len(items) % group_size == 0, "SAMPLE_SIZE 必须能被 GROUP_SIZE 整除。"
#
#     return [
#         items[i:i + group_size]
#         for i in range(0, len(items), group_size)
#     ]
#
#
# def save_template_mapping(templates: List[str], output_dir: str):
#     path = os.path.join(output_dir, "template_mapping.txt")
#
#     with open(path, "w", encoding="utf-8") as f:
#         for i, template in enumerate(templates):
#             f.write(f"T{i:02d}: {template}\n")
#
#     print(f"模板映射已保存: {path}")
#
#
# def save_sampled_descriptors(
#     sampled: List[str],
#     groups: List[List[str]],
#     output_dir: str
# ):
#     """保存采样结果和分组信息。"""
#     path = os.path.join(output_dir, "sampled_descriptors.csv")
#
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "global_index",
#             "group_index",
#             "index_in_group",
#             "style_id",
#             "raw_descriptor"
#         ])
#
#         global_idx = 0
#         for g_idx, group in enumerate(groups):
#             for local_idx, desc in enumerate(group):
#                 writer.writerow([
#                     global_idx,
#                     g_idx,
#                     local_idx,
#                     f"S{local_idx}",
#                     desc
#                 ])
#                 global_idx += 1
#
#     print(f"采样 descriptor 已保存: {path}")
#
#
# # ==================== Embedding 计算 ====================
#
# def encode_texts(
#     model: SentenceTransformer,
#     texts: List[str],
#     batch_size: int = BATCH_SIZE
# ) -> torch.Tensor:
#     """编码文本并做归一化，返回 tensor。"""
#     embeddings = model.encode(
#         texts,
#         batch_size=batch_size,
#         convert_to_tensor=True,
#         normalize_embeddings=True,
#         show_progress_bar=False
#     )
#
#     return embeddings
#
#
# def precompute_embeddings_for_model(
#     model: SentenceTransformer,
#     sampled_descriptors: List[str],
#     templates: List[str]
# ):
#     """
#     对一个模型预计算：
#     1. base_embeddings：原始风格词 embedding
#     2. template_embeddings：每个模板下所有风格词形式表述的 embedding
#     """
#     print("正在编码原始风格词...")
#     base_embeddings = encode_texts(model, sampled_descriptors)
#
#     template_embeddings = []
#
#     print("正在编码不同模板下的风格词表述...")
#     for t_idx, template in enumerate(tqdm(templates, desc="Encoding templates")):
#         sentences = [
#             template.format(style=s)
#             for s in sampled_descriptors
#         ]
#         emb = encode_texts(model, sentences)
#         template_embeddings.append(emb)
#
#     return base_embeddings, template_embeddings
#
#
# # ==================== 单组评估 ====================
#
# def evaluate_one_group(
#     group_indices: List[int],
#     base_embeddings: torch.Tensor,
#     template_embeddings: List[torch.Tensor],
#     templates: List[str]
# ):
#     """
#     评估一个 5 风格词组。
#
#     返回：
#     same_scores:
#         shape = [num_templates, group_size]
#         same_scores[t, i] = sim(raw_style_i, template_t(style_i))
#
#     cross_scores_by_template:
#         dict:
#         {
#             template_index: [
#                 {
#                     pair_index,
#                     style_i_local,
#                     style_j_local,
#                     similarity
#                 },
#                 ...
#             ]
#         }
#
#     same_avg:
#         每个模板的 same-style 平均分
#
#     cross_avg:
#         每个模板的 cross-style 平均分
#
#     separation:
#         same_avg - cross_avg
#     """
#     num_templates = len(templates)
#     group_size = len(group_indices)
#
#     base_sel = base_embeddings[group_indices]
#
#     same_scores = np.zeros((num_templates, group_size), dtype=np.float32)
#     cross_scores_by_template: Dict[int, List[Dict]] = {}
#
#     for t_idx in range(num_templates):
#         temp_sel = template_embeddings[t_idx][group_indices]
#
#         # same-style：raw style_i vs template(style_i)
#         same = torch.sum(base_sel * temp_sel, dim=1)
#         same_scores[t_idx, :] = same.detach().cpu().numpy()
#
#         # cross-style：template(style_i) vs template(style_j)
#         sim_matrix = torch.matmul(temp_sel, temp_sel.T).detach().cpu().numpy()
#
#         records = []
#         pair_idx = 0
#         for i in range(group_size):
#             for j in range(i + 1, group_size):
#                 records.append({
#                     "pair_index": pair_idx,
#                     "style_i_local": i,
#                     "style_j_local": j,
#                     "similarity": float(sim_matrix[i, j])
#                 })
#                 pair_idx += 1
#
#         cross_scores_by_template[t_idx] = records
#
#     same_avg = same_scores.mean(axis=1)
#
#     cross_avg = np.zeros(num_templates, dtype=np.float32)
#     for t_idx in range(num_templates):
#         vals = [
#             r["similarity"]
#             for r in cross_scores_by_template[t_idx]
#         ]
#         cross_avg[t_idx] = float(np.mean(vals)) if vals else 0.0
#
#     separation = same_avg - cross_avg
#
#     return same_scores, cross_scores_by_template, same_avg, cross_avg, separation
#
#
# # ==================== 绘图函数 ====================
#
# def plot_group_combined_figure(
#     model_name: str,
#     group_index: int,
#     style_words: List[str],
#     templates: List[str],
#     same_scores: np.ndarray,
#     cross_scores_by_template: Dict[int, List[Dict]],
#     same_avg: np.ndarray,
#     cross_avg: np.ndarray,
#     separation: np.ndarray,
#     output_path: str
# ):
#     """
#     画单个 group 的合并图。
#     """
#     num_templates = len(templates)
#     group_size = len(style_words)
#
#     fig, ax = plt.subplots(figsize=(22, 9))
#
#     x_base = np.arange(num_templates)
#
#     # Same-style points
#     same_offsets = np.linspace(-0.22, 0.22, group_size)
#
#     for s_idx in range(group_size):
#         xs = x_base + same_offsets[s_idx]
#         ys = same_scores[:, s_idx]
#
#         ax.scatter(
#             xs,
#             ys,
#             s=70,
#             marker="o",
#             color="#1f77b4",
#             alpha=0.82,
#             edgecolors="black",
#             linewidths=0.45
#         )
#
#     # Cross-style points
#     pairs = []
#     for i in range(group_size):
#         for j in range(i + 1, group_size):
#             pairs.append((i, j))
#
#     pair_offsets = np.linspace(-0.36, 0.36, len(pairs))
#
#     for pair_idx, (i, j) in enumerate(pairs):
#         xs = []
#         ys = []
#
#         for t_idx in range(num_templates):
#             records = cross_scores_by_template[t_idx]
#             for r in records:
#                 if r["style_i_local"] == i and r["style_j_local"] == j:
#                     xs.append(t_idx + pair_offsets[pair_idx])
#                     ys.append(r["similarity"])
#                     break
#
#         ax.scatter(
#             xs,
#             ys,
#             s=65,
#             marker="^",
#             color="#ff7f0e",
#             alpha=0.72,
#             edgecolors="black",
#             linewidths=0.35
#         )
#
#     # Average lines
#     ax.plot(
#         x_base,
#         same_avg,
#         marker="o",
#         linestyle="-",
#         linewidth=2.4,
#         color="#1f77b4",
#         label="Same-style average"
#     )
#
#     ax.plot(
#         x_base,
#         cross_avg,
#         marker="^",
#         linestyle="--",
#         linewidth=2.4,
#         color="#ff7f0e",
#         label="Cross-style average"
#     )
#
#     # Separation gap
#     for t_idx in range(num_templates):
#         ax.vlines(
#             x=t_idx,
#             ymin=min(same_avg[t_idx], cross_avg[t_idx]),
#             ymax=max(same_avg[t_idx], cross_avg[t_idx]),
#             linestyles=":",
#             linewidth=1.2,
#             color="gray",
#             alpha=0.55
#         )
#
#         if ANNOTATE_SEPARATION_SCORE:
#             ax.text(
#                 t_idx,
#                 max(same_avg[t_idx], cross_avg[t_idx]) + 0.015,
#                 f"{separation[t_idx]:.2f}",
#                 ha="center",
#                 va="bottom",
#                 fontsize=7.5,
#                 color="black"
#             )
#
#     # Style mapping box
#     if SHOW_STYLE_MAPPING_ON_FIG:
#         mapping_text = "\n".join([
#             f"S{i}: {truncate_text(style_words[i], 48)}"
#             for i in range(group_size)
#         ])
#
#         ax.text(
#             1.01,
#             0.50,
#             mapping_text,
#             transform=ax.transAxes,
#             fontsize=8,
#             va="center",
#             ha="left",
#             bbox=dict(
#                 boxstyle="round,pad=0.4",
#                 facecolor="white",
#                 edgecolor="gray",
#                 alpha=0.85
#             )
#         )
#
#     # Legend
#     custom_handles = [
#         Line2D(
#             [0], [0],
#             marker="o",
#             linestyle="None",
#             markerfacecolor="#1f77b4",
#             markeredgecolor="black",
#             markersize=8,
#             label="Same-style point"
#         ),
#         Line2D(
#             [0], [0],
#             marker="^",
#             linestyle="None",
#             markerfacecolor="#ff7f0e",
#             markeredgecolor="black",
#             markersize=8,
#             label="Cross-style point"
#         ),
#         Line2D(
#             [0], [0],
#             marker="o",
#             linestyle="-",
#             color="#1f77b4",
#             linewidth=2.4,
#             label="Same-style average"
#         ),
#         Line2D(
#             [0], [0],
#             marker="^",
#             linestyle="--",
#             color="#ff7f0e",
#             linewidth=2.4,
#             label="Cross-style average"
#         ),
#         Line2D(
#             [0], [0],
#             linestyle=":",
#             color="gray",
#             linewidth=1.5,
#             label="Separation gap"
#         ),
#     ]
#
#     ax.legend(
#         handles=custom_handles,
#         loc="lower left",
#         fontsize=9
#     )
#
#     ax.set_xlabel("Sentence template index")
#     ax.set_ylabel("Semantic similarity")
#     ax.set_title(
#         f"Group {group_index:03d} | Model: {model_name}\n"
#         "Circle = same-style similarity, Triangle = cross-style similarity, "
#         "Number above gap = separation score"
#     )
#
#     ax.set_xticks(x_base)
#     ax.set_xticklabels(
#         [f"T{i:02d}" for i in range(num_templates)],
#         rotation=45
#     )
#
#     # =========================
#     # 动态设置纵坐标范围
#     # =========================
#     all_y_values = []
#
#     all_y_values.extend(same_scores.flatten().tolist())
#
#     for t_idx in range(num_templates):
#         for r in cross_scores_by_template[t_idx]:
#             all_y_values.append(float(r["similarity"]))
#
#     all_y_values.extend(same_avg.tolist())
#     all_y_values.extend(cross_avg.tolist())
#
#     if all_y_values:
#         y_min = min(all_y_values) - 0.05
#         y_max = max(all_y_values) + 0.05
#
#         if abs(y_max - y_min) < 1e-6:
#             y_min -= 0.05
#             y_max += 0.05
#
#         ax.set_ylim(y_min, y_max)
#     else:
#         ax.set_ylim(0, 1.08)
#
#     ax.grid(True, linestyle="--", alpha=0.35)
#
#     if SHOW_STYLE_MAPPING_ON_FIG:
#         plt.tight_layout(rect=[0, 0, 0.86, 1])
#     else:
#         plt.tight_layout()
#
#     plt.savefig(output_path, dpi=FIG_DPI)
#     plt.close(fig)
#
#
# # ==================== CSV 输出 ====================
#
# def open_model_csv_writers(model_dir: str):
#     same_path = os.path.join(model_dir, "same_points.csv")
#     cross_path = os.path.join(model_dir, "cross_points.csv")
#     summary_path = os.path.join(model_dir, "group_template_summary.csv")
#
#     same_f = open(same_path, "w", newline="", encoding="utf-8")
#     cross_f = open(cross_path, "w", newline="", encoding="utf-8")
#     summary_f = open(summary_path, "w", newline="", encoding="utf-8")
#
#     same_writer = csv.writer(same_f)
#     cross_writer = csv.writer(cross_f)
#     summary_writer = csv.writer(summary_f)
#
#     same_writer.writerow([
#         "model_name",
#         "group_index",
#         "template_index",
#         "template_id",
#         "template",
#         "style_local_index",
#         "style_id",
#         "style_text",
#         "same_style_similarity"
#     ])
#
#     cross_writer.writerow([
#         "model_name",
#         "group_index",
#         "template_index",
#         "template_id",
#         "template",
#         "pair_index",
#         "style_i_local",
#         "style_j_local",
#         "style_pair_id",
#         "style_i_text",
#         "style_j_text",
#         "cross_style_similarity"
#     ])
#
#     summary_writer.writerow([
#         "model_name",
#         "group_index",
#         "template_index",
#         "template_id",
#         "template",
#         "same_style_avg",
#         "cross_style_avg",
#         "separation_score"
#     ])
#
#     return {
#         "same_file": same_f,
#         "cross_file": cross_f,
#         "summary_file": summary_f,
#         "same_writer": same_writer,
#         "cross_writer": cross_writer,
#         "summary_writer": summary_writer,
#         "same_path": same_path,
#         "cross_path": cross_path,
#         "summary_path": summary_path,
#     }
#
#
# def close_model_csv_writers(writers: Dict):
#     writers["same_file"].close()
#     writers["cross_file"].close()
#     writers["summary_file"].close()
#
#
# def write_group_results_to_csv(
#     writers: Dict,
#     model_name: str,
#     group_index: int,
#     style_words: List[str],
#     templates: List[str],
#     same_scores: np.ndarray,
#     cross_scores_by_template: Dict[int, List[Dict]],
#     same_avg: np.ndarray,
#     cross_avg: np.ndarray,
#     separation: np.ndarray
# ):
#     same_writer = writers["same_writer"]
#     cross_writer = writers["cross_writer"]
#     summary_writer = writers["summary_writer"]
#
#     num_templates = len(templates)
#     group_size = len(style_words)
#
#     for t_idx in range(num_templates):
#         template = templates[t_idx]
#
#         # same points
#         for s_idx in range(group_size):
#             same_writer.writerow([
#                 model_name,
#                 group_index,
#                 t_idx,
#                 f"T{t_idx:02d}",
#                 template,
#                 s_idx,
#                 f"S{s_idx}",
#                 style_words[s_idx],
#                 float(same_scores[t_idx, s_idx])
#             ])
#
#         # cross points
#         for r in cross_scores_by_template[t_idx]:
#             i = r["style_i_local"]
#             j = r["style_j_local"]
#
#             cross_writer.writerow([
#                 model_name,
#                 group_index,
#                 t_idx,
#                 f"T{t_idx:02d}",
#                 template,
#                 r["pair_index"],
#                 i,
#                 j,
#                 f"S{i}-S{j}",
#                 style_words[i],
#                 style_words[j],
#                 r["similarity"]
#             ])
#
#         # summary
#         summary_writer.writerow([
#             model_name,
#             group_index,
#             t_idx,
#             f"T{t_idx:02d}",
#             template,
#             float(same_avg[t_idx]),
#             float(cross_avg[t_idx]),
#             float(separation[t_idx])
#         ])
#
#
# def export_aggregate_template_ranking(
#     model_name: str,
#     model_dir: str,
#     templates: List[str],
#     aggregate_records: Dict[int, Dict[str, List[float]]]
# ):
#     """
#     汇总每个模板在 200 个 group 上的平均表现。
#     """
#     rows = []
#
#     for t_idx, template in enumerate(templates):
#         same_vals = aggregate_records[t_idx]["same"]
#         cross_vals = aggregate_records[t_idx]["cross"]
#         sep_vals = aggregate_records[t_idx]["sep"]
#
#         row = {
#             "template_index": t_idx,
#             "template": template,
#             "same_mean": statistics.mean(same_vals),
#             "same_std": statistics.pstdev(same_vals) if len(same_vals) > 1 else 0.0,
#             "cross_mean": statistics.mean(cross_vals),
#             "cross_std": statistics.pstdev(cross_vals) if len(cross_vals) > 1 else 0.0,
#             "sep_mean": statistics.mean(sep_vals),
#             "sep_std": statistics.pstdev(sep_vals) if len(sep_vals) > 1 else 0.0,
#         }
#
#         rows.append(row)
#
#     rows = sorted(rows, key=lambda x: x["sep_mean"], reverse=True)
#
#     path = os.path.join(model_dir, "aggregate_template_ranking.csv")
#
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#
#         writer.writerow([
#             "rank",
#             "model_name",
#             "template_index",
#             "template_id",
#             "template",
#             "same_mean",
#             "same_std",
#             "cross_mean",
#             "cross_std",
#             "sep_mean",
#             "sep_std"
#         ])
#
#         for rank, r in enumerate(rows, 1):
#             writer.writerow([
#                 rank,
#                 model_name,
#                 r["template_index"],
#                 f"T{r['template_index']:02d}",
#                 r["template"],
#                 r["same_mean"],
#                 r["same_std"],
#                 r["cross_mean"],
#                 r["cross_std"],
#                 r["sep_mean"],
#                 r["sep_std"]
#             ])
#
#     print(f"模型汇总排名已保存: {path}")
#
#
# # ==================== 单模型完整流程 ====================
#
# def process_one_model(
#     model_name: str,
#     sampled_descriptors: List[str],
#     groups: List[List[str]],
#     templates: List[str],
#     output_dir: str
# ):
#     print("\n" + "=" * 100)
#     print(f"开始处理模型: {model_name}")
#     print("=" * 100)
#
#     model_safe = safe_name(model_name)
#     model_dir = os.path.join(output_dir, model_safe)
#     fig_dir = os.path.join(model_dir, "figures")
#
#     ensure_dir(model_dir)
#     ensure_dir(fig_dir)
#
#     print("正在加载模型...")
#     model = SentenceTransformer(model_name)
#     print("模型加载完成。")
#
#     base_embeddings, template_embeddings = precompute_embeddings_for_model(
#         model=model,
#         sampled_descriptors=sampled_descriptors,
#         templates=templates
#     )
#
#     writers = open_model_csv_writers(model_dir)
#
#     aggregate_records = defaultdict(lambda: {
#         "same": [],
#         "cross": [],
#         "sep": []
#     })
#
#     try:
#         for group_index, style_words in enumerate(tqdm(groups, desc=f"Groups for {model_safe}")):
#             start_idx = group_index * GROUP_SIZE
#             group_indices = list(range(start_idx, start_idx + GROUP_SIZE))
#
#             same_scores, cross_scores_by_template, same_avg, cross_avg, separation = evaluate_one_group(
#                 group_indices=group_indices,
#                 base_embeddings=base_embeddings,
#                 template_embeddings=template_embeddings,
#                 templates=templates
#             )
#
#             write_group_results_to_csv(
#                 writers=writers,
#                 model_name=model_name,
#                 group_index=group_index,
#                 style_words=style_words,
#                 templates=templates,
#                 same_scores=same_scores,
#                 cross_scores_by_template=cross_scores_by_template,
#                 same_avg=same_avg,
#                 cross_avg=cross_avg,
#                 separation=separation
#             )
#
#             for t_idx in range(len(templates)):
#                 aggregate_records[t_idx]["same"].append(float(same_avg[t_idx]))
#                 aggregate_records[t_idx]["cross"].append(float(cross_avg[t_idx]))
#                 aggregate_records[t_idx]["sep"].append(float(separation[t_idx]))
#
#             fig_path = os.path.join(fig_dir, f"group_{group_index:03d}.png")
#
#             plot_group_combined_figure(
#                 model_name=model_name,
#                 group_index=group_index,
#                 style_words=style_words,
#                 templates=templates,
#                 same_scores=same_scores,
#                 cross_scores_by_template=cross_scores_by_template,
#                 same_avg=same_avg,
#                 cross_avg=cross_avg,
#                 separation=separation,
#                 output_path=fig_path
#             )
#
#     finally:
#         close_model_csv_writers(writers)
#
#     export_aggregate_template_ranking(
#         model_name=model_name,
#         model_dir=model_dir,
#         templates=templates,
#         aggregate_records=aggregate_records
#     )
#
#     print(f"模型处理完成: {model_name}")
#     print(f"图像目录: {fig_dir}")
#
#
# # ==================== 主程序 ====================
#
# def main():
#     ensure_dir(OUTPUT_DIR)
#
#     print("=" * 100)
#     print("大规模风格词语义相似度评估")
#     print("=" * 100)
#
#     print(f"输入文件: {INPUT_JSONL_PATH}")
#     print(f"输出目录: {OUTPUT_DIR}")
#     print(f"随机种子: {RANDOM_SEED}")
#     print(f"采样数量: {SAMPLE_SIZE}")
#     print(f"每组数量: {GROUP_SIZE}")
#     print(f"预期组数: {SAMPLE_SIZE // GROUP_SIZE}")
#     print(f"模型数量: {len(MODEL_NAMES)}")
#     print(f"预期生成图数: {(SAMPLE_SIZE // GROUP_SIZE) * len(MODEL_NAMES)}")
#
#     save_template_mapping(SENTENCE_TEMPLATES, OUTPUT_DIR)
#
#     descriptors = read_raw_descriptors(INPUT_JSONL_PATH)
#
#     sampled = sample_descriptors(
#         descriptors=descriptors,
#         sample_size=SAMPLE_SIZE,
#         seed=RANDOM_SEED,
#         dedup=DEDUP_DESCRIPTORS
#     )
#
#     groups = split_into_groups(sampled, GROUP_SIZE)
#
#     save_sampled_descriptors(
#         sampled=sampled,
#         groups=groups,
#         output_dir=OUTPUT_DIR
#     )
#
#     print("\n采样与分组完成。")
#     print(f"  实际采样数量: {len(sampled)}")
#     print(f"  实际分组数量: {len(groups)}")
#
#     for model_name in MODEL_NAMES:
#         process_one_model(
#             model_name=model_name,
#             sampled_descriptors=sampled,
#             groups=groups,
#             templates=SENTENCE_TEMPLATES,
#             output_dir=OUTPUT_DIR
#         )
#
#     print("\n" + "=" * 100)
#     print("全部评估完成！")
#     print("=" * 100)
#
#     print(f"\n输出目录: {OUTPUT_DIR}")
#     print("重点查看：")
#     for model_name in MODEL_NAMES:
#         model_safe = safe_name(model_name)
#         print(f"\n模型: {model_name}")
#         print(f"  图像目录: {os.path.join(OUTPUT_DIR, model_safe, 'figures')}")
#         print(f"  汇总排名: {os.path.join(OUTPUT_DIR, model_safe, 'aggregate_template_ranking.csv')}")
#         print(f"  每组模板汇总: {os.path.join(OUTPUT_DIR, model_safe, 'group_template_summary.csv')}")
#
#
# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
风格词语义一致性阈值分析脚本

输入：
semantic_similarity_large_scale/
├── sentence-transformers__all-mpnet-base-v2/
│   ├── same_points.csv
│   └── cross_points.csv
└── BAAI__bge-large-en-v1.5/
    ├── same_points.csv
    └── cross_points.csv

输出：
semantic_similarity_large_scale/threshold_analysis/
├── global_threshold_summary.csv
├── per_template_threshold_summary.csv
├── threshold_curve_xxx.csv
├── score_distribution_xxx.png
└── model_comparison_summary.csv
"""

import os
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================== 配置区域 ====================

BASE_DIR = "./semantic_similarity_large_scale"
OUTPUT_DIR = os.path.join(BASE_DIR, "threshold_analysis")

MODEL_DIRS = [
    "sentence-transformers__all-mpnet-base-v2",
    "BAAI__bge-large-en-v1.5",
]

# 如果你最终更重视“不误合并”，建议用这两个条件
TARGET_FPR_LIST = [0.01, 0.005, 0.001]
TARGET_PRECISION_LIST = [0.95, 0.98, 0.99]

# 阈值搜索步长。0.0005 已经比较细。
THRESHOLD_STEP = 0.0005


# ==================== 基础函数 ====================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def load_model_scores(model_dir: str) -> pd.DataFrame:
    """
    加载一个模型的 same_points.csv 和 cross_points.csv。

    返回 DataFrame:
    columns:
        score
        label: 1=same-style, 0=cross-style
        model_dir
        group_index
        template_index
        template_id
        template
        source
    """
    model_path = os.path.join(BASE_DIR, model_dir)

    same_path = os.path.join(model_path, "same_points.csv")
    cross_path = os.path.join(model_path, "cross_points.csv")

    if not os.path.exists(same_path):
        raise FileNotFoundError(same_path)
    if not os.path.exists(cross_path):
        raise FileNotFoundError(cross_path)

    same_df = pd.read_csv(same_path)
    cross_df = pd.read_csv(cross_path)

    same_records = pd.DataFrame({
        "score": same_df["same_style_similarity"].map(safe_float),
        "label": 1,
        "model_dir": model_dir,
        "group_index": same_df["group_index"],
        "template_index": same_df["template_index"],
        "template_id": same_df["template_id"],
        "template": same_df["template"],
        "source": "same"
    })

    cross_records = pd.DataFrame({
        "score": cross_df["cross_style_similarity"].map(safe_float),
        "label": 0,
        "model_dir": model_dir,
        "group_index": cross_df["group_index"],
        "template_index": cross_df["template_index"],
        "template_id": cross_df["template_id"],
        "template": cross_df["template"],
        "source": "cross"
    })

    df = pd.concat([same_records, cross_records], ignore_index=True)
    df = df.dropna(subset=["score"])
    df["score"] = df["score"].clip(-1.0, 1.0)

    return df


def compute_auc_by_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    计算 ROC-AUC。
    不依赖 sklearn。
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    pos = labels == 1
    neg = labels == 0

    n_pos = int(pos.sum())
    n_neg = int(neg.sum())

    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # 处理 ties：相同 score 使用平均 rank
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg_rank = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg_rank
        i = j

    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    return float(auc)


def compute_pr_auc(curve_df: pd.DataFrame) -> float:
    """
    根据 threshold curve 近似计算 PR-AUC。
    使用 recall-precision 曲线积分。
    """
    df = curve_df.sort_values("recall")
    recall = df["recall"].to_numpy()
    precision = df["precision"].to_numpy()

    valid = np.isfinite(recall) & np.isfinite(precision)
    recall = recall[valid]
    precision = precision[valid]

    if len(recall) < 2:
        return np.nan

    return float(np.trapz(precision, recall))


def build_thresholds(scores: np.ndarray, step: float = THRESHOLD_STEP) -> np.ndarray:
    """
    构造阈值搜索列表。
    """
    s_min = float(np.min(scores))
    s_max = float(np.max(scores))

    lo = math.floor((s_min - 0.001) / step) * step
    hi = math.ceil((s_max + 0.001) / step) * step

    thresholds = np.arange(lo, hi + step, step)

    return thresholds


def evaluate_thresholds(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    对所有阈值计算分类指标。
    score >= threshold 判定为 same-style。
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    thresholds = build_thresholds(scores)

    rows = []

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    for th in thresholds:
        pred = scores >= th

        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        acc = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
        balanced_acc = (recall + specificity) / 2.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and (precision + recall) > 0
            else 0.0
        )

        youden_j = recall - fpr

        rows.append({
            "threshold": float(th),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "fpr": fpr,
            "fnr": fnr,
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "f1": f1,
            "youden_j": youden_j,
            "positive_count": n_pos,
            "negative_count": n_neg,
        })

    return pd.DataFrame(rows)


def select_best_thresholds(curve_df: pd.DataFrame) -> Dict:
    """
    从 threshold curve 中选出多种推荐阈值。
    """
    result = {}

    # 1. F1 最优
    idx = curve_df["f1"].idxmax()
    row = curve_df.loc[idx]
    result["best_f1_threshold"] = row["threshold"]
    result["best_f1"] = row["f1"]
    result["best_f1_precision"] = row["precision"]
    result["best_f1_recall"] = row["recall"]
    result["best_f1_fpr"] = row["fpr"]

    # 2. Balanced Accuracy 最优
    idx = curve_df["balanced_accuracy"].idxmax()
    row = curve_df.loc[idx]
    result["best_balanced_acc_threshold"] = row["threshold"]
    result["best_balanced_acc"] = row["balanced_accuracy"]
    result["best_balanced_acc_precision"] = row["precision"]
    result["best_balanced_acc_recall"] = row["recall"]
    result["best_balanced_acc_fpr"] = row["fpr"]

    # 3. Youden J 最优
    idx = curve_df["youden_j"].idxmax()
    row = curve_df.loc[idx]
    result["best_youden_threshold"] = row["threshold"]
    result["best_youden_j"] = row["youden_j"]
    result["best_youden_precision"] = row["precision"]
    result["best_youden_recall"] = row["recall"]
    result["best_youden_fpr"] = row["fpr"]

    # 4. EER 阈值：FPR 和 FNR 最接近
    eer_idx = (curve_df["fpr"] - curve_df["fnr"]).abs().idxmin()
    row = curve_df.loc[eer_idx]
    result["eer_threshold"] = row["threshold"]
    result["eer_fpr"] = row["fpr"]
    result["eer_fnr"] = row["fnr"]
    result["eer_precision"] = row["precision"]
    result["eer_recall"] = row["recall"]

    # 5. FPR 约束阈值
    for target_fpr in TARGET_FPR_LIST:
        subset = curve_df[curve_df["fpr"] <= target_fpr].copy()

        key = str(target_fpr).replace(".", "_")

        if len(subset) == 0:
            result[f"threshold_at_fpr_{key}"] = np.nan
            result[f"recall_at_fpr_{key}"] = np.nan
            result[f"precision_at_fpr_{key}"] = np.nan
            result[f"actual_fpr_{key}"] = np.nan
        else:
            # 在满足 FPR 的阈值中，选 recall 最大的
            idx = subset["recall"].idxmax()
            row = subset.loc[idx]

            result[f"threshold_at_fpr_{key}"] = row["threshold"]
            result[f"recall_at_fpr_{key}"] = row["recall"]
            result[f"precision_at_fpr_{key}"] = row["precision"]
            result[f"actual_fpr_{key}"] = row["fpr"]

    # 6. Precision 约束阈值
    for target_precision in TARGET_PRECISION_LIST:
        subset = curve_df[curve_df["precision"] >= target_precision].copy()

        key = str(target_precision).replace(".", "_")

        if len(subset) == 0:
            result[f"threshold_at_precision_{key}"] = np.nan
            result[f"recall_at_precision_{key}"] = np.nan
            result[f"actual_precision_{key}"] = np.nan
            result[f"fpr_at_precision_{key}"] = np.nan
        else:
            # 在满足 Precision 的阈值中，选 recall 最大的
            idx = subset["recall"].idxmax()
            row = subset.loc[idx]

            result[f"threshold_at_precision_{key}"] = row["threshold"]
            result[f"recall_at_precision_{key}"] = row["recall"]
            result[f"actual_precision_{key}"] = row["precision"]
            result[f"fpr_at_precision_{key}"] = row["fpr"]

    return result


def describe_score_distribution(scores: np.ndarray) -> Dict:
    scores = np.asarray(scores, dtype=float)

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "p01": float(np.percentile(scores, 1)),
        "p05": float(np.percentile(scores, 5)),
        "p10": float(np.percentile(scores, 10)),
        "p25": float(np.percentile(scores, 25)),
        "median": float(np.percentile(scores, 50)),
        "p75": float(np.percentile(scores, 75)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "max": float(np.max(scores)),
    }


def analyze_one_dataframe(
    df: pd.DataFrame,
    model_dir: str,
    scope_name: str,
    output_prefix: str,
    save_curve: bool = True
) -> Dict:
    """
    分析一个分数集合，可以是全局，也可以是单个模板。
    """
    scores = df["score"].to_numpy(dtype=float)
    labels = df["label"].to_numpy(dtype=int)

    curve_df = evaluate_thresholds(scores, labels)

    if save_curve:
        curve_path = os.path.join(
            OUTPUT_DIR,
            f"threshold_curve_{output_prefix}.csv"
        )
        curve_df.to_csv(curve_path, index=False)

    auc = compute_auc_by_rank(scores, labels)
    pr_auc = compute_pr_auc(curve_df)

    best = select_best_thresholds(curve_df)

    pos_scores = df[df["label"] == 1]["score"].to_numpy(dtype=float)
    neg_scores = df[df["label"] == 0]["score"].to_numpy(dtype=float)

    pos_desc = describe_score_distribution(pos_scores)
    neg_desc = describe_score_distribution(neg_scores)

    # 分布重叠的一种简单描述：
    # 正类低于负类 95 分位的比例，以及负类高于正类 5 分位的比例
    neg_p95 = np.percentile(neg_scores, 95)
    pos_p05 = np.percentile(pos_scores, 5)

    pos_below_neg_p95 = float((pos_scores <= neg_p95).mean())
    neg_above_pos_p05 = float((neg_scores >= pos_p05).mean())

    record = {
        "model_dir": model_dir,
        "scope": scope_name,
        "num_positive": int((labels == 1).sum()),
        "num_negative": int((labels == 0).sum()),
        "roc_auc": auc,
        "pr_auc": pr_auc,

        "positive_mean": pos_desc["mean"],
        "positive_std": pos_desc["std"],
        "positive_p05": pos_desc["p05"],
        "positive_median": pos_desc["median"],
        "positive_p95": pos_desc["p95"],

        "negative_mean": neg_desc["mean"],
        "negative_std": neg_desc["std"],
        "negative_p05": neg_desc["p05"],
        "negative_median": neg_desc["median"],
        "negative_p95": neg_desc["p95"],

        "mean_gap": pos_desc["mean"] - neg_desc["mean"],
        "median_gap": pos_desc["median"] - neg_desc["median"],
        "pos_below_neg_p95": pos_below_neg_p95,
        "neg_above_pos_p05": neg_above_pos_p05,
    }

    record.update(best)

    # 推荐自动去重阈值
    # 取 FPR<=1% 阈值 和 Precision>=98% 阈值中更严格的那个
    fpr_key = "threshold_at_fpr_0_01"
    precision_key = "threshold_at_precision_0_98"

    candidates = []
    if fpr_key in record and np.isfinite(record[fpr_key]):
        candidates.append(record[fpr_key])
    if precision_key in record and np.isfinite(record[precision_key]):
        candidates.append(record[precision_key])

    if candidates:
        record["recommended_auto_merge_threshold"] = max(candidates)
    else:
        record["recommended_auto_merge_threshold"] = record["best_f1_threshold"]

    # 灰区下界用 F1 阈值
    record["gray_zone_lower_threshold"] = record["best_f1_threshold"]
    record["gray_zone_upper_threshold"] = record["recommended_auto_merge_threshold"]

    return record


def plot_score_distribution(
    df: pd.DataFrame,
    model_dir: str,
    output_prefix: str,
    threshold_record: Dict
):
    """
    绘制 same/cross 分布图，并标出几个关键阈值。
    """
    pos_scores = df[df["label"] == 1]["score"].to_numpy(dtype=float)
    neg_scores = df[df["label"] == 0]["score"].to_numpy(dtype=float)

    plt.figure(figsize=(12, 6))

    bins = np.linspace(
        min(df["score"].min(), 0),
        max(df["score"].max(), 1),
        80
    )

    plt.hist(
        pos_scores,
        bins=bins,
        alpha=0.55,
        density=True,
        label="Same-style / positive"
    )

    plt.hist(
        neg_scores,
        bins=bins,
        alpha=0.55,
        density=True,
        label="Cross-style / negative"
    )

    key_thresholds = [
        ("Best F1", "best_f1_threshold"),
        ("FPR<=1%", "threshold_at_fpr_0_01"),
        ("Precision>=98%", "threshold_at_precision_0_98"),
        ("Recommended", "recommended_auto_merge_threshold"),
    ]

    for name, key in key_thresholds:
        th = threshold_record.get(key, np.nan)
        if np.isfinite(th):
            plt.axvline(
                th,
                linestyle="--",
                linewidth=1.8,
                label=f"{name}: {th:.4f}"
            )

    plt.xlabel("Similarity score")
    plt.ylabel("Density")
    plt.title(f"Score Distribution | {model_dir} | {output_prefix}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"score_distribution_{output_prefix}.png")
    plt.savefig(path, dpi=220)
    plt.close()


def analyze_model(model_dir: str):
    print("\n" + "=" * 100)
    print(f"分析模型: {model_dir}")
    print("=" * 100)

    df = load_model_scores(model_dir)

    print(f"总样本数: {len(df)}")
    print(f"正类 same-style: {(df['label'] == 1).sum()}")
    print(f"负类 cross-style: {(df['label'] == 0).sum()}")

    model_records = []
    per_template_records = []

    # 1. 全局阈值
    global_record = analyze_one_dataframe(
        df=df,
        model_dir=model_dir,
        scope_name="global",
        output_prefix=f"{model_dir}__global",
        save_curve=True
    )

    model_records.append(global_record)

    plot_score_distribution(
        df=df,
        model_dir=model_dir,
        output_prefix=f"{model_dir}__global",
        threshold_record=global_record
    )

    # 2. 每个模板单独计算阈值
    for template_index in sorted(df["template_index"].unique()):
        tdf = df[df["template_index"] == template_index].copy()

        template_id = str(tdf["template_id"].iloc[0])
        template = str(tdf["template"].iloc[0])

        record = analyze_one_dataframe(
            df=tdf,
            model_dir=model_dir,
            scope_name=template_id,
            output_prefix=f"{model_dir}__{template_id}",
            save_curve=False
        )

        record["template_index"] = template_index
        record["template_id"] = template_id
        record["template"] = template

        per_template_records.append(record)

    return global_record, per_template_records


def main():
    ensure_dir(OUTPUT_DIR)

    all_global_records = []
    all_template_records = []

    for model_dir in MODEL_DIRS:
        global_record, per_template_records = analyze_model(model_dir)

        all_global_records.append(global_record)
        all_template_records.extend(per_template_records)

    global_df = pd.DataFrame(all_global_records)
    template_df = pd.DataFrame(all_template_records)

    global_path = os.path.join(OUTPUT_DIR, "global_threshold_summary.csv")
    template_path = os.path.join(OUTPUT_DIR, "per_template_threshold_summary.csv")

    global_df.to_csv(global_path, index=False)
    template_df.to_csv(template_path, index=False)

    print("\n" + "=" * 100)
    print("全局阈值结果")
    print("=" * 100)

    show_cols = [
        "model_dir",
        "roc_auc",
        "pr_auc",
        "positive_mean",
        "negative_mean",
        "mean_gap",
        "best_f1_threshold",
        "best_f1",
        "best_f1_precision",
        "best_f1_recall",
        "best_f1_fpr",
        "threshold_at_fpr_0_01",
        "recall_at_fpr_0_01",
        "precision_at_fpr_0_01",
        "threshold_at_precision_0_98",
        "recall_at_precision_0_98",
        "fpr_at_precision_0_98",
        "recommended_auto_merge_threshold",
        "gray_zone_lower_threshold",
        "gray_zone_upper_threshold",
    ]

    available_cols = [c for c in show_cols if c in global_df.columns]

    print(global_df[available_cols].to_string(index=False))

    print("\n输出文件：")
    print(f"  全局阈值: {global_path}")
    print(f"  模板阈值: {template_path}")
    print(f"  分布图目录: {OUTPUT_DIR}")

    print("\n建议读取方式：")
    print("  1. 先看 global_threshold_summary.csv 里的 recommended_auto_merge_threshold")
    print("  2. 如果你实际使用固定模板，再看 per_template_threshold_summary.csv 中对应模板的阈值")
    print("  3. 自动去重建议使用 recommended_auto_merge_threshold")
    print("  4. 灰区建议使用 gray_zone_lower_threshold 到 gray_zone_upper_threshold")


if __name__ == "__main__":
    main()