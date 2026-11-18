from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# scipy 可选但推荐
try:
    from scipy.stats import f_oneway
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# umap 可选
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# hdbscan 可选
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False


# ============================================================
# 配置区域
# ============================================================

INPUT_JSONL_PATH = "./dataset_experiments/fragment_level_records.jsonl"
BASE_OUTPUT_DIR = "./experiment_outputs/fragment_analysis"

TOP_N = 100
RANDOM_SEED = 42

EFFECT_DIMS = [
    "artistic_context",
    "realism",
    "stylization",
    "atmospheric_intensity",
    "sensory_salience",
    "platform_association",
    "depiction_explicitness",
    "violence_graphicness",
]

MECHANISM_TO_EFFECT_CHECKS = [
    ("reference_anchor", "artistic_context"),
    ("perceptual_cue", "sensory_salience"),
    ("affective_atmosphere", "atmospheric_intensity"),
    ("platform_context", "platform_association"),
]

# 可视化抽样上限
MAX_POINTS_PCA = 15000
MAX_POINTS_UMAP = 12000
MAX_POINTS_TSNE = 5000

# 聚类参数
KMEANS_K = 8
GMM_K = 8
HDBSCAN_MIN_CLUSTER_SIZE = 30
HDBSCAN_MIN_SAMPLES = 10

# t-SNE 参数
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1500

# UMAP 参数
UMAP_N_NEIGHBORS = 25
UMAP_MIN_DIST = 0.1

# ============================================================
# 文本 embedding 可视化参数
# ============================================================
TEXT_EMBED_TEST_SIZE = 0.2
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
# 也可换成：
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

EMBEDDING_BATCH_SIZE = 128
NORMALIZE_TEXT_EMBEDDINGS = True
FORCE_RECOMPUTE_TEXT_EMBEDDINGS = False

TEXT_EMBED_MAX_POINTS_PCA = 15000
TEXT_EMBED_MAX_POINTS_UMAP = 12000
TEXT_EMBED_MAX_POINTS_TSNE = 5000

TEXT_EMBED_KMEANS_K = 8

TEXT_EMBED_TSNE_PERPLEXITY = 30
TEXT_EMBED_TSNE_N_ITER = 1500

TEXT_EMBED_UMAP_N_NEIGHBORS = 25
TEXT_EMBED_UMAP_MIN_DIST = 0.1
# ============================================================
# 路径管理
# ============================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


BASE_OUTPUT_DIR = ensure_dir(BASE_OUTPUT_DIR)
BASIC_STATS_DIR = ensure_dir(BASE_OUTPUT_DIR / "basic_statistics")
MECH_EFFECT_DIR = ensure_dir(BASE_OUTPUT_DIR / "mechanism_effect_analysis")
GRAN_EFFECT_DIR = ensure_dir(BASE_OUTPUT_DIR / "granularity_effect_analysis")
CLUSTER_DIR = ensure_dir(BASE_OUTPUT_DIR / "clustering_visualization")
TEXT_EMBED_DIR = ensure_dir(BASE_OUTPUT_DIR / "text_embedding_visualization")


# ============================================================
# 基本工具
# ============================================================

def save_json(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_jsonl_records(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_idx}: {e}") from e
    return records


def normalize_sample_ids(value: Any) -> List[str]:
    if isinstance(value, list):
        out = []
        for x in value:
            if x is None:
                continue
            x = str(x).strip()
            if x:
                out.append(x)
        return out
    elif value is None:
        return []
    else:
        x = str(value).strip()
        return [x] if x else []


def safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def sample_dataframe(df: pd.DataFrame, max_points: int, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()
    return df.sample(n=max_points, random_state=random_seed).copy()


def compute_eta_squared(groups: List[np.ndarray]) -> Optional[float]:
    valid_groups = [g for g in groups if len(g) > 0]
    if len(valid_groups) < 2:
        return None

    all_vals = np.concatenate(valid_groups)
    grand_mean = np.mean(all_vals)

    ss_between = 0.0
    for g in valid_groups:
        ss_between += len(g) * (np.mean(g) - grand_mean) ** 2

    ss_total = np.sum((all_vals - grand_mean) ** 2)
    if ss_total <= 1e-12:
        return 0.0

    return float(ss_between / ss_total)


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2 or len(y) < 2:
        return None
    nx, ny = len(x), len(y)
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / math.sqrt(pooled))

def make_model_tag(model_name: str) -> str:
    return (
        model_name.replace("/", "__")
        .replace("\\", "__")
        .replace("-", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


def compute_text_embeddings(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    对 raw_descriptor 生成文本 embedding。
    返回：
        meta_df: 与 embedding 一一对应的元数据表
        embeddings: numpy array, shape = [N, D]
    """
    text_df = df.copy()
    text_df = text_df[text_df["raw_descriptor"].notna()].copy()
    text_df["raw_descriptor"] = text_df["raw_descriptor"].astype(str)

    text_df = text_df.reset_index(drop=True)

    model_tag = make_model_tag(EMBEDDING_MODEL_NAME)
    emb_path = TEXT_EMBED_DIR / f"text_embeddings__{model_tag}.npy"
    meta_path = TEXT_EMBED_DIR / f"text_embedding_metadata__{model_tag}.csv"
    summary_path = TEXT_EMBED_DIR / f"text_embedding_summary__{model_tag}.json"

    if (
        emb_path.exists()
        and meta_path.exists()
        and not FORCE_RECOMPUTE_TEXT_EMBEDDINGS
    ):
        meta_df = pd.read_csv(meta_path)
        embeddings = np.load(emb_path)
        if len(meta_df) == len(embeddings):
            print(f"[INFO] Loaded cached text embeddings: {emb_path}")
            return meta_df, embeddings

    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = text_df["raw_descriptor"].tolist()

    print(f"[INFO] Encoding {len(texts)} raw_descriptor texts...")
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE_TEXT_EMBEDDINGS,
    )

    meta_df = text_df[
        [
            "fragment_id",
            "raw_descriptor",
            "mechanism_category",
            "granularity",
            "support_count",
            "is_merged",
        ]
    ].copy()

    meta_df.to_csv(meta_path, index=False)
    np.save(emb_path, embeddings)

    save_json(
        {
            "embedding_model_name": EMBEDDING_MODEL_NAME,
            "num_texts": int(len(texts)),
            "embedding_shape": list(embeddings.shape),
            "normalize_embeddings": NORMALIZE_TEXT_EMBEDDINGS,
            "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        },
        summary_path,
    )

    return meta_df, embeddings
def compute_cluster_purity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ignore_cluster_labels: Optional[set] = None,
) -> Dict[str, Any]:
    """
    计算 cluster 与真实 mechanism 标签的一致性。
    返回：
        ari / nmi / purity / 有效样本数
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = np.ones(len(y_true), dtype=bool)
    if ignore_cluster_labels is not None and len(ignore_cluster_labels) > 0:
        for bad in ignore_cluster_labels:
            mask &= (y_pred != bad)

    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    if len(y_true_valid) == 0:
        return {
            "num_valid_samples": 0,
            "ari": None,
            "nmi": None,
            "purity": None,
        }

    ari = adjusted_rand_score(y_true_valid, y_pred_valid)
    nmi = normalized_mutual_info_score(y_true_valid, y_pred_valid)

    contingency = pd.crosstab(
        pd.Series(y_pred_valid, name="cluster"),
        pd.Series(y_true_valid, name="mechanism"),
    )
    purity = contingency.max(axis=1).sum() / contingency.to_numpy().sum()

    return {
        "num_valid_samples": int(len(y_true_valid)),
        "ari": float(ari),
        "nmi": float(nmi),
        "purity": float(purity),
    }


def save_confusion_matrix_plot(
    cm: np.ndarray,
    labels: List[str],
    out_path: Union[str, Path],
    title: str,
) -> None:
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)

    # 可选：格子里写数字
    max_val = cm.max() if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if max_val > 0 and val > max_val * 0.5 else "black"
            plt.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def evaluate_text_embedding_clusters(
    work_df: pd.DataFrame,
    out_dir: Union[str, Path],
) -> None:
    """
    评估无监督 cluster 与 mechanism_category 的一致性：
    - KMeans vs mechanism
    - HDBSCAN vs mechanism（若可用）
    """
    out_dir = Path(out_dir)
    y_true = work_df["mechanism_category"].astype(str).values

    # KMeans
    y_kmeans = work_df["kmeans_cluster"].values
    kmeans_metrics = compute_cluster_purity(y_true, y_kmeans)
    save_json(kmeans_metrics, out_dir / "text_embedding_kmeans_vs_mechanism_metrics.json")

    kmeans_cross = pd.crosstab(
        work_df["kmeans_cluster"],
        work_df["mechanism_category"],
    )
    kmeans_cross.to_csv(out_dir / "text_embedding_kmeans_vs_mechanism_crosstab.csv")

    # HDBSCAN
    if "hdbscan_cluster" in work_df.columns:
        y_hdb = work_df["hdbscan_cluster"].values
        # 忽略噪声标签 -1 和占位标签 -999
        hdb_metrics = compute_cluster_purity(
            y_true,
            y_hdb,
            ignore_cluster_labels={-1, -999},
        )
        save_json(hdb_metrics, out_dir / "text_embedding_hdbscan_vs_mechanism_metrics.json")

        hdb_valid = work_df[~work_df["hdbscan_cluster"].isin([-1, -999])].copy()
        if len(hdb_valid) > 0:
            hdb_cross = pd.crosstab(
                hdb_valid["hdbscan_cluster"],
                hdb_valid["mechanism_category"],
            )
            hdb_cross.to_csv(out_dir / "text_embedding_hdbscan_vs_mechanism_crosstab.csv")


def supervised_text_embedding_classification(
    work_df: pd.DataFrame,
    X: np.ndarray,
    out_dir: Union[str, Path],
) -> None:
    """
    用文本 embedding -> mechanism_category 做监督分类，
    输出：
    - accuracy
    - macro_f1
    - weighted_f1
    - classification_report
    - confusion_matrix
    """
    out_dir = Path(out_dir)

    y = work_df["mechanism_category"].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEXT_EMBED_TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        multi_class="auto",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = {
        "model": "LogisticRegression",
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
    }
    save_json(metrics, out_dir / "text_embedding_supervised_classification_metrics.json")

    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        out_dir / "text_embedding_supervised_classification_report.csv",
        index=True,
    )

    labels = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        out_dir / "text_embedding_supervised_confusion_matrix.csv",
        index=True,
    )

    save_confusion_matrix_plot(
        cm,
        labels,
        out_dir / "text_embedding_supervised_confusion_matrix.png",
        "Confusion Matrix: Text Embedding -> Mechanism Category",
    )

def clustering_and_visualization_text_embedding(df: pd.DataFrame) -> None:
    """
    基于 raw_descriptor 的文本 embedding 做：
    - PCA
    - UMAP
    - t-SNE
    - KMeans
    - HDBSCAN(可选)
    并按 mechanism_category / cluster 分别可视化。
    """
    meta_df, X = compute_text_embeddings(df)
    if len(meta_df) == 0:
        raise ValueError("No valid rows for text embedding visualization.")

    work_df = meta_df.copy()

    # ------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------
    pca_model = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca_model.fit_transform(X)
    work_df["pca_1"] = X_pca[:, 0]
    work_df["pca_2"] = X_pca[:, 1]

    save_json(
        {
            "embedding_model_name": EMBEDDING_MODEL_NAME,
            "explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
            "components_shape": list(pca_model.components_.shape),
        },
        TEXT_EMBED_DIR / "text_embedding_pca_info.json",
    )

    # ------------------------------------------------------------
    # KMeans
    # ------------------------------------------------------------
    kmeans = KMeans(
        n_clusters=TEXT_EMBED_KMEANS_K,
        random_state=RANDOM_SEED,
        n_init=20,
    )
    work_df["kmeans_cluster"] = kmeans.fit_predict(X)

    centers_df = pd.DataFrame(kmeans.cluster_centers_)
    centers_df.insert(0, "cluster_id", range(len(centers_df)))
    centers_df.to_csv(TEXT_EMBED_DIR / "text_embedding_kmeans_centers.csv", index=False)

    # ------------------------------------------------------------
    # HDBSCAN（可选）
    # ------------------------------------------------------------
    if HDBSCAN_AVAILABLE:
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
        )
        work_df["hdbscan_cluster"] = hdb.fit_predict(X)
    else:
        work_df["hdbscan_cluster"] = -999

    # 保存带聚类标签的全量表
    work_df.to_csv(TEXT_EMBED_DIR / "text_embedding_analysis_with_clusters.csv", index=False)

    # ------------------------------------------------------------
    # 无监督聚类 vs mechanism 的一致性评估
    # ------------------------------------------------------------
    evaluate_text_embedding_clusters(work_df, TEXT_EMBED_DIR)

    # ------------------------------------------------------------
    # 监督分类：embedding -> mechanism_category
    # ------------------------------------------------------------
    supervised_text_embedding_classification(work_df, X, TEXT_EMBED_DIR)

    # ------------------------------------------------------------
    # PCA 可视化
    # ------------------------------------------------------------
    plot_df = sample_dataframe(work_df, TEXT_EMBED_MAX_POINTS_PCA)

    scatter_by_category(
        plot_df,
        "pca_1",
        "pca_2",
        "mechanism_category",
        TEXT_EMBED_DIR / "text_embedding_pca_by_mechanism_category.png",
        "PCA Projection by Mechanism Category (Text Embedding)",
    )
    scatter_by_category(
        plot_df,
        "pca_1",
        "pca_2",
        "kmeans_cluster",
        TEXT_EMBED_DIR / "text_embedding_pca_by_kmeans_cluster.png",
        "PCA Projection by KMeans Cluster (Text Embedding)",
    )

    # ------------------------------------------------------------
    # UMAP 可视化
    # ------------------------------------------------------------
    if UMAP_AVAILABLE:
        umap_df = sample_dataframe(work_df, TEXT_EMBED_MAX_POINTS_UMAP)
        umap_idx = umap_df.index.to_numpy()
        X_umap_input = X[umap_idx]

        umap_model = umap.UMAP(
            n_neighbors=TEXT_EMBED_UMAP_N_NEIGHBORS,
            min_dist=TEXT_EMBED_UMAP_MIN_DIST,
            n_components=2,
            random_state=RANDOM_SEED,
        )
        X_umap = umap_model.fit_transform(X_umap_input)

        umap_df = umap_df.copy()
        umap_df["umap_1"] = X_umap[:, 0]
        umap_df["umap_2"] = X_umap[:, 1]
        umap_df.to_csv(TEXT_EMBED_DIR / "text_embedding_umap_projection_sample.csv", index=False)

        scatter_by_category(
            umap_df,
            "umap_1",
            "umap_2",
            "mechanism_category",
            TEXT_EMBED_DIR / "text_embedding_umap_by_mechanism_category.png",
            "UMAP Projection by Mechanism Category (Text Embedding)",
        )
        scatter_by_category(
            umap_df,
            "umap_1",
            "umap_2",
            "kmeans_cluster",
            TEXT_EMBED_DIR / "text_embedding_umap_by_kmeans_cluster.png",
            "UMAP Projection by KMeans Cluster (Text Embedding)",
        )

    # ------------------------------------------------------------
    # t-SNE 可视化
    # ------------------------------------------------------------
    tsne_df = sample_dataframe(work_df, TEXT_EMBED_MAX_POINTS_TSNE)
    tsne_idx = tsne_df.index.to_numpy()
    X_tsne_input = X[tsne_idx]

    tsne_model = TSNE(
        n_components=2,
        perplexity=TEXT_EMBED_TSNE_PERPLEXITY,
        n_iter=TEXT_EMBED_TSNE_N_ITER,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
    )
    X_tsne = tsne_model.fit_transform(X_tsne_input)

    tsne_df = tsne_df.copy()
    tsne_df["tsne_1"] = X_tsne[:, 0]
    tsne_df["tsne_2"] = X_tsne[:, 1]
    tsne_df.to_csv(TEXT_EMBED_DIR / "text_embedding_tsne_projection_sample.csv", index=False)

    scatter_by_category(
        tsne_df,
        "tsne_1",
        "tsne_2",
        "mechanism_category",
        TEXT_EMBED_DIR / "text_embedding_tsne_by_mechanism_category.png",
        "t-SNE Projection by Mechanism Category (Text Embedding)",
    )
    scatter_by_category(
        tsne_df,
        "tsne_1",
        "tsne_2",
        "kmeans_cluster",
        TEXT_EMBED_DIR / "text_embedding_tsne_by_kmeans_cluster.png",
        "t-SNE Projection by KMeans Cluster (Text Embedding)",
    )

    cluster_stats = {
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "num_rows_used": int(len(work_df)),
        "embedding_dim": int(X.shape[1]),
        "kmeans_cluster_counts": work_df["kmeans_cluster"].value_counts().sort_index().to_dict(),
        "hdbscan_cluster_counts": work_df["hdbscan_cluster"].value_counts().sort_index().to_dict(),
        "umap_available": UMAP_AVAILABLE,
        "hdbscan_available": HDBSCAN_AVAILABLE,
    }
    save_json(cluster_stats, TEXT_EMBED_DIR / "text_embedding_cluster_stats.json")

# ============================================================
# 读取并扁平化
# ============================================================

def build_flat_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for rec in records:
        sample_ids = normalize_sample_ids(rec.get("sample_id"))
        effect_profile = rec.get("effect_profile", {})

        row = {
            "sample_id": sample_ids,
            "support_count": len(sample_ids),
            "is_merged": int(len(sample_ids) > 1),
            "fragment_id": rec.get("fragment_id"),
            "raw_descriptor": rec.get("raw_descriptor"),
            "mechanism_category": rec.get("mechanism_category"),
            "granularity": rec.get("granularity"),
        }

        for dim in EFFECT_DIMS:
            row[dim] = safe_float(effect_profile.get(dim))

        rows.append(row)

    df = pd.DataFrame(rows)
    df["support_count"] = df["support_count"].fillna(0).astype(int)
    df["is_merged"] = df["is_merged"].fillna(0).astype(int)

    for dim in EFFECT_DIMS:
        df[dim] = pd.to_numeric(df[dim], errors="coerce")

    return df


# ============================================================
# 基础统计
# ============================================================

def basic_statistics(df: pd.DataFrame) -> None:
    # ------------------------------------------------------------
    # 1. mechanism_category 分布
    # ------------------------------------------------------------
    mech_dist = (
        df["mechanism_category"]
        .value_counts(dropna=False)
        .rename_axis("mechanism_category")
        .reset_index(name="count")
    )
    mech_dist["ratio"] = mech_dist["count"] / len(df)
    mech_dist.to_csv(BASIC_STATS_DIR / "mechanism_category_distribution.csv", index=False)

    # 按 support_count 加权后的 mechanism_category 分布
    mech_weighted = (
        df.groupby("mechanism_category", dropna=False)["support_count"]
        .sum()
        .reset_index(name="weighted_support")
        .sort_values("weighted_support", ascending=False)
    )
    mech_weighted["weighted_ratio"] = mech_weighted["weighted_support"] / mech_weighted["weighted_support"].sum()
    mech_weighted.to_csv(BASIC_STATS_DIR / "mechanism_category_distribution_weighted_by_support.csv", index=False)

    # ------------------------------------------------------------
    # 2. granularity 分布
    # ------------------------------------------------------------
    gran_dist = (
        df["granularity"]
        .value_counts(dropna=False)
        .rename_axis("granularity")
        .reset_index(name="count")
    )
    gran_dist["ratio"] = gran_dist["count"] / len(df)
    gran_dist.to_csv(BASIC_STATS_DIR / "granularity_distribution.csv", index=False)

    # 按 support_count 加权后的 granularity 分布
    gran_weighted = (
        df.groupby("granularity", dropna=False)["support_count"]
        .sum()
        .reset_index(name="weighted_support")
        .sort_values("weighted_support", ascending=False)
    )
    gran_weighted["weighted_ratio"] = gran_weighted["weighted_support"] / gran_weighted["weighted_support"].sum()
    gran_weighted.to_csv(BASIC_STATS_DIR / "granularity_distribution_weighted_by_support.csv", index=False)

    # ------------------------------------------------------------
    # 3. support_count 统计摘要
    # ------------------------------------------------------------
    singleton_count = int((df["support_count"] == 1).sum())
    repeated_count = int((df["support_count"] > 1).sum())

    support_summary = {
        "num_fragments": int(len(df)),
        "num_singleton_fragments": singleton_count,
        "num_repeated_fragments": repeated_count,
        "singleton_ratio": float(singleton_count / len(df)) if len(df) > 0 else 0.0,
        "repeated_ratio": float(repeated_count / len(df)) if len(df) > 0 else 0.0,
        "support_count_mean": float(df["support_count"].mean()),
        "support_count_std": float(df["support_count"].std(ddof=0)),
        "support_count_median": float(df["support_count"].median()),
        "support_count_min": int(df["support_count"].min()),
        "support_count_max": int(df["support_count"].max()),
        "support_count_q90": float(df["support_count"].quantile(0.90)),
        "support_count_q95": float(df["support_count"].quantile(0.95)),
        "support_count_q99": float(df["support_count"].quantile(0.99)),
    }
    save_json(support_summary, BASIC_STATS_DIR / "support_count_summary.json")

    # support_count 分桶统计
    support_bucket_rows = [
        {"bucket": "support=1", "count": int((df["support_count"] == 1).sum())},
        {"bucket": "support=2", "count": int((df["support_count"] == 2).sum())},
        {"bucket": "support=3-4", "count": int(((df["support_count"] >= 3) & (df["support_count"] <= 4)).sum())},
        {"bucket": "support=5-9", "count": int(((df["support_count"] >= 5) & (df["support_count"] <= 9)).sum())},
        {"bucket": "support=10-19", "count": int(((df["support_count"] >= 10) & (df["support_count"] <= 19)).sum())},
        {"bucket": "support=20-49", "count": int(((df["support_count"] >= 20) & (df["support_count"] <= 49)).sum())},
        {"bucket": "support>=50", "count": int((df["support_count"] >= 50).sum())},
    ]
    support_bucket_df = pd.DataFrame(support_bucket_rows)
    support_bucket_df["ratio"] = support_bucket_df["count"] / len(df)
    support_bucket_df.to_csv(BASIC_STATS_DIR / "support_count_bucket_summary.csv", index=False)

    # ------------------------------------------------------------
    # 4. effect_profile 整体分布摘要
    # ------------------------------------------------------------
    effect_summary_rows = []
    for dim in EFFECT_DIMS:
        vals = df[dim].dropna()
        effect_summary_rows.append({
            "dimension": dim,
            "count": int(vals.shape[0]),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "q25": float(vals.quantile(0.25)),
            "median": float(vals.median()),
            "q75": float(vals.quantile(0.75)),
            "max": float(vals.max()),
        })
    pd.DataFrame(effect_summary_rows).to_csv(BASIC_STATS_DIR / "effect_profile_overall_summary.csv", index=False)

    # ------------------------------------------------------------
    # 5. 高频 fragment Top-N
    # ------------------------------------------------------------
    top_frag = (
        df.sort_values(["support_count", "raw_descriptor"], ascending=[False, True])
          .loc[:, ["raw_descriptor", "mechanism_category", "granularity", "support_count"] + EFFECT_DIMS]
          .head(TOP_N)
    )
    top_frag.to_csv(BASIC_STATS_DIR / f"top_{TOP_N}_fragments_by_support.csv", index=False)

    # 仅保留 support_count > 1 的高频 fragment
    top_frag_gt1 = (
        df[df["support_count"] > 1]
        .sort_values(["support_count", "raw_descriptor"], ascending=[False, True])
        .loc[:, ["raw_descriptor", "mechanism_category", "granularity", "support_count"] + EFFECT_DIMS]
        .head(TOP_N)
    )
    top_frag_gt1.to_csv(BASIC_STATS_DIR / f"top_{TOP_N}_fragments_by_support_gt1.csv", index=False)

    # 不同 mechanism_category 下最常见的 fragment
    rows = []
    for mech, sub_df in df.groupby("mechanism_category"):
        sub_top = (
            sub_df.sort_values(["support_count", "raw_descriptor"], ascending=[False, True])
            .head(TOP_N)
        )
        for _, row in sub_top.iterrows():
            item = {
                "mechanism_category": mech,
                "raw_descriptor": row["raw_descriptor"],
                "granularity": row["granularity"],
                "support_count": row["support_count"],
            }
            for dim in EFFECT_DIMS:
                item[dim] = row[dim]
            rows.append(item)

    pd.DataFrame(rows).to_csv(
        BASIC_STATS_DIR / f"top_{TOP_N}_fragments_per_mechanism.csv",
        index=False
    )

    # ------------------------------------------------------------
    # 6. 图：基础分布
    # ------------------------------------------------------------
    # mechanism_category 分布
    plt.figure(figsize=(10, 5))
    plt.bar(mech_dist["mechanism_category"], mech_dist["count"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Mechanism Category Distribution")
    plt.tight_layout()
    plt.savefig(BASIC_STATS_DIR / "mechanism_category_distribution.png", dpi=200)
    plt.close()

    # mechanism_category 加权分布
    plt.figure(figsize=(10, 5))
    plt.bar(mech_weighted["mechanism_category"], mech_weighted["weighted_support"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Weighted Support")
    plt.title("Mechanism Category Distribution (Weighted by support_count)")
    plt.tight_layout()
    plt.savefig(BASIC_STATS_DIR / "mechanism_category_distribution_weighted_by_support.png", dpi=200)
    plt.close()

    # granularity 分布
    plt.figure(figsize=(6, 4))
    plt.bar(gran_dist["granularity"], gran_dist["count"])
    plt.ylabel("Count")
    plt.title("Granularity Distribution")
    plt.tight_layout()
    plt.savefig(BASIC_STATS_DIR / "granularity_distribution.png", dpi=200)
    plt.close()

    # granularity 加权分布
    plt.figure(figsize=(6, 4))
    plt.bar(gran_weighted["granularity"], gran_weighted["weighted_support"])
    plt.ylabel("Weighted Support")
    plt.title("Granularity Distribution (Weighted by support_count)")
    plt.tight_layout()
    plt.savefig(BASIC_STATS_DIR / "granularity_distribution_weighted_by_support.png", dpi=200)
    plt.close()

    # singleton vs repeated
    plt.figure(figsize=(6, 4))
    plt.bar(["support=1", "support>1"], [singleton_count, repeated_count])
    plt.ylabel("Count")
    plt.title("Singleton vs Repeated Fragments")
    plt.tight_layout()
    plt.savefig(BASIC_STATS_DIR / "singleton_vs_repeated.png", dpi=200)
    plt.close()

    # 原始 support_count 分布（保留）
    plt.figure(figsize=(7, 4))
    plt.hist(df["support_count"], bins=50)
    plt.xlabel("support_count")
    plt.ylabel("Frequency")
    plt.title("Support Count Distribution")
    plt.tight_layout()
    plt.savefig(BASIC_STATS_DIR / "support_count_distribution.png", dpi=200)
    plt.close()

    # 只看 support_count > 1
    support_gt1 = df.loc[df["support_count"] > 1, "support_count"]
    if len(support_gt1) > 0:
        plt.figure(figsize=(7, 4))
        plt.hist(support_gt1, bins=50)
        plt.xlabel("support_count (>1)")
        plt.ylabel("Frequency")
        plt.title("Support Count Distribution (support_count > 1)")
        plt.tight_layout()
        plt.savefig(BASIC_STATS_DIR / "support_count_distribution_gt1.png", dpi=200)
        plt.close()

        upper = support_gt1.quantile(0.99)
        plt.figure(figsize=(7, 4))
        plt.hist(support_gt1[support_gt1 <= upper], bins=50)
        plt.xlabel("support_count (>1, clipped at 99th percentile)")
        plt.ylabel("Frequency")
        plt.title("Support Count Distribution (Clipped)")
        plt.tight_layout()
        plt.savefig(BASIC_STATS_DIR / "support_count_distribution_gt1_clipped.png", dpi=200)
        plt.close()

    # y 轴对数
    plt.figure(figsize=(7, 4))
    plt.hist(df["support_count"], bins=50, log=True)
    plt.xlabel("support_count")
    plt.ylabel("Frequency (log scale)")
    plt.title("Support Count Distribution (log-y)")
    plt.tight_layout()
    plt.savefig(BASIC_STATS_DIR / "support_count_distribution_logy.png", dpi=200)
    plt.close()

    # x/y 双对数
    support_pos = df.loc[df["support_count"] > 0, "support_count"]
    if len(support_pos) > 0:
        plt.figure(figsize=(7, 4))
        plt.hist(support_pos, bins=50)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("support_count (log scale)")
        plt.ylabel("Frequency (log scale)")
        plt.title("Support Count Distribution (log-log)")
        plt.tight_layout()
        plt.savefig(BASIC_STATS_DIR / "support_count_distribution_loglog.png", dpi=200)
        plt.close()

    # 累计覆盖曲线
    support_sorted = np.sort(df["support_count"].values)[::-1]
    cum_support = np.cumsum(support_sorted)
    cum_ratio = cum_support / cum_support[-1]
    x = np.arange(1, len(cum_ratio) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(x, cum_ratio)
    plt.xlabel("Top-k fragments")
    plt.ylabel("Cumulative support ratio")
    plt.title("Cumulative Coverage by Top-k Fragments")
    plt.tight_layout()
    plt.savefig(BASIC_STATS_DIR / "support_count_cumulative_coverage.png", dpi=200)
    plt.close()

    # effect_profile 8 维分布
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, dim in enumerate(EFFECT_DIMS):
        axes[i].hist(df[dim].dropna(), bins=30)
        axes[i].set_title(dim)
    fig.suptitle("Effect Profile Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(BASIC_STATS_DIR / "effect_profile_distributions.png", dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# 机制类别与风格效应关系
# ============================================================

def mechanism_effect_analysis(df: pd.DataFrame) -> None:
    group_mean = df.groupby("mechanism_category")[EFFECT_DIMS].mean()
    group_std = df.groupby("mechanism_category")[EFFECT_DIMS].std(ddof=0)

    group_mean.to_csv(MECH_EFFECT_DIR / "mechanism_effect_mean.csv")
    group_std.to_csv(MECH_EFFECT_DIR / "mechanism_effect_std.csv")

    mean_long = group_mean.reset_index().melt(
        id_vars="mechanism_category",
        value_vars=EFFECT_DIMS,
        var_name="effect_dimension",
        value_name="mean_value",
    )
    mean_long.to_csv(MECH_EFFECT_DIR / "mechanism_effect_mean_long.csv", index=False)

    std_long = group_std.reset_index().melt(
        id_vars="mechanism_category",
        value_vars=EFFECT_DIMS,
        var_name="effect_dimension",
        value_name="std_value",
    )
    std_long.to_csv(MECH_EFFECT_DIR / "mechanism_effect_std_long.csv", index=False)

    sep_rows = []
    categories = sorted(df["mechanism_category"].dropna().unique().tolist())

    for dim in EFFECT_DIMS:
        groups = []
        for cat in categories:
            vals = df.loc[df["mechanism_category"] == cat, dim].dropna().values
            if len(vals) > 0:
                groups.append(vals)

        eta2 = compute_eta_squared(groups)
        f_value = None
        p_value = None

        if SCIPY_AVAILABLE and len(groups) >= 2:
            try:
                f_res = f_oneway(*groups)
                f_value = float(f_res.statistic)
                p_value = float(f_res.pvalue)
            except Exception:
                pass

        sep_rows.append({
            "effect_dimension": dim,
            "num_groups": len(groups),
            "anova_f": f_value,
            "anova_p": p_value,
            "eta_squared": eta2,
        })

    sep_df = pd.DataFrame(sep_rows).sort_values(
        by=["eta_squared", "anova_f"],
        ascending=[False, False],
        na_position="last"
    )
    sep_df.to_csv(MECH_EFFECT_DIR / "mechanism_effect_separation_strength.csv", index=False)

    check_rows = []
    for mech, dim in MECHANISM_TO_EFFECT_CHECKS:
        target = df.loc[df["mechanism_category"] == mech, dim].dropna().values
        others = df.loc[df["mechanism_category"] != mech, dim].dropna().values

        cat_means = df.groupby("mechanism_category")[dim].mean().sort_values(ascending=False)
        rank = int(cat_means.index.tolist().index(mech) + 1) if mech in cat_means.index else None

        check_rows.append({
            "mechanism_category": mech,
            "effect_dimension": dim,
            "target_mean": float(np.mean(target)) if len(target) > 0 else None,
            "others_mean": float(np.mean(others)) if len(others) > 0 else None,
            "target_minus_others": float(np.mean(target) - np.mean(others)) if len(target) > 0 and len(others) > 0 else None,
            "cohens_d_vs_rest": compute_cohens_d(target, others),
            "rank_among_categories": rank,
            "num_categories": int(df["mechanism_category"].nunique()),
        })

    pd.DataFrame(check_rows).to_csv(MECH_EFFECT_DIR / "mechanism_internal_consistency_checks.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    mat = group_mean.loc[:, EFFECT_DIMS].values
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(EFFECT_DIMS)))
    ax.set_xticklabels(EFFECT_DIMS, rotation=45, ha="right")
    ax.set_yticks(range(len(group_mean.index)))
    ax.set_yticklabels(group_mean.index.tolist())
    ax.set_title("Mean Effect Profile by Mechanism Category")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(MECH_EFFECT_DIR / "mechanism_effect_mean_heatmap.png", dpi=200)
    plt.close()


# ============================================================
# 粒度与风格效应关系
# ============================================================

def granularity_effect_analysis(df: pd.DataFrame) -> None:
    group_mean = df.groupby("granularity")[EFFECT_DIMS].mean()
    group_std = df.groupby("granularity")[EFFECT_DIMS].std(ddof=0)

    group_mean.to_csv(GRAN_EFFECT_DIR / "granularity_effect_mean.csv")
    group_std.to_csv(GRAN_EFFECT_DIR / "granularity_effect_std.csv")

    cross = pd.crosstab(df["granularity"], df["mechanism_category"])
    cross.to_csv(GRAN_EFFECT_DIR / "granularity_mechanism_count_crosstab.csv")

    cross_row_ratio = cross.div(cross.sum(axis=1), axis=0)
    cross_row_ratio.to_csv(GRAN_EFFECT_DIR / "granularity_mechanism_row_ratio.csv")

    cross_col_ratio = cross.div(cross.sum(axis=0), axis=1)
    cross_col_ratio.to_csv(GRAN_EFFECT_DIR / "granularity_mechanism_col_ratio.csv")

    rows = []
    for gran in ["token", "long_phrase"]:
        sub = (
            df[df["granularity"] == gran]
            .sort_values(["support_count", "raw_descriptor"], ascending=[False, True])
            .head(TOP_N)
        )
        mech_counter = sub["mechanism_category"].value_counts()
        for mech, cnt in mech_counter.items():
            rows.append({
                "granularity": gran,
                "mechanism_category": mech,
                "count_in_topN": int(cnt),
                "ratio_in_topN": float(cnt / len(sub)) if len(sub) > 0 else 0.0,
            })
        sub.loc[:, ["raw_descriptor", "mechanism_category", "support_count"]].to_csv(
            GRAN_EFFECT_DIR / f"top_{TOP_N}_{gran}_fragments.csv", index=False
        )

    pd.DataFrame(rows).to_csv(GRAN_EFFECT_DIR / "high_freq_token_vs_long_phrase_mechanism_bias.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 4))
    mat = group_mean.loc[:, EFFECT_DIMS].values
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(EFFECT_DIMS)))
    ax.set_xticklabels(EFFECT_DIMS, rotation=45, ha="right")
    ax.set_yticks(range(len(group_mean.index)))
    ax.set_yticklabels(group_mean.index.tolist())
    ax.set_title("Mean Effect Profile by Granularity")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(GRAN_EFFECT_DIR / "granularity_effect_mean_heatmap.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 4))
    mat = cross_row_ratio.values
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(cross_row_ratio.columns)))
    ax.set_xticklabels(cross_row_ratio.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(range(len(cross_row_ratio.index)))
    ax.set_yticklabels(cross_row_ratio.index.tolist())
    ax.set_title("Mechanism Proportions within Each Granularity")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(GRAN_EFFECT_DIR / "granularity_mechanism_row_ratio_heatmap.png", dpi=200)
    plt.close()


# ============================================================
# 聚类与可视化
# ============================================================

def scatter_by_category(
    df_plot: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    out_path: Union[str, Path],
    title: str,
) -> None:
    labels = df_plot[label_col].fillna("NA").astype(str).unique().tolist()

    plt.figure(figsize=(8, 6))
    for lab in labels:
        sub = df_plot[df_plot[label_col].fillna("NA").astype(str) == lab]
        plt.scatter(sub[x_col], sub[y_col], s=8, alpha=0.7, label=lab)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend(markerscale=2, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def clustering_and_visualization(df: pd.DataFrame) -> None:
    work_df = df.dropna(subset=EFFECT_DIMS).copy()
    if len(work_df) == 0:
        raise ValueError("No valid rows with complete effect_profile for clustering.")

    X = work_df[EFFECT_DIMS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_model = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca_model.fit_transform(X_scaled)
    work_df["pca_1"] = X_pca[:, 0]
    work_df["pca_2"] = X_pca[:, 1]

    pca_info = {
        "explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
        "components": pca_model.components_.tolist(),
    }
    save_json(pca_info, CLUSTER_DIR / "pca_info.json")

    kmeans = KMeans(n_clusters=KMEANS_K, random_state=RANDOM_SEED, n_init=20)
    work_df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

    centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=EFFECT_DIMS)
    centers_df.insert(0, "cluster_id", range(len(centers_df)))
    centers_df.to_csv(CLUSTER_DIR / "kmeans_cluster_centers_scaled.csv", index=False)

    gmm = GaussianMixture(n_components=GMM_K, random_state=RANDOM_SEED)
    work_df["gmm_cluster"] = gmm.fit_predict(X_scaled)
    gmm_centers_df = pd.DataFrame(gmm.means_, columns=EFFECT_DIMS)
    gmm_centers_df.insert(0, "cluster_id", range(len(gmm_centers_df)))
    gmm_centers_df.to_csv(CLUSTER_DIR / "gmm_cluster_centers_scaled.csv", index=False)

    if HDBSCAN_AVAILABLE:
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
        )
        work_df["hdbscan_cluster"] = hdb.fit_predict(X_scaled)
    else:
        work_df["hdbscan_cluster"] = -999

    work_df.to_csv(CLUSTER_DIR / "fragment_analysis_with_clusters.csv", index=False)

    plot_df = sample_dataframe(work_df, MAX_POINTS_PCA)
    scatter_by_category(
        plot_df,
        "pca_1",
        "pca_2",
        "mechanism_category",
        CLUSTER_DIR / "pca_by_mechanism_category.png",
        "PCA Projection by Mechanism Category",
    )
    scatter_by_category(
        plot_df,
        "pca_1",
        "pca_2",
        "kmeans_cluster",
        CLUSTER_DIR / "pca_by_kmeans_cluster.png",
        "PCA Projection by KMeans Cluster",
    )

    if UMAP_AVAILABLE:
        umap_df = sample_dataframe(work_df, MAX_POINTS_UMAP)
        X_umap_input = scaler.fit_transform(umap_df[EFFECT_DIMS].values)
        umap_model = umap.UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            n_components=2,
            random_state=RANDOM_SEED,
        )
        X_umap = umap_model.fit_transform(X_umap_input)
        umap_df["umap_1"] = X_umap[:, 0]
        umap_df["umap_2"] = X_umap[:, 1]

        umap_df.to_csv(CLUSTER_DIR / "umap_projection_sample.csv", index=False)

        scatter_by_category(
            umap_df,
            "umap_1",
            "umap_2",
            "mechanism_category",
            CLUSTER_DIR / "umap_by_mechanism_category.png",
            "UMAP Projection by Mechanism Category",
        )
        scatter_by_category(
            umap_df,
            "umap_1",
            "umap_2",
            "kmeans_cluster",
            CLUSTER_DIR / "umap_by_kmeans_cluster.png",
            "UMAP Projection by KMeans Cluster",
        )

    tsne_df = sample_dataframe(work_df, MAX_POINTS_TSNE)
    X_tsne_input = scaler.fit_transform(tsne_df[EFFECT_DIMS].values)
    tsne_model = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        n_iter=TSNE_N_ITER,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
    )
    X_tsne = tsne_model.fit_transform(X_tsne_input)
    tsne_df["tsne_1"] = X_tsne[:, 0]
    tsne_df["tsne_2"] = X_tsne[:, 1]
    tsne_df.to_csv(CLUSTER_DIR / "tsne_projection_sample.csv", index=False)

    scatter_by_category(
        tsne_df,
        "tsne_1",
        "tsne_2",
        "mechanism_category",
        CLUSTER_DIR / "tsne_by_mechanism_category.png",
        "t-SNE Projection by Mechanism Category",
    )
    scatter_by_category(
        tsne_df,
        "tsne_1",
        "tsne_2",
        "kmeans_cluster",
        CLUSTER_DIR / "tsne_by_kmeans_cluster.png",
        "t-SNE Projection by KMeans Cluster",
    )

    cluster_stats = {
        "num_rows_used": int(len(work_df)),
        "kmeans_cluster_counts": work_df["kmeans_cluster"].value_counts().sort_index().to_dict(),
        "gmm_cluster_counts": work_df["gmm_cluster"].value_counts().sort_index().to_dict(),
        "hdbscan_cluster_counts": work_df["hdbscan_cluster"].value_counts().sort_index().to_dict(),
        "umap_available": UMAP_AVAILABLE,
        "hdbscan_available": HDBSCAN_AVAILABLE,
    }
    save_json(cluster_stats, CLUSTER_DIR / "cluster_stats.json")


# ============================================================
# 总控
# ============================================================

def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 80)
    print("Loading records...")
    records = load_jsonl_records(INPUT_JSONL_PATH)
    print(f"Loaded {len(records)} records.")

    print("=" * 80)
    print("Building flat dataframe...")
    df = build_flat_dataframe(records)
    df.to_csv(BASE_OUTPUT_DIR / "fragment_analysis_flattened.csv", index=False)
    print(f"Flattened dataframe saved: {BASE_OUTPUT_DIR / 'fragment_analysis_flattened.csv'}")

    print("=" * 80)
    print("Running basic statistics...")
    basic_statistics(df)

    print("=" * 80)
    print("Running mechanism-effect analysis...")
    mechanism_effect_analysis(df)

    print("=" * 80)
    print("Running granularity-effect analysis...")
    granularity_effect_analysis(df)

    print("=" * 80)
    print("Running clustering and visualization...")
    clustering_and_visualization(df)

    print("=" * 80)
    print("Running text-embedding clustering and visualization...")
    clustering_and_visualization_text_embedding(df)

    summary = {
        "input_path": INPUT_JSONL_PATH,
        "base_output_dir": str(BASE_OUTPUT_DIR),
        "num_records": int(len(df)),
        "num_mechanism_categories": int(df["mechanism_category"].nunique(dropna=True)),
        "num_granularity_types": int(df["granularity"].nunique(dropna=True)),
        "effect_dimensions": EFFECT_DIMS,
        "top_n": TOP_N,
        "random_seed": RANDOM_SEED,
        "scipy_available": SCIPY_AVAILABLE,
        "umap_available": UMAP_AVAILABLE,
        "hdbscan_available": HDBSCAN_AVAILABLE,
        "text_embedding_model_name": EMBEDDING_MODEL_NAME,
        "subdirs": {
            "basic_statistics": str(BASIC_STATS_DIR),
            "mechanism_effect_analysis": str(MECH_EFFECT_DIR),
            "granularity_effect_analysis": str(GRAN_EFFECT_DIR),
            "clustering_visualization": str(CLUSTER_DIR),
            "text_embedding_visualization": str(TEXT_EMBED_DIR),
        },
        "text_embedding_outputs": {
            "dir": str(TEXT_EMBED_DIR),
            "embedding_model_name": EMBEDDING_MODEL_NAME,
            "unsupervised_metrics": [
                "text_embedding_kmeans_vs_mechanism_metrics.json",
                "text_embedding_hdbscan_vs_mechanism_metrics.json",
            ],
            "supervised_metrics": [
                "text_embedding_supervised_classification_metrics.json",
                "text_embedding_supervised_classification_report.csv",
                "text_embedding_supervised_confusion_matrix.csv",
                "text_embedding_supervised_confusion_matrix.png",
            ],
        }
    }
    save_json(summary, BASE_OUTPUT_DIR / "analysis_summary.json")

    print("=" * 80)
    print("All analyses finished.")
    print(f"Results saved under: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()