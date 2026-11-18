from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import stanza
import spacy
import re
from openai import OpenAI
import os
import json
# from difflib import SequenceMatcher
from pathlib import Path
from tools_paths import TOOLS_ROOT
from PIL import Image
from stanza.pipeline.core import DownloadMethod
from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    BadRequestError,
)
import time
import random
# api调用重试
def chat_completion_with_retry(
    client,
    *,
    model: str,
    messages,
    response_format=None,
    temperature: float = 0.0,
    max_retries: int = 8,
    base_sleep: float = 2.0,
):
    """
    对 chat.completions.create 做自动重试。
    适合处理 502 / 503 / 504 / 网络波动 / 限流 等临时错误。
    """
    last_err = None

    for attempt in range(max_retries + 1):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format

            return client.chat.completions.create(**kwargs)
        except BadRequestError:
            # 400 这类通常是请求本身有问题，不该重试
            raise
        except (InternalServerError, APIConnectionError, APITimeoutError, RateLimitError) as e:
            last_err = e
            if attempt >= max_retries:
                raise
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.8)
            print(f"[LLM RETRY] attempt={attempt + 1}/{max_retries}, sleep={sleep_s:.1f}s, error={repr(e)}")
            time.sleep(sleep_s)
    raise last_err
def tokenize_prompts(prompts: List[str]) -> List[List[str]]:
    """
    将提示词列表切分为 token 列表。
    规则：
    1. 使用 spaCy 英文 tokenizer 做基础分词
    2. 忽略纯分隔符/纯标点（如 , | ; / :: () []）
    3. 将形如 word-word 的连字符表达合并为一个 token，如 close-up、anime-style
    4. 输出结果为：每个 prompt 对应一个 token 子列表
    参数：
        prompts: 提示词字符串列表
    返回：
        List[List[str]]
    """
    nlp = spacy.blank("en")
    # 这些符号只作为切分边界，不计入 token
    skip_tokens = {
        ",", "|", ";", "/", "::", "(", ")", "[", "]", "{", "}",
    }
    all_tokens: List[List[str]] = []
    for prompt in prompts:
        doc = nlp(prompt)
        # 先拿原始 token 文本
        raw = [t.text.strip() for t in doc if t.text.strip()]
        merged: List[str] = []
        i = 0
        while i < len(raw):
            tok = raw[i]
            # 跳过纯分隔符
            if tok in skip_tokens:
                i += 1
                continue
            # 合并 word - word 形式为一个 token
            if (
                i + 2 < len(raw)
                and raw[i + 1] == "-"
                and raw[i] not in skip_tokens
                and raw[i + 2] not in skip_tokens
            ):
                merged.append(f"{raw[i]}-{raw[i + 2]}")
                i += 3
                continue
            # 跳过单独的连字符
            if tok == "-":
                i += 1
                continue
            # 跳过其他纯标点
            if all(not ch.isalnum() for ch in tok):
                i += 1
                continue
            merged.append(tok)
            i += 1
        all_tokens.append(merged)
    return all_tokens

# fragment分割
# 第一步：只用强分隔符做粗切
STRONG_SEPARATOR_PATTERN = re.compile(
    r"""
    \s*
    (?:
        \|\| | \| | :: | ; | ； | 、 | \n
        | (?<!\d)[.:] | [.:](?!\d)
        | [!?]
        | [。！？]
    )+
    \s*
    """,
    re.VERBOSE
)
# 2) 逗号切分：后续启发式处理
COMMA_SPLIT_PATTERN = re.compile(r"\s*[，,]\s*")
def normalize_prompt_for_coarse_split(prompt: str) -> str:
    if prompt is None:
        return ""
    prompt = prompt.strip()
    prompt = re.sub(r"[ \t\r\f\v]+", " ", prompt)
    return prompt
def token_len(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", text))
def should_split_by_comma(chunk: str) -> bool:
    """
    只在明显像“逗号分隔的标签式 prompt”时才按逗号继续切。
    """
    parts = [p.strip() for p in COMMA_SPLIT_PATTERN.split(chunk) if p.strip()]
    if len(parts) < 2:
        return False
    lens = [token_len(p) for p in parts]
    # 情况1：全部都很短，像标签串
    if all(x <= 5 for x in lens):
        return True
    # 情况2：第一段是内容短语，后面基本都是短风格标签
    # 例如: "a woman lying on the sofa, oil painting, happy, smooth, fine-detailed"
    if lens[0] <= 12 and len(parts) >= 3 and all(x <= 4 for x in lens[1:]):
        return True
    # 其余情况不按逗号切
    return False
def split_one_chunk_by_comma_if_needed(chunk: str) -> List[str]:
    chunk = chunk.strip()
    if not chunk:
        return []
    if should_split_by_comma(chunk):
        return [p.strip() for p in COMMA_SPLIT_PATTERN.split(chunk) if p.strip()]
    return [chunk]
def coarse_split_prompts(prompts: List[str]) -> List[List[str]]:
    results: List[List[str]] = []
    for prompt in prompts:
        text = normalize_prompt_for_coarse_split(prompt)
        if not text:
            results.append([])
            continue
        # 先按强分隔符切
        first_level_chunks = [c.strip() for c in STRONG_SEPARATOR_PATTERN.split(text) if c.strip()]
        # 再对每个 chunk 做“有条件的逗号切分”
        final_chunks = []
        for chunk in first_level_chunks:
            final_chunks.extend(split_one_chunk_by_comma_if_needed(chunk))
        results.append(final_chunks)
    return results

# 第二步：生成候选短语分割
PHRASE_LABELS = {
    "NP",    # 名词短语
    "VP",    # 动词短语（有时 prompt 里会出现）
    "PP",    # 介词短语
    "ADJP",  # 形容词短语
    "ADVP",  # 副词短语
    "NML",   # 名词修饰短语
    "NX"     # 某些 parser 里的名词扩展结构
}
def build_constituency_parser(use_gpu=True) -> stanza.Pipeline:
    """
    构建 Stanza constituency parser.
    tokenize_no_ssplit=True 保证每个 chunk 被当作一个整体处理。
    """
    return stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,constituency",
        tokenize_no_ssplit=True,
        use_gpu=use_gpu,
        model_dir=str(TOOLS_ROOT / 'stanza_models'),
        download_method=DownloadMethod.REUSE_RESOURCES,
    )
def normalize_chunk_for_parser(chunk: str) -> str:
    """
    轻量归一化，避免 parser 被多余空白干扰。
    """
    chunk = chunk.strip()
    chunk = re.sub(r"\s+", " ", chunk)
    return chunk
def _safe_leaf_text(tree) -> str:
    """
    从 constituency 子树中恢复文本。
    """
    try:
        leaves = tree.leaf_labels()
        return " ".join(leaves).strip()
    except Exception:
        return ""
def _collect_constituency_candidates(tree) -> List[str]:
    """
    从 constituency tree 里收集候选短语。
    """
    candidates = []
    def dfs(node):
        if node is None:
            return
        # 先尝试从当前节点取 label
        label = getattr(node, "label", None)
        # 如果是关心的短语类型，就收集
        if label in PHRASE_LABELS:
            text = _safe_leaf_text(node)
            if text:
                candidates.append(text)
        # 递归 children
        children = getattr(node, "children", None)
        if children:
            for child in children:
                dfs(child)
    dfs(tree)
    return candidates
def _collect_pos_fallback_candidates(sentence) -> List[str]:
    """
    constituency parser 对 prompt fragment 不总是稳定，
    所以加一层基于 POS 的候选兜底。
    """
    BAD_SINGLETONS = {
        "a", "an", "the", "by", "of", "in", "on", "to", "for", "with",
        "and", "or", "-", "’s", "'s"
    }
    candidates = []
    words = [w.text for w in sentence.words]
    upos = [w.upos for w in sentence.words]
    n = len(words)
    # 单词候选（更严格：过滤明显无意义残片）
    for w, p in zip(words, upos):
        wl = w.strip().lower()
        if not wl:
            continue
        if wl in BAD_SINGLETONS:
            continue
        if p in {"DET", "ADP", "PART", "PUNCT", "AUX", "CCONJ", "SCONJ"}:
            continue
        candidates.append(w)
    # 常见短语模式
    patterns = [
        ["ADJ", "NOUN"],
        ["ADJ", "ADJ", "NOUN"],
        ["NOUN", "NOUN"],
        ["PROPN", "PROPN"],
        ["ADJ", "NOUN", "NOUN"],
        ["ADV", "ADJ"],
        ["ADV", "ADJ", "NOUN"],
    ]
    for i in range(n):
        for pat in patterns:
            L = len(pat)
            if i + L <= n and upos[i:i+L] == pat:
                phrase = " ".join(words[i:i+L]).strip()
                if phrase:
                    candidates.append(phrase)
    # 整段候选
    if words:
        candidates.append(" ".join(words))

    return candidates
def _deduplicate_keep_order(items: List[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            output.append(item.strip())
    return output
def parser_candidates_for_chunk(chunk: str, parser: stanza.Pipeline) -> Dict[str, Any]:
    """
    对单个 chunk 生成 parser 候选短语。
    返回:
    {
        "chunk": 原始chunk,
        "candidates": [...候选短语...]
    }
    """
    chunk = normalize_chunk_for_parser(chunk)
    if not chunk:
        return {"chunk": chunk, "candidates": []}
    try:
        doc = parser(chunk)
    except Exception:
        # parser 出错时退化为整段保留
        return {"chunk": chunk, "candidates": [chunk]}
    if not doc.sentences:
        return {"chunk": chunk, "candidates": [chunk]}
    sent = doc.sentences[0]
    candidates = []
    # 1) constituency tree 候选
    tree = getattr(sent, "constituency", None)
    if tree is not None:
        candidates.extend(_collect_constituency_candidates(tree))
    # 2) POS fallback 候选
    candidates.extend(_collect_pos_fallback_candidates(sent))
    # 3) 强制保留整段
    candidates.append(chunk)
    # 去重
    candidates = _deduplicate_keep_order(candidates)
    return {
        "chunk": chunk,
        "candidates": candidates
    }
def parser_candidates_for_prompts(
    coarse_chunks_2d: List[List[str]],
    parser: stanza.Pipeline) -> List[List[Dict[str, Any]]]:
    """
    输入:
        第一步粗切后的二维列表
    输出:
        二维列表，每个 chunk 对应一个 dict:
        {
            "chunk": ...,
            "candidates": [...]
        }
    """
    results: List[List[Dict[str, Any]]] = []
    for chunk_list in coarse_chunks_2d:
        prompt_result = []
        for chunk in chunk_list:
            prompt_result.append(parser_candidates_for_chunk(chunk, parser))
        results.append(prompt_result)
    return results

def build_online_llm_client() -> OpenAI:
    """Create the annotation client using environment configuration."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY before running LLM annotation steps.")
    kwargs = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def build_fragment_judge_messages(chunk: str, candidates: List[str]) -> List[Dict[str, str]]:
    """为单个 chunk 构造 LLM 消息。"""
    # system_prompt = """
    # You are a prompt fragment extraction assistant.
    #
    # Task:
    # Extract the smallest independently annotatable core semantic fragments from ONE prompt chunk.
    #
    # Goal:
    # Keep only the semantic units that are useful for downstream annotation.
    # You may discard non-essential framing or function-like expressions.
    #
    # Rules:
    # 1. Extract the smallest independently annotatable semantic fragments.
    # 2. Keep content-bearing units such as entities, objects, attributes, moods, references, mediums, composition cues, and scene descriptors.
    # 3. You may discard non-essential framing expressions, function-like wrappers, and instruction scaffolding if they do not carry independent annotation value.
    # 4. The returned fragments do NOT need to cover every word in the original chunk.
    # 5. Do not output incomplete fragments such as:
    #    "A narrow", "inspired by", "with a heavy emphasis on", "to create a sense of".
    # 6. Proper names, fixed expressions, medium phrases, composition phrases, and content noun phrases should usually stay intact.
    # 7. Do not output overlapping or nested fragments.
    # 8. Do not rewrite, paraphrase, or invent text.
    # 9. Every returned fragment must be a continuous substring of the original chunk.
    #
    # Output JSON only:
    # {
    #   "final_fragments": ["fragment1", "fragment2", "..."]
    # }
    # """.strip()
    system_prompt = """
    You are a prompt fragment extraction assistant.

    Task:
    Extract independently annotatable semantic fragments from ONE prompt chunk.

    Goal:
    Preserve content-related information as completely as possible, while allowing style-related information to be reduced to its core annotatable phrase.

    Definitions:
    - Content-related information includes entities, objects, scenes, actions, attributes, spatial relations, and other scene-defining descriptions.
    - Style-related information includes artistic style, medium, aesthetic flavor, mood cues, composition cues, references, cinematography, color grading, rendering style, and similar stylistic modifiers.
    - Function-like or framing expressions are wrappers that do not carry independent annotation value.

    Rules:
    1. For content-related fragments, preserve them as complete semantic units whenever possible.
    2. For style-related fragments, keep only the smallest core phrase that still preserves annotatable meaning.
    3. You may discard non-essential framing or function-like expressions if they do not carry independent annotation value.
    4. The returned fragments do NOT need to cover every word in the original chunk.
    5. Do not output incomplete fragments such as:
       "A narrow", "inspired by", "with a heavy emphasis on", "to create a sense of".
    6. Proper names, fixed expressions, medium phrases, composition phrases, reference phrases, and content noun phrases should usually stay intact.
    7. Do not output overlapping or nested fragments.
    8. Do not rewrite, paraphrase, normalize, or invent text.
    9. Every returned fragment must be a continuous substring of the original chunk.

    Preference:
    - Prefer semantic completeness for content.
    - Prefer compact core phrases for style.
    - Prefer dropping framing language over keeping redundant wrappers.

    Examples:
    Chunk: "a sense of urban claustrophobia and melancholic isolation"
    Output:
    {"final_fragments": ["urban claustrophobia", "melancholic isolation"]}

    Chunk: "A cinematic aesthetic inspired by Wong Kar-wai’s cinematography"
    Output:
    {"final_fragments": ["cinematic aesthetic", "Wong Kar-wai’s cinematography"]}

    Chunk: "A narrow subterranean alleyway in a vertical megacity"
    Output:
    {"final_fragments": ["A narrow subterranean alleyway", "in a vertical megacity"]}

    Output JSON only:
    {
      "final_fragments": ["fragment1", "fragment2", "..."]
    }
    """.strip()
    user_payload = {
        "chunk": chunk,
        "candidate_phrases": candidates
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]
def validate_final_fragments(chunk: str, fragments: List[str]) -> List[str]:
    """
    对模型输出做校验与去冗余：
    - 去空
    - 保证每个 fragment 是原 chunk 的连续子串
    - 去重但保序
    - 去掉被更长 fragment 完整包含的嵌套片段
    """
    cleaned: List[str] = []
    seen = set()
    # 1) 基础清洗 + 精确去重
    for frag in fragments:
        if not isinstance(frag, str):
            return [chunk]
        frag = frag.strip()
        if not frag:
            continue
        if frag not in chunk:
            return [chunk]
        key = frag.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(frag)
    if not cleaned:
        return [chunk]
    # 2) 去掉被更长片段完整包含的嵌套片段
    # 规则：如果某个 fragment 完整出现在另一个更长 fragment 中，则丢掉较短者
    pruned: List[str] = []
    for i, frag in enumerate(cleaned):
        frag_norm = frag.strip().lower()
        is_nested = False
        for j, other in enumerate(cleaned):
            if i == j:
                continue
            other_norm = other.strip().lower()
            if frag_norm == other_norm:
                continue
            if frag_norm in other_norm and len(frag_norm) < len(other_norm):
                is_nested = True
                break
        if not is_nested:
            pruned.append(frag)
    if not pruned:
        return [chunk]
    return pruned
def llm_finalize_chunk(
    chunk: str,
    candidates: List[str],
    client: OpenAI,
    model: str = "gpt-4o",) -> Dict[str, Any]:
    """
    对单个 chunk 做最终 fragment 裁决。
    """
    messages = build_fragment_judge_messages(chunk, candidates)

    response = chat_completion_with_retry(
        client,
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "fragment_segmentation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "final_fragments": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["final_fragments"],
                    "additionalProperties": False
                }
            }
        }
    )
    content = response.choices[0].message.content
    data = json.loads(content)
    final_fragments = validate_final_fragments(
        chunk=chunk,
        fragments=data.get("final_fragments", [chunk])
    )
    return {
        "chunk": chunk,
        "candidates": candidates,
        "final_fragments": final_fragments
    }
def llm_finalize_prompts(
    parser_results_2d: List[List[Dict[str, Any]]],
    client: OpenAI,
    model: str = "gpt-4o",) -> List[List[Dict[str, Any]]]:
    """
    输入:
        parser_candidates_for_prompts 的输出，格式形如：
        [
            [
                {"chunk": "...", "candidates": ["...", "..."]},
                {"chunk": "...", "candidates": ["...", "..."]}
            ],
            ...
        ]

    输出:
        [
            [
                {
                    "chunk": "...",
                    "candidates": [...],
                    "final_fragments": [...]
                },
                ...
            ],
            ...
        ]
    """
    results: List[List[Dict[str, Any]]] = []
    for prompt_chunks in parser_results_2d:
        prompt_result: List[Dict[str, Any]] = []
        for item in prompt_chunks:
            chunk = item["chunk"]
            candidates = item["candidates"]
            judged = llm_finalize_chunk(
                chunk=chunk,
                candidates=candidates,
                client=client,
                model=model
            )
            prompt_result.append(judged)
        results.append(prompt_result)
    return results
def flatten_final_fragments(final_results_2d: List[List[Dict[str, Any]]]) -> List[List[str]]:
    """
    把最终结果压平成每条 prompt 的 fragment 列表。
    """
    outputs: List[List[str]] = []
    for prompt_chunks in final_results_2d:
        frags: List[str] = []
        for item in prompt_chunks:
            frags.extend(item["final_fragments"])
        outputs.append(frags)
    return outputs

# 计算片段提取前后相似度
def merge_fragments_by_prompt(
    fragment_lists_2d: List[List[str]],
    sep: str = " "
) -> List[str]:
    """
    将二维 fragment 列表按 prompt 合并成一维字符串列表。
    不做额外清洗，只做简单拼接。
    """
    merged_prompts = []
    for frag_list in fragment_lists_2d:
        merged_prompts.append(sep.join(frag_list))
    return merged_prompts
def compute_semantic_similarity_list(
    original_prompts: List[str],
    fragment_lists_2d: List[List[str]],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    sep: str = " "
) -> List[float]:
    from sentence_transformers import SentenceTransformer, util
    """
    输入：
        original_prompts: 原始 prompt 列表
        fragment_lists_2d: LLM 返回的二维 fragment 列表
    输出：
        与原始 prompt 一一对应的语义相似度列表
    """
    if len(original_prompts) != len(fragment_lists_2d):
        raise ValueError("original_prompts 和 fragment_lists_2d 长度必须一致")
    merged_prompts = merge_fragments_by_prompt(fragment_lists_2d, sep=sep)
    model = SentenceTransformer(model_name)
    emb1 = model.encode(original_prompts, convert_to_tensor=True, normalize_embeddings=True)
    emb2 = model.encode(merged_prompts, convert_to_tensor=True, normalize_embeddings=True)
    sim_matrix = util.cos_sim(emb1, emb2)
    scores = [float(sim_matrix[i, i]) for i in range(len(original_prompts))]
    return scores

# 区分content和style两类fragment
def build_content_style_classify_messages(fragments: List[str]) -> List[Dict[str, str]]:
    """
    输入一个 prompt 的 fragment 列表，让 LLM 将其分成 content / style 两类。
    """
    system_prompt = """
    You are a fragment classification assistant.
    
    Task:
    Given a list of already segmented prompt fragments, classify each fragment into exactly one of two categories:
    - content
    - style
    
    Definitions:
    - Content fragments describe entities, objects, scenes, actions, attributes, spatial relations, events, or scene-defining descriptions.
    - Style fragments describe artistic style, medium, aesthetic flavor, mood cues, composition cues, color grading, cinematography, rendering style, references, or other presentational modifiers.
    
    Rules:
    1. Every fragment must be assigned to exactly one category: content or style.
    2. Do not rewrite, paraphrase, split, merge, or invent fragments.
    3. Return each fragment exactly as provided.
    4. If a fragment is ambiguous, choose the dominant role:
       - choose content if it mainly defines what is depicted
       - choose style if it mainly defines how it is depicted
    
    Output JSON only:
    {
      "content_fragments": ["..."],
      "style_fragments": ["..."]
    }
    """.strip()
    user_payload = {
        "fragments": fragments
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]
def validate_content_style_split(
    input_fragments: List[str],
    content_fragments: List[str],
    style_fragments: List[str]
) -> Dict[str, List[str]]:
    """
    校验并修正 LLM 输出：
    - 每个原始 fragment 必须且只能出现一次
    - 不能改写
    - 若遗漏，则默认补到 content（更保守）
    """
    original = [x.strip() for x in input_fragments if isinstance(x, str) and x.strip()]
    original_set = set(original)
    content = [x.strip() for x in content_fragments if isinstance(x, str) and x.strip() in original_set]
    style = [x.strip() for x in style_fragments if isinstance(x, str) and x.strip() in original_set]
    # 去重但保序
    def dedup_keep_order(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    content = dedup_keep_order(content)
    style = dedup_keep_order(style)
    used = set(content) | set(style)
    # 漏掉的 fragment 默认补到 content
    # 原因：内容信息通常更关键，保守起见不丢
    missing = [x for x in original if x not in used]
    content.extend(missing)
    # 如果同时出现在两边，优先保留 content，style 删除
    content_set = set(content)
    style = [x for x in style if x not in content_set]
    return {
        "content_fragments": content,
        "style_fragments": style
    }
def llm_classify_fragments_for_one_prompt(
    fragments: List[str],
    client: OpenAI,
    model: str = "gpt-4o"
) -> Dict[str, List[str]]:
    """
    对单个 prompt 的 fragment 列表做 content/style 二分类。
    """
    messages = build_content_style_classify_messages(fragments)
    response = chat_completion_with_retry(
        client,
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "fragment_content_style_split",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "content_fragments": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "style_fragments": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["content_fragments", "style_fragments"],
                    "additionalProperties": False
                }
            }
        }
    )
    content = json.loads(response.choices[0].message.content)
    checked = validate_content_style_split(
        input_fragments=fragments,
        content_fragments=content.get("content_fragments", []),
        style_fragments=content.get("style_fragments", [])
    )
    return checked
def llm_split_content_and_style_fragments(
    fragments_2d: List[List[str]],
    client: OpenAI,
    model: str = "gpt-4o"
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    输入：
        fragments_2d = [
            ["fragment1", "fragment2", ...],   # 第1条 prompt 的 fragments
            ["fragment1", "fragment2", ...],   # 第2条 prompt 的 fragments
            ...
        ]

    输出：
        (
            content_fragments_2d,
            style_fragments_2d
        )

    其中：
        content_fragments_2d[i] 对应第 i 条 prompt 的内容相关 fragments
        style_fragments_2d[i] 对应第 i 条 prompt 的风格相关 fragments
    """
    content_fragments_2d: List[List[str]] = []
    style_fragments_2d: List[List[str]] = []
    for fragments in fragments_2d:
        result = llm_classify_fragments_for_one_prompt(
            fragments=fragments,
            client=client,
            model=model
        )
        content_fragments_2d.append(result["content_fragments"])
        style_fragments_2d.append(result["style_fragments"])
    return content_fragments_2d, style_fragments_2d

# 利用LLM进行mechanism_category分类
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
def build_mechanism_category_messages(fragments: List[str]) -> List[Dict[str, str]]:
    """
    让 LLM 对一个 prompt 下的多个 fragments 做 mechanism_category 单选分类。
    输入: List[str]
    输出: 与输入等长的类别列表
    """
    system_prompt = """
    You are a fragment classification assistant.
    
    Task:
    Classify each fragment into exactly one mechanism_category.
    
    Available mechanism_category labels:
    1. reference_anchor
       - references to specific artists, creators, iconic works, established style schools, or recognizable artistic anchors
       - examples: "Wong Kar-wai’s cinematography", "in the style of Klimt"
    
    2. rendering_medium
       - medium, material, drawing/painting/rendering technique, or production form
       - examples: "oil painting", "watercolor", "charcoal sketch", "line art"
    
    3. representational_regime
       - overall representational mode, such as realistic, abstract, stylized, painterly, sketch-like, photorealistic
       - examples: "photorealistic", "abstract", "stylized"
    
    4. affective_atmosphere
       - mood, emotional tone, dramatic atmosphere, psychological feeling
       - examples: "melancholic isolation", "eerie", "dreamy", "urban claustrophobia"
    
    5. spatial_composition
       - framing, viewpoint, camera/lens language, shot type, spatial arrangement, composition
       - examples: "wide-angle anamorphic lens perspective", "low-angle shot", "centered composition"
    
    6. perceptual_cue
       - perceptual surface cues such as color, contrast, sharpness, texture, detail intensity, lighting salience
       - examples: "high-contrast color grading", "soft lighting", "high detail", "neon pink and teal"
    
    7. contextual_frame
       - presentation wrapper or genre frame such as poster, concept art, key visual, cover art, promo image
       - examples: "concept art", "key visual", "poster design", "album cover"
    
    8. platform_context
       - platform/community/tag-context expressions
       - examples: "trending on artstation", "pixiv style", "featured on behance"
    
    Rules:
    1. Assign exactly one label to each fragment.
    2. Do not rewrite, paraphrase, split, merge, or invent fragments.
    3. If ambiguous, choose the dominant mechanism.
    4. Return labels in the same order as the input fragments.
    5. Output JSON only.
    
    Output format:
    {
      "mechanism_categories": ["label1", "label2", "..."]
    }
    """.strip()
    user_payload = {
        "fragments": fragments
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]
def validate_mechanism_categories(
    fragments: List[str],
    categories: List[str],
) -> List[str]:
    """
    校验 LLM 输出：
    - 长度必须与输入 fragments 一致
    - 每个类别必须在允许集合内
    - 若不合法则抛错
    """
    if not isinstance(categories, list):
        raise ValueError("mechanism_categories 必须是列表")
    if len(categories) != len(fragments):
        raise ValueError(
            f"mechanism_categories 长度与 fragments 不一致: "
            f"{len(categories)} vs {len(fragments)}"
        )
    normalized = []
    for c in categories:
        if not isinstance(c, str):
            raise ValueError("mechanism_category 必须是字符串")
        c = c.strip()
        if c not in MECHANISM_CATEGORIES:
            raise ValueError(f"非法 mechanism_category: {c}")
        normalized.append(c)
    return normalized
def llm_classify_mechanism_categories_one_prompt(
    fragments: List[str],
    client: OpenAI,
    model: str = "gpt-4o",
) -> List[str]:
    """
    输入:
        一个 prompt 的 fragments 列表
    输出:
        与输入等长的 mechanism_category 列表

    示例:
        ["oil painting", "Wong Kar-wai’s cinematography", "melancholic isolation"]
        ->
        ["rendering_medium", "reference_anchor", "affective_atmosphere"]
    """
    messages = build_mechanism_category_messages(fragments)
    response = chat_completion_with_retry(
        client,
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "fragment_mechanism_category_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "mechanism_categories": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": sorted(list(MECHANISM_CATEGORIES))
                            }
                        }
                    },
                    "required": ["mechanism_categories"],
                    "additionalProperties": False
                }
            }
        }
    )
    data = json.loads(response.choices[0].message.content)
    categories = data["mechanism_categories"]
    categories = validate_mechanism_categories(fragments, categories)
    return categories
def llm_classify_mechanism_categories_2d(
    fragments_2d: List[List[str]],
    client: OpenAI,
    model: str = "gpt-4o",
) -> List[List[str]]:
    """
    输入:
        二维列表，每个子列表是一条 prompt 切分后的 fragments

    输出:
        与输入结构一致的二维列表，只不过内容从 fragment 变为 mechanism_category

    示例:
        [
            ["oil painting", "Wong Kar-wai’s cinematography"],
            ["urban claustrophobia", "wide-angle anamorphic lens perspective"]
        ]
        ->
        [
            ["rendering_medium", "reference_anchor"],
            ["affective_atmosphere", "spatial_composition"]
        ]
    """
    results: List[List[str]] = []
    for fragments in fragments_2d:
        categories = llm_classify_mechanism_categories_one_prompt(
            fragments=fragments,
            client=client,
            model=model
        )
        results.append(categories)
    return results

# 对风格fragment进行effect_profile打分
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
def build_effect_profile_messages(fragments: List[str]) -> List[Dict[str, str]]:
    """
    输入一个 prompt 下的 style fragments，
    让 LLM 为每个 fragment 输出一组 effect_profile 分数。
    """
    system_prompt = """
    You are a fragment scoring assistant.

    Task:
    For each input style fragment, assign a score in [0,1] for each effect_profile dimension.

    Dimensions:
    - artistic_context: strength of artistic tradition, style reference, or art-historical context
    - realism: how strongly the fragment pushes depiction toward realistic or natural rendering
    - stylization: how strongly the fragment pushes depiction toward artistic, stylized, or non-natural rendering
    - atmospheric_intensity: how strongly the fragment increases mood, atmosphere, or dramatic tension
    - sensory_salience: how strongly the fragment enhances color, texture, detail, light, contrast, or other perceptual vividness
    - platform_association: how strongly the fragment is associated with platform/community/tag culture
    - depiction_explicitness: how strongly the fragment makes depiction more direct or explicit, especially for sexual or nudity-related content
    - violence_graphicness: how strongly the fragment makes violent, bloody, gory, injury-related, or harmful content more graphic, direct, or visually intense

    Rules:
    1. Score each dimension on an absolute scale, not merely relative to the other fragments in the batch.
    2. Use the full [0,1] range when justified, but avoid extreme values unless the evidence is very strong.
    3. Keep clear distinctions across fragments in the same batch; do not collapse everything into the middle.
    4. platform_association should stay low unless the fragment clearly refers to a platform, community, or tag culture.
    5. depiction_explicitness should stay low for most ordinary style fragments unless the fragment clearly increases directness or explicitness, especially for sexual or nudity-related depiction.
    6. violence_graphicness should stay low for most ordinary style fragments unless the fragment clearly increases graphic violence, gore, bloodiness, injury visibility, or harm intensity.
    7. Return scores in the same order as the input fragments.
    8. Do not rewrite, paraphrase, split, or merge fragments.
    9. Return JSON only.

    Output format:
    {
      "effect_profiles": [
        {
          "artistic_context": 0.0,
          "realism": 0.0,
          "stylization": 0.0,
          "atmospheric_intensity": 0.0,
          "sensory_salience": 0.0,
          "platform_association": 0.0,
          "depiction_explicitness": 0.0,
          "violence_graphicness": 0.0
        }
      ]
    }
    """.strip()
    user_payload = {
        "fragments": fragments
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]
def validate_effect_profiles(
    fragments: List[str],
    effect_profiles: List[Dict[str, Any]],
) -> List[Dict[str, float]]:
    """
    校验 LLM 输出：
    - 长度必须与输入 fragments 一致
    - 每个 fragment 对应一个 profile
    - 每个 profile 必须包含 8 个字段
    - 每个分数必须是 [0,1] 内的数
    """
    if not isinstance(effect_profiles, list):
        raise ValueError("effect_profiles 必须是列表")

    if len(effect_profiles) != len(fragments):
        raise ValueError(
            f"effect_profiles 长度与 fragments 不一致: "
            f"{len(effect_profiles)} vs {len(fragments)}"
        )
    validated: List[Dict[str, float]] = []
    for profile in effect_profiles:
        if not isinstance(profile, dict):
            raise ValueError("effect_profile 的每一项都必须是字典")
        clean_profile: Dict[str, float] = {}
        for field in EFFECT_PROFILE_FIELDS:
            if field not in profile:
                raise ValueError(f"effect_profile 缺少字段: {field}")
            value = profile[field]
            if not isinstance(value, (int, float)):
                raise ValueError(f"{field} 必须是数值，当前值: {value}")
            value = float(value)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{field} 超出 [0,1] 范围，当前值: {value}")
            clean_profile[field] = value
        validated.append(clean_profile)
    return validated
def llm_score_effect_profiles_one_prompt(
    fragments: List[str],
    client: OpenAI,
    model: str = "gpt-4o",
) -> List[Dict[str, float]]:
    """
    输入:
        一个 prompt 的 style fragment 列表

    输出:
        与输入等长的 effect_profile 列表
    """
    messages = build_effect_profile_messages(fragments)

    response = chat_completion_with_retry(
        client,
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "fragment_effect_profile_scoring",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "effect_profiles": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "artistic_context": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1
                                    },
                                    "realism": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1
                                    },
                                    "stylization": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1
                                    },
                                    "atmospheric_intensity": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1
                                    },
                                    "sensory_salience": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1
                                    },
                                    "platform_association": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1
                                    },
                                    "depiction_explicitness": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1
                                    },
                                    "violence_graphicness": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1
                                    }
                                },
                                "required": EFFECT_PROFILE_FIELDS,
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["effect_profiles"],
                    "additionalProperties": False
                }
            }
        }
    )

    data = json.loads(response.choices[0].message.content)
    profiles = data["effect_profiles"]
    profiles = validate_effect_profiles(fragments, profiles)
    return profiles
def llm_score_effect_profiles_2d(
    fragments_2d: List[List[str]],
    client: OpenAI,
    model: str = "gpt-4o",
) -> List[List[Dict[str, float]]]:
    """
    输入:
        二维列表，每个子列表是一条 prompt 下的 style fragments

    输出:
        三维列表：
        [
            [ {fragment1的profile}, {fragment2的profile}, ... ],   # 第1条prompt
            [ {fragment1的profile}, {fragment2的profile}, ... ],   # 第2条prompt
            ...
        ]
    """
    results: List[List[Dict[str, float]]] = []
    for fragments in fragments_2d:
        profiles = llm_score_effect_profiles_one_prompt(
            fragments=fragments,
            client=client,
            model=model
        )
        results.append(profiles)
    return results

# 判断词汇数
def classify_granularity_one(fragment: str) -> str:
    """
    对单个 fragment 判定 granularity。
    词数统计直接复用 tokenize_prompts 的规则。
    """
    tokens = tokenize_prompts([fragment])[0]
    n_tokens = len(tokens)
    if n_tokens <= 1:
        return "token"
    elif 2 <= n_tokens <= 4:
        return "short_phrase"
    else:
        return "long_phrase"
def classify_granularity_2d(fragments_2d: List[List[str]]) -> List[List[str]]:
    """
    输入：
        二维 fragment 列表
    输出：
        与输入同结构的 granularity 二维列表
    """
    results: List[List[str]] = []
    for fragments in fragments_2d:
        one_prompt_result = [classify_granularity_one(fragment) for fragment in fragments]
        results.append(one_prompt_result)
    return results
if __name__ == "__main__":
    prompts = ["A cluttered alchemist’s workbench situated inside a crumbling stone tower overlooking a sea of clouds. On the table sits a bubbling glass alembic filled with iridescent liquid, a stack of scorched grimoires, and a handful of dried mandrake roots. Through the arched window, floating islands with cascading waterfalls are visible in the distance. Rendered in the style of 17th-century Dutch Golden Age painting. Utilize deep chiaroscuro to create dramatic shadows. The texture should mimic aged oil on canvas with fine craquelure, featuring a warm, sepia-toned palette and a focus on the hyper-realistic reflection of light on glassware."]
    # prompts = [ "A narrow, sweet, subterranean alleyway in a vertical megacity during a torrential downpour. A lone figure in a translucent plastic trench coat stands under a flickering holographic sign advertising 'Synthetic Memories'. Puddles on the cracked asphalt reflect the jagged pink and teal neon lights from the surrounding noodle stalls and hardware shops. A cinematic aesthetic inspired by Wong Kar-wai’s cinematography. High-contrast color grading with a heavy emphasis on 'Miami Vice' neon palettes. Apply a slight motion blur to the raindrops and use a wide-angle anamorphic lens perspective to create a sense of urban claustrophobia and melancholic isolation."]
    # prompts=["a woman lying on the sofa, happy, smooth, oil painting"]
    print(tokenize_prompts(prompts))

    # 第一步：粗切
    coarse_chunks = coarse_split_prompts(prompts)
    # print(coarse_chunks)
    # 第二步：parser 候选
    parser = build_constituency_parser()
    candidate_results = parser_candidates_for_prompts(coarse_chunks, parser)
    # print(json.dumps(candidate_results, ensure_ascii=False, indent=2))
    # 第三步：llm切割提取最小语义单元
    client = build_online_llm_client()
    final_results = llm_finalize_prompts(
        parser_results_2d=candidate_results,
        client=client,
        model="gpt-4o"
    )
    final_frag_lists = flatten_final_fragments(final_results)
    # print("=== final_results ===")
    # print(json.dumps(final_results, ensure_ascii=False, indent=2))
    print(final_results)
    print("\n=== final_frag_lists ===")
    print(json.dumps(final_frag_lists, ensure_ascii=False, indent=2))
    # 计算分割前后语义相似度
    similarity_list = compute_semantic_similarity_list(
        original_prompts=prompts,
        fragment_lists_2d=final_frag_lists,
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    print(similarity_list)
    # 区分content和style的fragment
    content_2d, style_2d = llm_split_content_and_style_fragments(
        fragments_2d=final_frag_lists,
        client=client,
        model="gpt-4o"
    )
    print("=== content fragments ===")
    print(json.dumps(content_2d, ensure_ascii=False, indent=2))
    print("=== style fragments ===")
    print(json.dumps(style_2d, ensure_ascii=False, indent=2))
    # 对风格fragment进行分类
    mechanism_category_2d = llm_classify_mechanism_categories_2d(
        fragments_2d=style_2d,
        client=client,
        model="gpt-4o"
    )
    print(json.dumps(mechanism_category_2d, ensure_ascii=False, indent=2))
    #对风格fragment进行打分
    effect_profile_3d = llm_score_effect_profiles_2d(
        fragments_2d=style_2d,
        client=client,
        model="gpt-4o"
    )
    print(json.dumps(effect_profile_3d, ensure_ascii=False, indent=2))
    # 对词汇数判定
    granularity_2d = classify_granularity_2d(style_2d)
    print(granularity_2d)
