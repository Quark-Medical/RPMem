"""
PERMA 数据适配器：加载 PERMA benchmark 的多 session 对话数据，
并提供 session 文本化与任务元信息解析。
"""
import json
import os
import re
from dataclasses import dataclass
from typing import Optional


PERMA_DATA_ROOT = os.environ.get(
    "PERMA_DATA_ROOT",
    os.path.join(os.path.dirname(__file__), "data"),
)

COUNTRY_USERS = {
    "Canada": [334], "Mexico": [354], "Finland": [123],
    "United States": [1377], "Australia": [507],
    "United Kingdom": [914], "Switzerland": [112],
    "Israel": [419], "Russian Federation": [108], "Belgium": [109],
}
ALL_USER_IDS = [uid for ids in COUNTRY_USERS.values() for uid in ids]


@dataclass
class PermaTask:
    user_id: int
    task_id: str
    task_type: int            # 1=Zero-Memory, 2=In-Time, 3=Post-Intervention
    variant: str
    topic: list[str]
    sessions: list[dict]      # [{"text": str, "date": str}, ...]
    raw_conversations: list   # [[{role, content}, ...], ...] per session
    question: str
    options: list[str]
    gold_label: str
    preferences: list[str]
    affinity_links: list[dict]


def _parse_options(raw) -> list[str]:
    """解析 PERMA 选项：原始格式为 'A: text\\nB: text\\n...' 的单个字符串"""
    if isinstance(raw, list):
        return raw
    matches = list(re.finditer(r'(?:^|\n)([A-Z]):\s', raw))
    if not matches:
        return [raw]
    options = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        options.append(raw[start:end].strip())
    return options


def _session_to_text(conversation: list[dict]) -> str:
    return "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in conversation
    )


def _flatten_sessions(context_list: list) -> tuple[list[dict], list]:
    """将 PERMA 的 context 格式转为 [{text, date}, ...] 和原始对话列表"""
    sessions = []
    raw_conversations = []
    for entry in context_list:
        conv = entry[0]  # list of {role, content}
        date = entry[1] if len(entry) > 1 else ""
        sessions.append({
            "text": _session_to_text(conv),
            "date": str(date),
        })
        raw_conversations.append(conv)
    return sessions, raw_conversations


# PERMA 子集变体 → input_data 文件后缀
#   c=clean, n=noisy, s=style, s_long=style long-context；multi_ 前缀=multi-domain
PERMA_VARIANTS = {
    "clean_sd": "_c",
    "noisy_sd": "_n",
    "style_sd": "_s",
    "style_long_sd": "_s_long",
    "clean_md": "_multi_c",
    "noisy_md": "_multi_n",
    "style_md": "_multi_s",
}


def load_tasks(
    user_ids: Optional[list[int]] = None,
    noise: bool = False,
    multi_domain: bool = False,
    variant: Optional[str] = None,
) -> list[PermaTask]:
    """加载 PERMA 任务。

    variant 优先：可传 PERMA_VARIANTS 的键（如 "clean_md"/"noisy_sd"/"style_long_sd"）
    或直接传 input_data 文件后缀（如 "_multi_c"）。variant=None 时回退到
    noise/multi_domain 的布尔组合（向后兼容，仅覆盖 clean/noisy × sd/md）。
    """
    if user_ids is None:
        user_ids = ALL_USER_IDS

    if variant is not None:
        file_suffix = PERMA_VARIANTS.get(variant, variant)
        if not file_suffix.startswith("_"):
            file_suffix = f"_{file_suffix}"
    else:
        version = "_multi" if multi_domain else ""
        suffix = "_n" if noise else "_c"
        file_suffix = f"{version}{suffix}"
    tasks = []

    for uid in user_ids:
        task_path = os.path.join(
            PERMA_DATA_ROOT, "tasks", f"user{uid}",
            f"input_data{file_suffix}.json",
        )
        if not os.path.exists(task_path):
            continue

        with open(task_path, "r") as f:
            data = json.load(f)

        meta_dir = os.path.join(
            PERMA_DATA_ROOT, "evaluation", f"user{uid}", "meta", "overall",
        )

        for ev in data.get("overall", []):
            task_id = ev.get("task_id", "")
            task_type = int(ev.get("type", 0))
            context = ev.get("context", [])
            sessions, raw_conversations = _flatten_sessions(context)

            meta_path = os.path.join(meta_dir, f"{task_id}_{task_type}.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r") as f:
                meta = json.load(f)

            tasks.append(PermaTask(
                user_id=uid,
                task_id=task_id,
                task_type=task_type,
                variant=variant or file_suffix,
                topic=ev.get("topic", []),
                sessions=sessions,
                raw_conversations=raw_conversations,
                question=meta.get("question", ""),
                options=_parse_options(meta.get("options", [])),
                gold_label=meta.get("gold_label", ""),
                preferences=ev.get("preferences", []),
                affinity_links=ev.get("affinity_links", []),
            ))

    return tasks
