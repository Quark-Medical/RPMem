"""Pinned PrefEval data model shared by preparation and evaluation code."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DATASET_FORMAT = "memlora_prefeval_dataset_freeze_v1"
EXAMPLE_FORMAT = "memlora_prefeval_example_v1"
NOISE_SESSION_FORMAT = "memlora_prefeval_noise_session_v1"
SOURCE_REPOSITORY = "https://github.com/amazon-science/PrefEval"
SOURCE_REVISION = "50795054b5ff5f418d2b768a331d71e480f93331"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "8b5e1c47529cf34f2f6d313fd8607be1f6d0aa909bc94fa246e63936739f7ebe"
)
EXPECTED_BASE_EXAMPLES = 1000
EXPECTED_EXAMPLES = 3000
EXPECTED_NOISE_SESSIONS = 316
PRIMARY_TURN_COUNTS = (10, 70, 300)
OPTION_SHUFFLE_SEED = 42

# This is the exact topic order in PrefEval's SFT/train_sft.py.
TOPICS = (
    "travel_transportation",
    "shop_motors",
    "lifestyle_beauty",
    "travel_restaurant",
    "shop_fashion",
    "entertain_shows",
    "pet_ownership",
    "lifestyle_fit",
    "entertain_games",
    "shop_home",
    "lifestyle_health",
    "travel_activities",
    "education_learning_styles",
    "entertain_music_book",
    "professional_work_location_style",
    "education_resources",
    "lifestyle_dietary",
    "shop_technology",
    "travel_hotel",
    "entertain_sports",
)

# sklearn.model_selection.train_test_split(..., test_size=.2, random_state=42)
# applied to TOPICS by the official SFT implementation.
TEST_TOPICS = (
    "travel_transportation",
    "shop_technology",
    "education_resources",
    "shop_motors",
)
TRAIN_TOPICS = tuple(topic for topic in TOPICS if topic not in TEST_TOPICS)
PREFERENCE_FORMS = ("explicit", "implicit_choice", "implicit_persona")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _text(value: Any, *, field: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"PrefEval field is empty: {field}")
    return text


def _optional_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def source_paths(source_root: Path) -> list[Path]:
    paths = []
    for topic in TOPICS:
        paths.extend(
            [
                source_root / "explicit_preference" / f"{topic}.json",
                source_root
                / "implicit_preference"
                / "choice-based"
                / f"{topic}.json",
                source_root
                / "implicit_preference"
                / "persona-driven"
                / f"{topic}.json",
                source_root / "mcq_options" / f"{topic}.json",
            ]
        )
    paths.append(source_root / "filtered_inter_turns.json")
    return paths


def build_source_manifest(source_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in source_paths(source_root):
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned PrefEval source file: {path}")
        rows.append(
            {
                "path": path.relative_to(source_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def parse_choice_conversation(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, dict):
        raise ValueError("PrefEval choice conversation must be an object")
    expected = (
        "query",
        "assistant_options",
        "user_selection",
        "assistant_acknowledgment",
    )
    if tuple(raw) != expected:
        raise ValueError(
            "unexpected PrefEval choice conversation fields/order: "
            f"expected={expected} actual={tuple(raw)}"
        )
    roles = ("user", "assistant", "user", "assistant")
    return [
        {"role": role, "content": _text(raw[key], field=f"conversation.{key}")}
        for key, role in zip(expected, roles)
    ]


def parse_persona_conversation(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("PrefEval persona conversation must be a non-empty object")
    try:
        keys = sorted(raw, key=lambda value: int(value))
    except ValueError as exc:
        raise ValueError("PrefEval persona turn keys must be integers") from exc
    if [int(value) for value in keys] != list(range(len(keys))):
        raise ValueError("PrefEval persona turn indices must be contiguous from zero")
    messages = []
    for key in keys:
        turn = raw[key]
        if not isinstance(turn, dict) or set(turn) != {"user", "assistant"}:
            raise ValueError(f"invalid PrefEval persona turn: {key}")
        messages.extend(
            [
                {
                    "role": "user",
                    "content": _text(turn["user"], field=f"conversation.{key}.user"),
                },
                {
                    "role": "assistant",
                    "content": _text(
                        turn["assistant"], field=f"conversation.{key}.assistant"
                    ),
                },
            ]
        )
    return messages


def parse_noise_sessions(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("PrefEval noise source must be a list")
    sessions = []
    for conversation_index, conversation in enumerate(raw):
        messages = conversation.get("conversation")
        if not isinstance(messages, list) or len(messages) % 2:
            raise ValueError(
                f"PrefEval noise conversation {conversation_index} is not paired"
            )
        for offset in range(0, len(messages), 2):
            pair = messages[offset : offset + 2]
            if [item.get("role") for item in pair] != ["user", "assistant"]:
                raise ValueError(
                    "PrefEval noise messages must alternate user/assistant: "
                    f"conversation={conversation_index} offset={offset}"
                )
            sessions.append(
                {
                    "format": NOISE_SESSION_FORMAT,
                    "session_id": f"noise-{len(sessions):04d}",
                    "source_conversation_id": str(
                        conversation.get("conversation_id", conversation_index)
                    ),
                    "source_message_offset": offset,
                    "messages": [
                        {
                            "role": str(item["role"]),
                            "content": _text(
                                item.get("content"),
                                field=(
                                    f"noise[{conversation_index}]"
                                    f".conversation[{offset}]"
                                ),
                            ),
                        }
                        for item in pair
                    ],
                }
            )
    return sessions


def _load_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"PrefEval source must be a list of objects: {path}")
    return value


def _assert_aligned(
    topic: str,
    index: int,
    explicit: dict[str, Any],
    choice: dict[str, Any],
    persona: dict[str, Any],
    mcq: dict[str, Any],
) -> None:
    for field in ("question", "explanation"):
        values = [row.get(field) for row in (explicit, choice, persona, mcq)]
        if len({str(value) for value in values}) != 1:
            raise ValueError(
                f"PrefEval sources are misaligned: topic={topic} index={index} "
                f"field={field}"
            )
    implicit_preferences = [
        row.get("preference") for row in (choice, persona, mcq)
    ]
    if len({str(value) for value in implicit_preferences}) != 1:
        raise ValueError(
            "PrefEval implicit sources are misaligned: "
            f"topic={topic} index={index} field=preference"
        )


def build_examples(
    source_root: Path,
    *,
    require_expected_counts: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    examples = []
    topic_counts: dict[str, int] = {}
    for topic in TOPICS:
        explicit_rows = _load_list(
            source_root / "explicit_preference" / f"{topic}.json"
        )
        choice_rows = _load_list(
            source_root
            / "implicit_preference"
            / "choice-based"
            / f"{topic}.json"
        )
        persona_rows = _load_list(
            source_root
            / "implicit_preference"
            / "persona-driven"
            / f"{topic}.json"
        )
        mcq_rows = _load_list(source_root / "mcq_options" / f"{topic}.json")
        counts = {
            len(explicit_rows),
            len(choice_rows),
            len(persona_rows),
            len(mcq_rows),
        }
        if len(counts) != 1:
            raise ValueError(f"PrefEval topic sources have different lengths: {topic}")
        topic_counts[topic] = len(explicit_rows)
        split = "test" if topic in TEST_TOPICS else "train"
        for index, (explicit, choice, persona, mcq) in enumerate(
            zip(explicit_rows, choice_rows, persona_rows, mcq_rows)
        ):
            _assert_aligned(topic, index, explicit, choice, persona, mcq)
            options = mcq.get("classification_task_options")
            if (
                not isinstance(options, list)
                or len(options) != 4
                or any(not str(value).strip() for value in options)
            ):
                raise ValueError(f"invalid PrefEval MCQ options: {topic} row {index}")
            common = {
                "format": EXAMPLE_FORMAT,
                "source_example_id": f"prefeval_{topic}_{index:04d}",
                "topic": topic,
                "source_index": index,
                "split": split,
                "question": _text(explicit.get("question"), field="question"),
                "explanation": _optional_text(explicit.get("explanation")),
                "options": [str(value).strip() for value in options],
                "correct_option_index": 0,
            }
            form_preferences = {
                "explicit": _text(
                    explicit.get("preference"), field="explicit.preference"
                ),
                "implicit_choice": _text(
                    choice.get("preference"), field="implicit_choice.preference"
                ),
                "implicit_persona": _text(
                    persona.get("preference"), field="implicit_persona.preference"
                ),
            }
            form_messages = {
                "explicit": [
                    {"role": "user", "content": form_preferences["explicit"]}
                ],
                "implicit_choice": parse_choice_conversation(
                    choice.get("conversation")
                ),
                "implicit_persona": parse_persona_conversation(
                    persona.get("conversation")
                ),
            }
            for form in PREFERENCE_FORMS:
                examples.append(
                    {
                        **common,
                        "instance_id": f"{common['source_example_id']}_{form}",
                        "preference_form": form,
                        "preference": form_preferences[form],
                        "memory_sessions": [
                            {
                                "session_id": "preference",
                                "messages": form_messages[form],
                            }
                        ],
                    }
                )
    if require_expected_counts and (
        sum(topic_counts.values()) != EXPECTED_BASE_EXAMPLES
        or len(examples) != EXPECTED_EXAMPLES
    ):
        raise ValueError(
            "unexpected PrefEval cardinality: "
            f"base={sum(topic_counts.values())} examples={len(examples)}"
        )
    return examples, topic_counts


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    *,
    require_pinned_source: bool = True,
    require_expected_counts: bool = True,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    source_manifest = build_source_manifest(source_root)
    source_manifest_sha256 = canonical_sha256(source_manifest)
    if (
        require_pinned_source
        and source_manifest_sha256 != EXPECTED_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError(
            "PrefEval source does not match the pinned revision: "
            f"expected={EXPECTED_SOURCE_MANIFEST_SHA256} "
            f"actual={source_manifest_sha256}"
        )
    examples, topic_counts = build_examples(
        source_root, require_expected_counts=require_expected_counts
    )
    noise = parse_noise_sessions(
        json.loads(
            (source_root / "filtered_inter_turns.json").read_text(encoding="utf-8")
        )
    )
    if require_expected_counts and len(noise) != EXPECTED_NOISE_SESSIONS:
        raise ValueError(
            f"PrefEval has {len(noise)} noise sessions, expected "
            f"{EXPECTED_NOISE_SESSIONS}"
        )

    examples_path = output_root / "examples.jsonl"
    noise_path = output_root / "noise_sessions.jsonl"
    source_manifest_path = output_root / "source_manifest.json"
    write_jsonl_atomic(examples_path, examples)
    write_jsonl_atomic(noise_path, noise)
    write_json_atomic(source_manifest_path, source_manifest)

    split_counts = Counter(str(row["split"]) for row in examples)
    form_counts = Counter(str(row["preference_form"]) for row in examples)
    contract = {
        "primary_turn_counts": list(PRIMARY_TURN_COUNTS),
        "topic_split_source": (
            "official SFT train_test_split(test_size=0.2, random_state=42)"
        ),
        "train_topics": list(TRAIN_TOPICS),
        "test_topics": list(TEST_TOPICS),
        "option_shuffle": {
            "algorithm": "sha256_sort_v1",
            "seed": OPTION_SHUFFLE_SEED,
        },
        "memory_order": "preference_session_then_first_n_official_noise_sessions",
        "question_visible_to_memory_writer": False,
        "explicit_preference_normalization": (
            "single_user_message_without_model_generated_acknowledgement"
        ),
    }
    dataset_sha256 = canonical_sha256(
        {
            "examples_sha256": sha256_file(examples_path),
            "noise_sessions_sha256": sha256_file(noise_path),
            "source_manifest_sha256": source_manifest_sha256,
            "contract": contract,
        }
    )
    freeze = {
        "format": DATASET_FORMAT,
        "dataset": "prefeval",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_manifest_file": source_manifest_path.name,
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_file_sha256": sha256_file(source_manifest_path),
        "examples_file": examples_path.name,
        "examples_sha256": sha256_file(examples_path),
        "noise_sessions_file": noise_path.name,
        "noise_sessions_sha256": sha256_file(noise_path),
        "dataset_sha256": dataset_sha256,
        "base_examples": sum(topic_counts.values()),
        "examples": len(examples),
        "noise_sessions": len(noise),
        "topic_counts": topic_counts,
        "split_counts": dict(sorted(split_counts.items())),
        "preference_form_counts": dict(sorted(form_counts.items())),
        "contract": contract,
    }
    write_json_atomic(output_root / "memlora_dataset_freeze.json", freeze)
    return freeze


def validate_dataset_root(
    root: str | Path,
    *,
    require_expected_counts: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(root)
    freeze_path = root / "memlora_dataset_freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(
            "formal PrefEval data is incomplete; run "
            f"experiments/prefeval/prepare_formal_data.sh: {root}"
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("format") != DATASET_FORMAT:
        raise ValueError(f"unsupported PrefEval dataset freeze: {freeze_path}")
    files = (
        ("examples_file", "examples_sha256"),
        ("noise_sessions_file", "noise_sessions_sha256"),
        ("source_manifest_file", "source_manifest_file_sha256"),
    )
    for file_key, sha_key in files:
        path = root / str(freeze[file_key])
        if not path.is_file() or sha256_file(path) != str(freeze[sha_key]):
            raise ValueError(f"PrefEval checksum mismatch: {path}")
    examples = read_jsonl(root / str(freeze["examples_file"]))
    noise = read_jsonl(root / str(freeze["noise_sessions_file"]))
    if require_expected_counts and (
        len(examples) != EXPECTED_EXAMPLES
        or len(noise) != EXPECTED_NOISE_SESSIONS
        or freeze.get("source_manifest_sha256")
        != EXPECTED_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("formal PrefEval data does not match the pinned release")
    ids = [str(row.get("instance_id")) for row in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("PrefEval instance IDs are not unique")
    return freeze, examples, noise


def materialize_history(
    example: dict[str, Any],
    noise_sessions: list[dict[str, Any]],
    turn_count: int,
) -> list[dict[str, Any]]:
    if turn_count < 0 or turn_count > len(noise_sessions):
        raise ValueError(
            f"invalid PrefEval turn count {turn_count}; available={len(noise_sessions)}"
        )
    sessions = [dict(session) for session in example["memory_sessions"]]
    sessions.extend(
        {
            "session_id": str(session["session_id"]),
            "messages": list(session["messages"]),
        }
        for session in noise_sessions[:turn_count]
    )
    return sessions


def materialized_rows(
    examples: Iterable[dict[str, Any]],
    noise_sessions: list[dict[str, Any]],
    *,
    split: str,
    turn_counts: tuple[int, ...] = PRIMARY_TURN_COUNTS,
) -> list[dict[str, Any]]:
    """Expand frozen form rows over the formal history-length axis."""

    rows = []
    for example in examples:
        if str(example["split"]) != split:
            continue
        for turn_count in turn_counts:
            rows.append(
                {
                    **example,
                    "base_instance_id": str(example["instance_id"]),
                    "instance_id": f"{example['instance_id']}_turns{turn_count:03d}",
                    "turn_count": int(turn_count),
                    "memory_sessions": materialize_history(
                        example, noise_sessions, turn_count
                    ),
                }
            )
    return rows


def shuffled_options(
    example: dict[str, Any],
    *,
    seed: int = OPTION_SHUFFLE_SEED,
) -> tuple[list[str], int]:
    options = list(example["options"])
    correct = int(example["correct_option_index"])
    if correct < 0 or correct >= len(options):
        raise ValueError("PrefEval correct option index is out of range")
    keyed = list(enumerate(options))
    keyed.sort(
        key=lambda item: hashlib.sha256(
            f"{seed}\0{example['instance_id']}\0{item[0]}".encode("utf-8")
        ).digest()
    )
    shuffled = [value for _, value in keyed]
    shuffled_correct = next(
        index
        for index, (source_index, _) in enumerate(keyed)
        if source_index == correct
    )
    return shuffled, shuffled_correct
