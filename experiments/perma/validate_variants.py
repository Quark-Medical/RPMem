"""
PERMA variant 严格数据体检（fail fast）。

直接读 input_data 与 meta 文件，逐 task 检查 meta 是否存在——绕过 load_tasks
的静默 continue，避免"静默少样本"污染矩阵。对每个 variant × user 报告：
  raw task 数 / meta 匹配数 / 缺失文件 / 缺失 meta 列表 / type 分布。
任何缺失文件或缺失 meta 默认以非零退出码失败（除非 --no_strict）。

用法:
  PERMA_DATA_ROOT=experiments/perma/data \
  python experiments/perma/validate_variants.py
  # 只查部分 variant:
  python experiments/perma/validate_variants.py --variants clean_sd noisy_sd clean_md
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from experiments.perma.data_adapter import (
    ALL_USER_IDS,
    PERMA_DATA_ROOT,
    PERMA_VARIANTS,
)


def check_variant(variant: str, max_show: int) -> dict:
    suffix = PERMA_VARIANTS.get(variant, variant)
    if not suffix.startswith("_"):
        suffix = f"_{suffix}"

    total_raw = 0
    total_matched = 0
    missing_files = []
    missing_meta = []
    type_counter = Counter()
    per_user = []

    for uid in ALL_USER_IDS:
        task_path = os.path.join(
            PERMA_DATA_ROOT, "tasks", f"user{uid}", f"input_data{suffix}.json",
        )
        if not os.path.exists(task_path):
            missing_files.append(f"user{uid}: {task_path}")
            per_user.append((uid, 0, 0))
            continue

        with open(task_path) as f:
            data = json.load(f)
        overall = data.get("overall", [])
        meta_dir = os.path.join(
            PERMA_DATA_ROOT, "evaluation", f"user{uid}", "meta", "overall",
        )

        u_raw, u_matched = 0, 0
        for ev in overall:
            task_id = ev.get("task_id", "")
            task_type = int(ev.get("type", 0))
            u_raw += 1
            meta_path = os.path.join(meta_dir, f"{task_id}_{task_type}.json")
            if os.path.exists(meta_path):
                u_matched += 1
                type_counter[task_type] += 1
            else:
                missing_meta.append(f"user{uid}: {task_id}_{task_type}")
        total_raw += u_raw
        total_matched += u_matched
        per_user.append((uid, u_raw, u_matched))

    print(f"\n=== {variant}  (input_data{suffix}.json) ===")
    print(f"  raw tasks: {total_raw} | meta matched: {total_matched} | "
          f"missing meta: {len(missing_meta)} | missing files: {len(missing_files)}")
    print(f"  type 分布 (matched): " +
          ", ".join(f"T{t}={type_counter[t]}" for t in sorted(type_counter)))
    uneven = [f"user{u}({r}->{m})" for u, r, m in per_user if r != m or r == 0]
    if uneven:
        print(f"  ⚠ 不一致/空的 user: {', '.join(uneven)}")
    for item in (missing_files + missing_meta)[:max_show]:
        print(f"    - missing: {item}")
    extra = len(missing_files) + len(missing_meta) - max_show
    if extra > 0:
        print(f"    ... 还有 {extra} 条缺失未显示")

    return {
        "variant": variant,
        "total_raw": total_raw,
        "total_matched": total_matched,
        "n_missing_meta": len(missing_meta),
        "n_missing_files": len(missing_files),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=list(PERMA_VARIANTS.keys()))
    parser.add_argument("--max_show", type=int, default=20)
    parser.add_argument("--no_strict", action="store_true",
                        help="不因缺失而失败（默认缺失即 fail fast）")
    args = parser.parse_args()

    print(f"PERMA_DATA_ROOT = {PERMA_DATA_ROOT}")
    print(f"检查 {len(args.variants)} 个 variant × {len(ALL_USER_IDS)} users")

    summaries = [check_variant(v, args.max_show) for v in args.variants]

    print("\n" + "=" * 60)
    print("汇总:")
    bad = False
    for s in summaries:
        flag = ""
        if s["n_missing_files"] or s["n_missing_meta"]:
            flag = "  ❌"
            bad = True
        print(f"  {s['variant']:16s}: matched {s['total_matched']}/{s['total_raw']}{flag}")

    if bad and not args.no_strict:
        print("\n存在缺失文件或缺失 meta，fail fast 退出（用 --no_strict 可忽略）。")
        sys.exit(1)
    print("\n全部 variant 数据完整 ✓" if not bad else "\n（--no_strict：忽略缺失继续）")


if __name__ == "__main__":
    main()
