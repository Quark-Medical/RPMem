# Third-party notices

The following notices apply to retained third-party portions in the listed
files. RPMem's own code is licensed under Apache-2.0. The retained third-party
portions remain subject to their original license terms.

## Perceiver implementation

`rpmem/encoder/perceiver.py` adapts
`src/ctx_to_lora/modeling/idefics2.py` from
[Doc-to-LoRA](https://github.com/SakanaAI/Doc-to-LoRA).
That upstream file carries the following notice:

> Copyright 2024 the HuggingFace Inc. team. All rights reserved.

The upstream file is licensed under Apache-2.0; its copyright and license
header are retained in the modified file, and the license text is included in
`LICENSES/Apache-2.0.txt`. Modifications include plain PyTorch module interfaces,
local configuration, and an optional SDPA attention path. The upstream project's
MIT notice is also retained for applicable Doc-to-LoRA contributions.

## Adapter generation and LoRA utilities

The following files retain portions adapted from Doc-to-LoRA:

| Local file | Upstream file under `src/ctx_to_lora/modeling/` |
| --- | --- |
| `rpmem/head/layers.py` | `hypernet.py` |
| `rpmem/head/hypernet_head.py` | `hypernet.py` |
| `rpmem/lora/injection.py` | `lora_layer.py` |
| `rpmem/lora/merger.py` | `lora_merger.py` |

The MIT copyright and permission notice for these portions is preserved in
`LICENSES/Doc-to-LoRA-MIT.txt` (Copyright (c) 2026 Sakana AI).
Changes include direct PyTorch tensor operations, standalone configuration,
and target-module and input-shape handling.

Doc-to-LoRA is not a runtime dependency. The upstream checkpoint conversion
entry point is not included. The public Python namespace is `rpmem`.
`rpmem/checkpoint/compat.py` supports trusted native research checkpoints that
contain configuration objects saved under the historical package name.

## External dependencies

PyTorch, Transformers, NumPy, PyArrow, PyYAML, Accelerate, TensorBoard, and
optional FlashAttention are external dependencies with their own licenses.
No model weights or benchmark datasets are bundled in this snapshot.
