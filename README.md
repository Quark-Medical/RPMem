<h1 align="center">RPMem: Learning Long-Term Recurrent Parametric Memory Across Sessions for LLM Agents</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.23466"><img src="https://img.shields.io/badge/arXiv-2609.23466-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="Paper on arXiv"></a>
  <a href="https://quark-medical.github.io/rpmem/"><img src="https://img.shields.io/badge/Project-Page-146C70?style=for-the-badge" alt="Project page"></a>
  <a href="docs/README.md"><img src="https://img.shields.io/badge/Read-Documentation-2563EB?style=for-the-badge&amp;logo=readthedocs&amp;logoColor=white" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-287C61?style=for-the-badge&amp;logo=apache&amp;logoColor=white" alt="Apache 2.0 license"></a>
</p>

<p align="center">
  <a href="#overview">🧠 Overview</a> &middot;
  <a href="#installation">🚀 Installation</a> &middot;
  <a href="#usage">💻 Usage</a> &middot;
  <a href="#training-and-evaluation">🛠️ Training</a> &middot;
  <a href="#citation">📝 Citation</a>
</p>

## Overview

**Give your LLM a memory that persists across sessions, without replaying the full conversation.**

RPMem turns past interactions into a compact, recurrent memory that conditions
an LLM through generated LoRA parameters. Each new session updates the memory;
the reader answers using that state rather than retrieving or re-encoding the
entire history. The base LLM stays frozen.

<p align="center">
  <a href="assets/method.png"><img src="assets/method.png" width="100%" alt="RPMem: single-session compilation, recurrent consolidation, and deployment through a backbone-specific LoRA decoder."></a>
  <br>
  <em>Compile each session. Consolidate across sessions. Decode memory into model parameters.</em>
</p>

**Two-stage learning.** A session compiler learns from a history-aware teacher
with Fixed FKL. A lightweight recurrent gate then learns to retain and update
session memory under downstream supervision.

**Fixed-size memory, adaptable readers.** The retained state does not grow with
the number of sessions. Backbone-specific decoder heads let the shared memory
representation support different LLMs.

This repository provides the installable `rpmem` library, training and inference
code, data preparation, and main-method evaluation on **PERMA**, **PersonaMem-v2**,
and **PrefEval**.

> Our implementation builds extensively on
> [Doc-to-LoRA (D2L)](https://github.com/SakanaAI/Doc-to-LoRA) by Sakana AI.
> We thank its authors for making their code available. See
> [Acknowledgements](#acknowledgements) for the adapted components and attribution.

## Installation

Use **Python 3.10+** and a dedicated environment:

```bash
git clone https://github.com/Quark-Medical/rpmem.git
cd rpmem
python -m venv .venv
source .venv/bin/activate
```

For full-size training and inference, install the PyTorch build for your CUDA
environment inside this environment first. Then install RPMem. FlashAttention
is optional; the documented examples use PyTorch SDPA.

```bash
python -m pip install -e .
```

### Try it on CPU

No GPU, dataset, or pretrained checkpoint is needed:

```bash
python examples/core_smoke.py
```

The example compiles synthetic session features, consolidates memory, and trains
the gate using tiny random modules. A successful run prints
`Gate training passed; compiler and backbone remained frozen.`

> **Data and checkpoints:** Prepared datasets and teacher targets are not released
> yet; trained compiler and gate checkpoints are not included. Data-preparation
> code is provided. For the workflows below, supply your own prepared inputs and
> matching checkpoints. See [data preparation](docs/data.md).

## Usage

With a trained compiler and gate, update memory as sessions arrive and apply it
when answering a query:

```python
from rpmem import RPMemModel

model = RPMemModel.from_checkpoint(
    "checkpoints/compiler.bin",
    gate_ckpt_path="checkpoints/gate.pt",
    base_model_path="Qwen/Qwen3-8B",
    ctx_encoder_path="answerdotai/ModernBERT-base",
    use_flash_attn=False,
)

memory = model.update_memory("User: I prefer vegetarian meals.")
memory = model.update_memory("User: I now avoid dairy, too.", memory)

model.apply_memory(memory)
answer = model.generate("What could I have for dinner?")
print(answer)
model.reset()  # Remove the injected memory before serving another user.
```

The memory tensor is explicit: keep one state per user and pass it into the next
update. See [API usage](docs/usage.md) for saving state and session token budgets,
and [checkpoint setup](docs/checkpoints.md) for compatible compiler/gate pairs.

## Training and Evaluation

Install the training and benchmark dependencies:

```bash
python -m pip install -e '.[train,experiments]'
```

### Stage 1: Train the session compiler

Prepare the session/probe corpus and matching teacher targets using the
[data guide](docs/data.md). The following uses the Qwen3-8B configuration
with eight GPUs; `CORPUS_ROOT` and `TEACHER_ROOT` point to your local inputs.

```bash
export CORPUS_ROOT=/path/to/compiler-corpus
export TEACHER_ROOT=/path/to/qwen3-8b-teacher

accelerate launch --config_file configs/accelerate_8gpu.yaml -m rpmem.training.train_hypernet \
  --config configs/compiler/qwen3_8b.yaml --train_data "$CORPUS_ROOT/train.corpus.json" \
  --teacher_logprobs_dir "$TEACHER_ROOT/train" --output_dir outputs/compiler \
  --no-use_flash_attn
```

### Stage 2: Train and evaluate consolidation

Freeze the compiler and train a downstream gate. For one PERMA variant and
held-out user:

```bash
CHECKPOINT=outputs/compiler/pytorch_model.bin \
BASE_MODEL_PATH=Qwen/Qwen3-8B CTX_ENCODER_PATH=answerdotai/ModernBERT-base \
PERMA_DATA_ROOT=data/perma FIRST_SESSION_RULE=gate_zero_state \
VARIANT=clean_sd TEST_USER=334 bash scripts/run_perma.sh
```

The full PERMA protocol has seven variants and ten held-out users. PersonaMem-v2
and PrefEval use their own train/test splits. Follow the guides for the full run:

| Goal | Start here |
| :--- | :--- |
| Train from prepared data, including teacher-target generation when needed | [Training guide](docs/reproduction.md) |
| Prepare and run each of the three benchmarks | [Benchmark workflows](docs/benchmarks.md) |
| Evaluate the main-method table from saved checkpoints | [Saved-weight evaluation](docs/result-reproduction.md) |
| Transfer memory to another LLM backbone | [Decoder-head transfer](docs/reproduction.md#decoder-head-transfer) |

<details>
<summary><strong>Reproducing the paper's first-session initialization</strong></summary>

The paper results used `h1 = gate(0, q1)` (`gate_zero_state`), as selected
in the training command. New gates default to `h1 = q1` (`direct`); saved gates
automatically restore their recorded convention. These are different training
configurations. See [initialization compatibility](docs/usage.md#first-session-initialization)
for details.

</details>

## Repository Layout

The library lives in `rpmem/`; main-method workflows are in `experiments/`,
model configurations in `configs/`, and runnable examples in `examples/`.
The [documentation index](docs/README.md) links to the complete guides.

Questions, bug reports, and contributions are welcome:
see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If RPMem is useful for your work, please cite:

```bibtex
@article{zhao2026rpmem,
  title={RPMem: Learning Long-Term Recurrent Parametric Memory Across Sessions for LLM Agents},
  author={Zhao, Fanyu and Cao, Ruike and Dong, Liang and Yao, Fugen and Xu, Jian and
          Jiang, Guanjun and Zhang, Han and Zhao, Yifei and Li, Yinsheng},
  journal={arXiv preprint arXiv:2609.23466},
  year={2026},
  url={https://arxiv.org/abs/2609.23466}
}
```

Machine-readable citation: [CITATION.cff](CITATION.cff).

## Acknowledgements

This work was supported by Qwen Business Unit through Alibaba Research Intern Program.

This codebase builds extensively on the open-source implementation of
**[Doc-to-LoRA (D2L)](https://github.com/SakanaAI/Doc-to-LoRA)** by Sakana AI,
including the Perceiver resampler, hypernetwork-based LoRA generation, and
LoRA injection and merging utilities. We are grateful to the D2L authors and
contributors for sharing their work, which provided an important foundation for
this implementation. When building on these components, please also cite the
[Doc-to-LoRA paper](https://arxiv.org/abs/2602.15902).

We also thank the Hugging Face team for the Idefics2 implementation used by the
resampler, and the authors of PERMA, PersonaMem-v2, and PrefEval for their
benchmarks. File-level attribution and the retained upstream licenses are
documented in [third-party notices](THIRD_PARTY_NOTICES.md).

## License

Code: [Apache-2.0](LICENSE). See [NOTICE](NOTICE) and
[third-party notices](THIRD_PARTY_NOTICES.md) for attribution. Upstream datasets
and models retain their own terms. The separately hosted
[project page](https://quark-medical.github.io/rpmem/) has its
own [license and attributions](https://github.com/Quark-Medical/rpmem/tree/gh-pages).
