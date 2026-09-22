# Contributing to RPMem

This repository contains the paper's method library, main-method training and
evaluation workflows, and project page. Please keep changes scoped to these
components. Internal cluster launchers, private storage integrations, baseline
sweeps, and plotting pipelines are maintained separately.

## Development

Use a dedicated Python environment and install the PyTorch build appropriate for
your hardware first. Then install the development and workflow dependencies:

```bash
python -m pip install -e '.[dev,train,experiments]'
python examples/core_smoke.py
python -m build
```

The CPU example uses tiny randomly initialized modules and does not download
pretrained models. Include a minimal reproduction with changes that fix a bug.

## Method changes

Separate API and packaging fixes from changes to the learned method. Changes to
session segmentation, token budgets, objectives, Gate initialization, LoRA scaling,
or scoring protocols can invalidate old weights or results. Describe these changes
explicitly and do not present previously measured scores as new validation.

Keep checkpoint readers compatible with documented historical formats where
possible. Never rewrite a user's existing checkpoint or result directory as part
of migration.

## Reporting issues

Include the command, Python/PyTorch/Transformers versions, GPU model when relevant,
and a minimal traceback. State whether the failure occurs in the core CPU example,
pretrained inference, or a particular benchmark stage. Redact credentials and user
conversation content before posting logs. Do not upload private training data or
third-party model weights in an issue.

## Website

The [project page](https://quark-medical.github.io/rpmem/) is
maintained on this repository's [gh-pages branch](https://github.com/Quark-Medical/rpmem/tree/gh-pages).
Submit website changes against `gh-pages`; pushes to that branch automatically
deploy through GitHub Pages. The `main` branch contains the method code.
Use paper figures and reported numbers, not newly inferred results, and preserve
the website's template and asset attributions.
