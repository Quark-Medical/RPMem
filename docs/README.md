# RPMem documentation

Start with the repository's [CPU quick start](../README.md#try-it-on-cpu).
It requires no pretrained weights, dataset, or GPU.

## Using the method

| Goal | Read |
| --- | --- |
| Compile sessions, update memory, and generate responses | [API usage](usage.md) |
| Load compiler and downstream gate checkpoints | [Checkpoints](checkpoints.md) |

## Training and evaluation

| Goal | Read |
| --- | --- |
| Prepare compiler data and teacher targets | [Data preparation](data.md) |
| Train the compiler, consolidation gate, or transfer head | [Reproduction](reproduction.md) |
| Prepare and evaluate the three benchmarks | [Benchmark workflows](benchmarks.md) |
| Evaluate saved checkpoints across the main table | [Saved-weight evaluation](result-reproduction.md) |

Prepared datasets and trained weights are not released yet. Commands requiring
them assume that you supply the corresponding local files.

For implementation entry points, see [experiments](../experiments/README.md).
For contributions, see [CONTRIBUTING.md](../CONTRIBUTING.md).
