# SCoNE: Selecting Context-aware Neurons for RAG

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

SCoNE improves robustness to retrieval noise in RAG by identifying and reweighting selectively context-aware neurons, without fine-tuning or additional inference-time modules.

Built on [BERGEN](https://github.com/naver/bergen) — this repository is a derivative of
BERGEN and is released under the same CC BY-NC-SA 4.0 license. It also includes an
[IRCAN](https://github.com/danshi777/IRCAN) (Shi et al., NeurIPS 2024) baseline, plus CAD,
RankCoT, Ret-Robust and PA-RAG generator configs, so every method runs through one pipeline.

The core implementation is in [`models/generators/neuron_strategies.py`](models/generators/neuron_strategies.py)
(neuron mining, scoring and reweighting), wired into the generation path in
[`models/generators/generator.py`](models/generators/generator.py) and
[`models/generators/llm.py`](models/generators/llm.py).

SCoNE modifies BERGEN's generation and retrieval pipeline and adds new modules; the
original BERGEN README is preserved at [documentation/BERGEN_README.md](documentation/BERGEN_README.md).

## Installation
```bash
conda create -n scone python=3.10
conda activate scone
pip install -r SCoNE-requirements.txt
```


## Quick Start

The following command mines selectively context-aware neurons from 100 HotpotQA training samples and
evaluates SCoNE using the selected neurons. In this configuration, SCoNE selects the top-5 neurons
using local candidate pools of 50 neurons for attribution strength and cross-input variability, and
applies an enhancement strength of $\alpha=7$.

```bash
CUDA_VISIBLE_DEVICES=0 python bergen.py \
    generator=llama-3-8b-instruct \
    dataset=kilt_hotpotqa \
    retriever=splade-v3 \
    generator.init_args.use_attr=true \
    generator.init_args.attr_ds_name=hotpotqa \
    generator.init_args.num_attr_samples=100 \
    experiment_mode=scone \
    enhance_strength=7.0 \
    save_result=true \
    top_k=5 \
    top_n=50
```

* `experiment_mode=scone`: runs SCoNE using both attribution strength and cross-input variability.
* `num_attr_samples=100`: sets the number of HotpotQA training samples used for neuron mining.
* `top_n=50`: sets the size of each local candidate pool before intersection.
* `top_k=5`: sets the final number of selected neurons.
* `enhance_strength=7.0`: sets the neuron enhancement strength $\alpha$.

The repository defaults in `config/rag.yaml` are `top_k=50`, `top_n=50` and `enhance_strength=5.0`,
so the values above must be passed explicitly to reproduce the reported setting.

## Ablation Studies

The main ablations can be reproduced by changing `experiment_mode` or the corresponding
hyperparameter while keeping the remaining settings identical.

### Neuron Selection Criteria

```bash
# Attribution only
experiment_mode=scone_attronly

# Cross-input variability only
experiment_mode=scone_varonly

# Full SCoNE
experiment_mode=scone
```

### Variability Measures

```bash
# Running residual (SCoNE)
experiment_mode=scone

# Variance
experiment_mode=scone_var

# Standard deviation
experiment_mode=scone_std

# Mean absolute deviation (MAD)
experiment_mode=scone_mad
```

### Hyperparameters

The hyperparameter analyses can be reproduced by varying the corresponding arguments:

```bash
# Number of selected neurons
top_k=5
top_k=15

# Local candidate pool size
top_n=20
top_n=50
top_n=80

# Enhancement strength
enhance_strength=3.0
enhance_strength=5.0
enhance_strength=7.0
```

For the context-window ablation, use:

```bash
experiment_mode=scone_w1
experiment_mode=scone_w2
experiment_mode=scone
experiment_mode=scone_w5
experiment_mode=scone_w10
```

`experiment_mode=scone` is the window-3 setting, so it serves as the mid-point of this sweep.
All other experimental settings should be kept identical to the main configuration.

## Baselines

All baselines run through the same pipeline as SCoNE, so only the differing argument is shown.

```bash
# IRCAN (Shi et al., NeurIPS 2024) - attribution-only neuron selection
experiment_mode=ircan

# Context-aware decoding (CAD)
generator.init_args.use_cad=true
```

Fine-tuned baselines are swapped in through `generator=`:

```bash
generator=rankcot            # MignonMiyoung/RankCoT
generator=retrobust          # Ori/llama-2-13b-peft-nq-retrobust
generator=pa-rag             # wuqiong1/PA-RAG_Meta-Llama-3-8B-Instruct
```

`retrobust-qwen`, `retrobust-qwen-mix`, `retrobust-llama3` and `pa-rag-qwen` are also available
under `config/generator/`.

## Selectivity Analysis
```bash
python scripts/selectivity_probe.py --n 1000
python scripts/selectivity_table.py --data selectivity_results/actdata_signed_n1000.json
```

## Controlled-noise Experiment
```bash
# Vanilla RAG
bash scripts/run_distractor_noise.sh base '[]'

# SCoNE with the fixed neurons used across all noise conditions
bash scripts/run_distractor_noise.sh scone '[[27,8140],[13,2158],[21,12666],[30,3382],[30,5035]]'

python scripts/collect_noise_results.py --dir results/noise
```

## Evaluation

Each run writes to `experiments/<run_id>/`, where `eval_dev_metrics.json` holds the metric.

```bash
cat experiments/<run_id>/eval_dev_metrics.json
```

## Cite

If you use SCoNE, please cite our paper (citation coming soon).

<details>
<summary>This repository builds on BERGEN and includes the IRCAN baseline — please also cite them</summary>

```bibtex
@inproceedings{shi2024ircan,
      title={IRCAN: Mitigating Knowledge Conflicts in LLM Generation via Identifying and Reweighting Context-Aware Neurons},
      author={Dan Shi and Renren Jin and Tianhao Shen and Weilong Dong and Xinwei Wu and Deyi Xiong},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2024},
      url={https://github.com/danshi777/IRCAN},
}

@misc{rau2024bergenbenchmarkinglibraryretrievalaugmented,
      title={BERGEN: A Benchmarking Library for Retrieval-Augmented Generation},
      author={David Rau and Hervé Déjean and Nadezhda Chirkova and Thibault Formal and
      Shuai Wang and Vassilina Nikoulina and Stéphane Clinchant},
      year={2024},
      eprint={2407.01102},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.01102},
}
```
</details>

## License

Derivative of [NAVER's BERGEN](https://github.com/naver/bergen), released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); SCoNE's additions are
released under the same license. See [LICENCE.md](LICENCE.md) and [NOTICE.md](NOTICE.md).
