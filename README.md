<img src="documentation/images/BERGEN.png" width="500">

# SCoNE: Selecting Context-aware Neurons for RAG

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

SCoNE identifies and reweights the neurons that a language model relies on to actually use retrieved context (as opposed to falling back on parametric memory), improving context-faithfulness in RAG. It is implemented on top of [NAVER's BERGEN](https://github.com/naver/bergen) benchmarking library, and includes an [IRCAN](https://github.com/danshi777/IRCAN) (Shi et al., NeurIPS 2024) baseline for comparison. This repository is a derivative of BERGEN and is released under the same CC BY-NC-SA 4.0 license (see [License](#license)).

## SCoNE Quick Start

```bash
CUDA_VISIBLE_DEVICES=0 python bergen.py \
    generator=qwen-25-7b-instruct \
    dataset=kilt_hotpotqa \
    retriever=splade-v3 \
    generator.init_args.use_attr=true \
    generator.init_args.attr_ds_name=hotpotqa \
    generator.init_args.num_attr_samples=100 \
    experiment_mode=scone \
    enhance_strength=7.0 \
    save_result=true \
    top_k=15
```

- `experiment_mode=ircan` runs the IRCAN baseline; `experiment_mode=scone` runs SCoNE (ours). Ablations are available as `scone_attronly`, `scone_varonly`, `scone_var`, `scone_std`, `scone_mad`, `scone_w1`/`w2`/`w5`/`w10`.
- `generator.init_args.use_attr=true` + `attr_ds_name=<name>` selects which HuggingFace dataset (see `DS_NAME_MAP` in `models/generators/generator.py`) is used to mine context-aware neurons; `num_attr_samples` controls how many samples are used.
- Baseline generator configs for CAD, RankCoT, Ret-Robust and PA-RAG are also included under `config/generator/` for comparison.

---

# BERGEN: A Benchmarking Library for Retrieval-Augmented Generation
 
[![arXiv](https://img.shields.io/badge/arXiv-2407.01102-b31b1b.svg)](https://arxiv.org/abs/2407.01102)
[![arXiv](https://img.shields.io/badge/arXiv-2407.01463-b31b1b.svg)](https://arxiv.org/abs/2407.01463)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

BERGEN (BEnchmarking Retrieval-augmented GENeration) is a library designed to benchmark RAG systems with a focus on question-answering (QA). It addresses the challenge of inconsistent benchmarking in comparing approaches and understanding the impact of each component in a RAG pipeline.

## Key Features

- Easy reproducibility and integration of new datasets and models
- Support for various retrievers (20+), rerankers(4) and large language models (20+)
- Flexible configuration system using YAML files
- Comprehensive evaluation metrics (*Match, EM, LLMEval*, ... )
- Support for multilingual experiments

![](documentation/images/teaser_bergen.jpg) 

For more information and experimental findings, please see:
- The initial BERGEN paper: https://arxiv.org/abs/2407.01102 and our [EMNLP'24 slides](documentation/BERGEN.pdf)
- The Multilingual RAG paper: https://arxiv.org/abs/2407.01463

## Quick Start

A typical RAG setup follows this pipeline:

`question` >> `retriever` >> `reranker` >> `LLM` >> `answer`

You can configure each component using simple YAML files. Here's an example of running an experiment:

```bash
python3 bergen.py retriever="bm25" reranker="minilm6" generator='tinyllama-chat' dataset='kilt_nq'
```

## Installation

Check the [installation guide](documentation/INSTALL.md) for detailed instructions.


## Usage

```
# simple setup for benchmarking
# run the retriever and cache results
# do the generation with VLLM
for dataset in kilt_nq kilt_hotpotqa kilt_triviaqa asqa popqa ; do
   
   python3 bergen.py  retriever=splade-v3 reranker=debertav3  dataset=$dataset
    
   python3 bergen.py  retriever=splade-v3 reranker=debertav3 dataset=$dataset  generator=vllm_SOLAR-107B
done
```


To fully configure BERGEN, please read our [configuration guide](documentation/config.md)

## Evaluation

Run the evaluation script to calculate LLMEval metrics and print the results:

```bash
python3 evaluate.py --experiments_folder experiments/ --llm_batch_size 16 --split 'dev' --llm vllm_SOLAR-107B

#parse all the experiments files into a panda dataframe
python print_results.py --folder experiments/ --format=tiny
```

Bergen also offers the possiblity to run pairwise comparisons using an LLM as judge. For more evaluation options and details, refer to the [Evaluation section](documentation/evaluations.md) in the complete documentation.

## RAG Baselines
Bergen provides results for several models and many datasets aiming to **provide strong baselines**. On the important datasets for RAG, the match metric is given by this table (see more in our paper): 
### Match Metric
 Model | ASQA | NQ | TriviaQA | POPQA | HotPotQA|
:----------:|:----------:|:----------:|:----------:|:----------:|:----------:
Llama-2-7B  | 68.4 | 61.6 | 87.9 | 60.2 |  45.9|
Llama-2-70B | 73.2 | 65.8 | 92.3 | 65.5  | 53.6|
Mistral-8x7B| 73.5 | 67.1 | 91.8 | 67.9 |  54.5|
Solar-10.7B   | 76.2 | 70.2 | 92.8 | 71.2 |  53.9|


## Multilingual Experiments

Refer to our [multilingual RAG guide](documentation/multilingual.md) for running experiments with multilingual user queries and/or multilingual Wikipedia as a datastore.


## Training

To train a model, add a training config:

```bash
python3 bergen.py retriever="bm25" reranker="minilm6" generator='tinyllama-chat' dataset='kilt_nq' train='lora'
```

## Extensions

To add new datasets and models, or configure prompts, see our [reference guide](/extensions.md).


## Cite

If you use SCoNE, please cite our paper (citation coming soon). This repository builds on BERGEN and includes the IRCAN baseline — if you use those, please also cite:

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

@misc{chirkova2024retrievalaugmentedgenerationmultilingualsettings,
      title={Retrieval-augmented generation in multilingual settings}, 
      author={Nadezhda Chirkova and David Rau and Hervé Déjean and Thibault Formal and Stéphane Clinchant and Vassilina Nikoulina},
      year={2024},
      eprint={2407.01463},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.01463}, 
}
```

## License

This repository is a derivative of [NAVER's BERGEN](https://github.com/naver/bergen), released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license, and SCoNE's additions are released under the same license. For more details, see the [LICENCE.md](LICENCE.md) and [NOTICE.md](NOTICE.md) files.

---
