"""
OOD side-effect eval for neuron-upweighted models, one model load.
Runs (standard lm-evaluation-harness scoring, each task's default split):
  - HellaSwag       (5-shot, validation split)   # HS test has no gold labels
  - ARC-Challenge   (5-shot, test split)
  - MemoTrap        (0-shot, inverse_scaling_memo_trap, its only split)

Neuron edit replicates generator:  down_proj.weight[:, pos] *= enhance_strength

Env with lm_eval:
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 /opt/conda/envs/bergen_lmeval/bin/python ood_eval.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --neurons "[[27,8140],[13,2158],[21,12666],[30,3382],[30,5035]]" \
    --output_path results/llama_scone_ood.json
Use --neurons "[]" for the untouched base model.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

FIVE_SHOT = ["hellaswag", "arc_challenge"]
ZERO_SHOT = ["inverse_scaling_memo_trap"]
ORDER = ["hellaswag", "arc_challenge", "inverse_scaling_memo_trap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--neurons", required=True,
                    help='JSON list of [layer, pos] pairs; "[]" = base')
    ap.add_argument("--enhance_strength", type=float, default=7.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="None = full split")
    ap.add_argument("--output_path", default=None)
    args = ap.parse_args()

    cns = json.loads(args.neurons)
    enc = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto")
    model.config.use_cache = True
    model.eval()

    with torch.no_grad():
        for layer, pos in cns:
            model.model.layers[layer].mlp.down_proj.weight[:, pos] *= args.enhance_strength
    print(f"Applied {len(cns)} neurons x{args.enhance_strength}: {cns}")

    lm = HFLM(pretrained=model, tokenizer=enc, batch_size=args.batch_size)
    res = {}
    res.update(simple_evaluate(model=lm, tasks=FIVE_SHOT, num_fewshot=5, limit=args.limit)["results"])
    res.update(simple_evaluate(model=lm, tasks=ZERO_SHOT, num_fewshot=0, limit=args.limit)["results"])

    print("\n===== OOD RESULTS  (acc / acc_norm) =====")
    for t in ORDER:
        r = res[t]
        print(f"{t:28} acc={r.get('acc,none'):.4f}  acc_norm={r.get('acc_norm,none'):.4f}")

    if args.output_path:
        d = os.path.dirname(args.output_path)
        if d:
            os.makedirs(d, exist_ok=True)
        slim = {t: {k: v for k, v in res[t].items() if "stderr" not in k} for t in ORDER}
        with open(args.output_path, "w") as f:
            json.dump({"model_path": args.model_path, "neurons": cns,
                       "enhance_strength": args.enhance_strength,
                       "fewshot": {"hellaswag": 5, "arc_challenge": 5,
                                   "inverse_scaling_memo_trap": 0},
                       "results": slim}, f, indent=2)
        print(f"\nSaved -> {args.output_path}")


if __name__ == "__main__":
    main()
