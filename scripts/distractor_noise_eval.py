"""
Controlled-noise evaluation on HotpotQA: vary the number, position and nature of
distractors for the same questions, with the editing pipeline held IDENTICAL.

  gold        = the 2 HotpotQA supporting paragraphs (from supporting_facts.title)
  distractors = HotpotQA's OWN hard, topically-related paragraphs (the other 8)
  noise level N in {0, 2, 4, 8}:   context = 2 gold + N distractors

  N=0 -> gold only            N=8 -> full HotpotQA context (2 gold + 8 distractors)

The three axes:
  * number   : --n_distractors 0|2|4|8   (ONE value per run, so each runs separately)
  * position : --gold_pos shuffle|first|last  (where gold sits among the distractors)
  * nature   : HotpotQA's built-in hard distractors (semantically close to the gold)

Neurons are passed in (NOT re-mined) and up-weighted exactly like the editing pipeline
in models/generators/generator.py:  down_proj.weight[:, pos] *= enhance_strength .
Use --neurons "[]" for the untouched base model (the "vanilla RAG under noise" control).

The eval prompt / chat template / greedy decoding / Match metric match the main eval
(config/prompt/basic.yaml + modules.metrics), so numbers are comparable to the main
table. Same seed + same --n_examples => SAME questions across all noise levels.

Run one cell of the grid (needs GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/distractor_noise_eval.py \
      --neurons "[[27,8140],[13,2158],[21,12666],[30,3382],[30,5035]]" \
      --n_distractors 4 --gold_pos shuffle --n_examples 1000 --seed 0 \
      --out results/noise/scone_shuffle_n4.json

Full sweep: scripts/run_distractor_noise.sh ; aggregate: scripts/collect_noise_results.py
"""
import argparse
import json
import os
import random
import sys

import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from modules.metrics import normalize, match_single, em_single, f1_single


# ----------------------------- eval prompt (config/prompt/basic.yaml) -----------------------------
SYS = ("You are a helpful assistant. Your task is to extract relevant information from "
       "provided documents and to answer to questions as briefly as possible.")
SYS_NO_DOCS = "You are a helpful assistant. Answer the questions as briefly as possible."
HF_DATASET = ("hotpotqa/hotpot_qa", "distractor")


def norm_title(t):
    return t.strip().lower()


def split_gold_distractor(ex):
    """Return (gold_paras, distractor_paras) as [(title, sentences_list), ...]."""
    ctx = ex["context"]
    gold_titles = set(norm_title(t) for t in ex["supporting_facts"]["title"])
    gold, dist = [], []
    for title, sents in zip(ctx["title"], ctx["sentences"]):
        (gold if norm_title(title) in gold_titles else dist).append((title, sents))
    return gold, dist


def para_text(para):
    title, sents = para
    return f"{title} {''.join(sents)}".strip()


FULL_N_DISTRACTORS = 8   # standard HotpotQA distractor setting: 2 gold + 8 distractors = 10 paras


def build_context(ex, n_distractors, gold_pos, ex_rng, require_full=True):
    """Build the doc list for a given noise level. Returns list of paragraph strings, or None.

    require_full=True  -> keep ONLY the canonical '2 gold + 8 distractor' examples
    (>=2 gold and >=8 distractor), so N=0/2/4/8 all draw from the SAME fully-populated
    pool and N=8 is a genuine full context. Drops the ~0.85% short-context tail.
    """
    gold, dist = split_gold_distractor(ex)
    if len(gold) < 2:
        return None                       # need the 2 supporting paragraphs
    if require_full and len(dist) < FULL_N_DISTRACTORS:
        return None                       # not a canonical 2-gold + 8-distractor example
    gold = gold[:2]
    n = min(n_distractors, len(dist))
    chosen_dist = ex_rng.sample(dist, n) if n > 0 else []

    if gold_pos == "first":
        docs = gold + chosen_dist
    elif gold_pos == "last":
        docs = chosen_dist + gold
    else:                                 # "shuffle": vary gold position (reviewer: position axis)
        docs = gold + chosen_dist
        ex_rng.shuffle(docs)
    return [para_text(p) for p in docs]


def make_prompt(tok, docs, question, max_doc_len):
    """Return the chat-templated prompt STRING (batched-tokenized later, with left padding)."""
    if docs:
        parts = []
        for i, d in enumerate(docs):
            if max_doc_len:
                d = " ".join(d.split()[:max_doc_len])
            parts.append(f"Document {i+1}: {d}")
        user = f"Background:\n{chr(10).join(parts)}\n\nQuestion: {question}"
        system = SYS
    else:
        user = f"Question: {question}"
        system = SYS_NO_DOCS
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def load_neurons(args):
    if args.neuron_json:
        j = json.load(open(args.neuron_json))
        return [[d["layer"], d["neuron"]] for d in j["selected_neurons"]]
    return json.loads(args.neurons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--neurons", default="[]",
                    help='JSON list of [layer, pos]; "[]" = untouched base model')
    ap.add_argument("--neuron_json", default=None,
                    help="path to final_selection.json (overrides --neurons)")
    ap.add_argument("--enhance_strength", type=float, default=7.0)
    ap.add_argument("--n_distractors", type=int, required=True, choices=[0, 2, 4, 8],
                    help="noise level: number of distractor paragraphs added to the 2 gold ones")
    ap.add_argument("--gold_pos", default="shuffle", choices=["shuffle", "first", "last"])
    ap.add_argument("--require_full", dest="require_full", action="store_true", default=True,
                    help="(default) use ONLY canonical 2-gold + 8-distractor examples")
    ap.add_argument("--no_require_full", dest="require_full", action="store_false",
                    help="also allow short-context examples (<8 distractors); N=8 uses whatever is available")
    ap.add_argument("--split", default="validation", choices=["validation", "train"],
                    help="validation (default) avoids the train[:100] mining set")
    ap.add_argument("--skip_first", type=int, default=0,
                    help="skip first K examples (use 100 with --split train to exclude the mining set)")
    ap.add_argument("--n_examples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0,
                    help="fixes example sampling + distractor choice + position => SAME questions across noise levels")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_doc_len", type=int, default=None, help="truncate each doc to K words (None=no truncation)")
    ap.add_argument("--batch_size", type=int, default=16, help="generation batch size (left-padded)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    cns = load_neurons(args)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # left-pad so newly generated tokens align across the batch
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = True
    model.eval()

    # in-memory neuron up-weight (no checkpoint saved) -- identical to editing pipeline
    with torch.no_grad():
        for layer, pos in cns:
            model.model.layers[layer].mlp.down_proj.weight[:, pos] *= args.enhance_strength
    print(f"model={args.model_path}  edited {len(cns)} neurons (strength={args.enhance_strength})  "
          f"N_distractors={args.n_distractors}  gold_pos={args.gold_pos}")

    name, subset = HF_DATASET
    ds = datasets.load_dataset(name, subset)[args.split]

    # SAME question pool for every noise level: seed-fixed shuffle over the eligible indices,
    # then the per-example RNG below (seeded by index) makes distractor choice/position
    # deterministic *per example* so it is consistent whatever N is.
    eligible = list(range(args.skip_first, len(ds)))
    rng.shuffle(eligible)

    # 1) collect the fixed example set (prompt + gold) up to n_examples
    recs, n_skip = [], 0
    for idx in eligible:
        if len(recs) >= args.n_examples:
            break
        ex = ds[idx]
        ex_rng = random.Random((args.seed, idx))          # per-example, N-independent
        docs = build_context(ex, args.n_distractors, args.gold_pos, ex_rng, require_full=args.require_full)
        if docs is None:
            n_skip += 1
            continue
        recs.append({
            "q_id": ex["id"], "question": ex["question"], "answer": ex["answer"],
            "n_docs": len(docs),
            "prompt": make_prompt(tok, docs, ex["question"], args.max_doc_len),
        })

    # 2) batched greedy generation (left-padded). Sort by prompt length to minimise padding.
    order = sorted(range(len(recs)), key=lambda i: len(recs[i]["prompt"]))
    rows = [None] * len(recs)
    for b in range(0, len(order), args.batch_size):
        batch_idx = order[b:b + args.batch_size]
        prompts = [recs[i]["prompt"] for i in batch_idx]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                  do_sample=False, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]          # left-pad => same input width for all rows
        for j, i in enumerate(batch_idx):
            pred = tok.decode(gen[j], skip_special_tokens=True).strip()
            r = recs[i]; gold_ans = r["answer"]
            rows[i] = {
                "q_id": r["q_id"], "question": r["question"], "answer": gold_ans,
                "n_docs": r["n_docs"], "response": pred,
                "M": match_single(pred, gold_ans),
                "EM": em_single(pred, gold_ans),
                "F1": f1_single(pred, gold_ans)[0],   # f1_single returns (f1, precision, recall)
            }
        done = b + len(batch_idx)
        if done % (args.batch_size * 5) < args.batch_size:
            m = sum(rows[i]["M"] for i in order[:done]) / max(done, 1)
            print(f"...{done}/{len(recs)}  running M={m:.4f}")

    summary = {
        "model_path": args.model_path, "neurons": cns, "enhance_strength": args.enhance_strength,
        "n_distractors": args.n_distractors, "gold_pos": args.gold_pos, "split": args.split,
        "require_full": args.require_full,
        "skip_first": args.skip_first, "seed": args.seed, "n_examples": len(rows), "n_skipped": n_skip,
        "M": sum(r["M"] for r in rows) / len(rows) if rows else 0.0,
        "EM": sum(r["EM"] for r in rows) / len(rows) if rows else 0.0,
        "F1": sum(r["F1"] for r in rows) / len(rows) if rows else 0.0,
    }
    print(f"\n=== N={args.n_distractors}  n={summary['n_examples']} (skipped {n_skip})  "
          f"M={summary['M']:.4f}  EM={summary['EM']:.4f}  F1={summary['F1']:.4f} ===")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
