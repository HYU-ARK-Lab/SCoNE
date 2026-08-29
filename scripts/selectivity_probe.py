"""
Selectivity probe: mean activation of selected neurons under GG / GD / DD contexts.

Reproduces the activation data behind the per-neuron selectivity table.
Activations are SIGNED (raw values, no abs) and read at the final input position
before answer generation.

Conditions, holding the question fixed so only gold-vs-distractor varies:
  GG : 2 gold documents
  GD : 1 gold + 1 distractor
  DD : 2 distractor documents

Uses the HotpotQA train split, skipping the first SKIP examples (the mining set).

  python scripts/selectivity_probe.py --n 1000
  python scripts/selectivity_probe.py --n 5000
"""
import argparse, json, random
import torch, datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
SYS = ("Your task is to extract relevant information from provided documents and to "
       "answer to questions as briefly as possible")
IRCAN = [[27, 8140], [13, 2158], [14, 12683], [19, 10739], [30, 5035]]
SCONE = [[27, 8140], [13, 2158], [21, 12666], [30, 3382], [30, 5035]]
N_LAYERS, INTER, RANDOM_POOL_N = 32, 14336, 100


def norm(t):
    return t.strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="probe size")
    ap.add_argument("--skip", type=int, default=100, help="train examples used for mining, excluded")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: selectivity_results/actdata_signed_n{N}.json")
    args = ap.parse_args()

    out = args.out or f"selectivity_results/actdata_signed_n{args.n}.json"
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # RNG call order below is significant: the random pool is drawn before the
    # example shuffle, which is drawn before the per-example distractor sampling.
    rng = random.Random(args.seed)
    rand_pool = [[rng.randrange(N_LAYERS), rng.randrange(INTER)] for _ in range(RANDOM_POOL_N)]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0").eval()
    ds = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor")["train"]

    union = sorted({(l, n) for s in [IRCAN, SCONE, rand_pool] for l, n in s})
    cap = {}

    def mk(L):
        def h(m, a):
            cap[L] = a[0].detach()
        return h

    hooks = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk(L))
             for L in sorted({l for l, _ in union})]

    def fwd(docs, q):
        parts = [f"Document {j+1}: {t} {''.join(s)}" for j, (t, s) in enumerate(docs)]
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": f"Background: {chr(10).join(parts)}\nQuestion: {q}"}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(ids)
        return {f"{L}@{n}": cap[L][0, -1, n].item() for (L, n) in union}   # SIGNED

    eligible = list(range(args.skip, len(ds)))
    rng.shuffle(eligible)
    data, done = [], 0
    for idx in eligible:
        if done >= args.n:
            break
        e = ds[idx]
        gt = set(norm(t) for t in e["supporting_facts"]["title"])
        gold = [(t, s) for t, s in zip(e["context"]["title"], e["context"]["sentences"]) if norm(t) in gt]
        dist = [(t, s) for t, s in zip(e["context"]["title"], e["context"]["sentences"]) if norm(t) not in gt]
        if len(gold) < 2 or len(dist) < 2:
            continue
        d2 = rng.sample(dist, 2)
        conds = {"GG": [gold[0], gold[1]], "GD": [gold[0], d2[0]], "DD": [d2[0], d2[1]]}
        data.append({c: fwd(dd, e["question"]) for c, dd in conds.items()})
        done += 1
        if done % 50 == 0:
            print(f"...{done}")
    for h in hooks:
        h.remove()

    json.dump(data, open(out, "w"))
    print(f"saved signed actdata, n={len(data)} -> {out}")


if __name__ == "__main__":
    main()
