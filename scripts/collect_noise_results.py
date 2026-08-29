"""
Collect the distractor-noise runs into per-position tables and sanity-check them.

Reads <dir>/{scone,base}_{first,shuffle,last}_n{2,4,8}.json (+ {tag}_n0.json) as written
by scripts/run_distractor_noise.sh, and prints:
  * 3 tables (first / shuffle / last): rows = base / SCoNE / gap, cols = N in {0,2,4,8}
  * verification that every run used the SAME questions (paired), same require_full/seed

Usage:  python scripts/collect_noise_results.py --dir results/noise
"""
import argparse
import json
import os

POS = ["first", "shuffle", "last"]
NS = [0, 2, 4, 8]
TAGS = ["base", "scone"]


def path(directory, tag, pos, n):
    # N=0 is position-independent -> single file per tag
    return os.path.join(directory, f"{tag}_n0.json" if n == 0 else f"{tag}_{pos}_n{n}.json")


def load(directory, tag, pos, n):
    p = path(directory, tag, pos, n)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def qids(d):
    return [r["q_id"] for r in d["rows"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/noise", help="directory holding the run JSONs")
    args = ap.parse_args()

    runs = {}          # (tag,pos,n) -> data
    missing = []
    for tag in TAGS:
        for pos in POS:
            for n in NS:
                d = load(args.dir, tag, pos, n)
                if d is None:
                    missing.append(path(args.dir, tag, pos, n))
                else:
                    runs[(tag, pos, n)] = d

    if missing:
        print("[MISSING] runs not found yet:")
        for m in sorted(set(missing)):
            print("   ", m)
        print()

    # ---- verification ----
    print("=== checks ===")
    ref_q = None
    ok_paired = True
    ns, seeds, rf = set(), set(), set()
    for k, d in runs.items():
        s = d["summary"]
        ns.add(s["n_examples"]); seeds.add(s["seed"]); rf.add(s.get("require_full"))
        q = qids(d)
        if ref_q is None:
            ref_q = q
        elif q != ref_q:
            ok_paired = False
            print(f"  [WARN] {k} question set differs from the reference (len={len(q)})")
    print(f"  n_examples: {ns}   seed: {seeds}   require_full: {rf}")
    print(f"  all runs share the same questions (paired): {ok_paired}")
    print(f"  cells loaded: {len(runs)}/{len(TAGS) * len(POS) * len(NS)} "
          f"(N=0 is reused across positions, so only {len(TAGS) * (len(POS) * (len(NS) - 1) + 1)} files exist)\n")

    # ---- tables ----
    def M(tag, pos, n):
        d = runs.get((tag, pos, n))
        return d["summary"]["M"] * 100 if d else None

    def fmt(x):
        return f"{x:5.2f}" if x is not None else "  -- "

    for pos in POS:
        print(f"=== gold_pos = {pos}   (M, %) ===")
        print(f"  {'N':<8}" + "".join(f"{n:>8}" for n in NS))
        for tag in TAGS:
            print(f"  {tag:<8}" + "".join(f"{fmt(M(tag,pos,n)):>8}" for n in NS))
        gap = []
        for n in NS:
            b, s = M("base", pos, n), M("scone", pos, n)
            gap.append(s - b if (b is not None and s is not None) else None)
        print(f"  {'gap':<8}" + "".join(f"{fmt(g):>8}" for g in gap))
        print()

    print("Note: at N=0 the gold position is meaningless -> {base,scone}_n0.json is shown in all three tables.")


if __name__ == "__main__":
    main()
