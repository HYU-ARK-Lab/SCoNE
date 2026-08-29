"""
Per-neuron selectivity table from probe activations.

Reports, for every neuron selected by SCoNE (and by IRCAN, for comparison), the
mean SIGNED activation under GG / GD / DD, the gold-vs-distractor gap, and a
paired Wilcoxon test over examples.

  python scripts/selectivity_table.py --data selectivity_results/actdata_signed_n1000.json
"""
import argparse, json
import statistics as st
from scipy.stats import wilcoxon

IRCAN = [[27, 8140], [13, 2158], [14, 12683], [19, 10739], [30, 5035]]
SCONE = [[27, 8140], [13, 2158], [21, 12666], [30, 3382], [30, 5035]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="selectivity_results/actdata_signed_n1000.json")
    ap.add_argument("--sets", default="scone", choices=["scone", "all"],
                    help="'scone' lists the SCoNE neurons only; 'all' adds IRCAN's")
    args = ap.parse_args()

    data = json.load(open(args.data))
    tag = {}
    for x in IRCAN:
        tag[tuple(x)] = tag.get(tuple(x), "") + "I"
    for x in SCONE:
        tag[tuple(x)] = tag.get(tuple(x), "") + "S"
    keys = sorted(tag) if args.sets == "all" else sorted(tuple(x) for x in SCONE)

    print(f"n={len(data)}  (mean signed activation, final input position)\n")
    print(f"{'neuron':>9} {'set':4} {'GG':>8} {'GD':>8} {'DD':>8} {'|GG-DD|':>8} {'GD between':>11} {'p(GGvsDD)':>10}")
    for (L, nn) in keys:
        k = f"{L}@{nn}"
        gg = [r["GG"][k] for r in data]
        gd = [r["GD"][k] for r in data]
        dd = [r["DD"][k] for r in data]
        mg, md, mdd = st.mean(gg), st.mean(gd), st.mean(dd)
        between = min(mg, mdd) < md < max(mg, mdd)
        _, p = wilcoxon(gg, dd)
        print(f"{k:>9} {tag[(L, nn)]:4} {mg:8.3f} {md:8.3f} {mdd:8.3f} "
              f"{abs(mg - mdd):8.3f} {'yes' if between else 'no':>11} {p:10.1e}")
    print("\n(set: S = selected by SCoNE, I = by IRCAN, IS = by both)")
    print("'GD between' = the mixed condition falls between the gold-only and distractor-only means")


if __name__ == "__main__":
    main()
