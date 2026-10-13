"""MVR sensitivity to the tolerance delta (post-hoc, saved margins only).

MVR_x(delta) = (1/(L-1)) * #{s : sigma(M_x(s+1)) - sigma(M_x(s)) > delta},
averaged over all complete curves. Usage: python -I mvr_sensitivity.py <artifacts_dir>
"""
import sys, os, json, math, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid_robustness as g
import minipq

A = sys.argv[1]; g.A = A
DELTAS = [0.0, 0.01, 0.02, 0.05, 0.10]
SRC = collections.OrderedDict([
    ('Qwen2.5-14B, immediate', 'day3/raw_scores/qwen2_5_14b_instruct.parquet'),
    ('Llama-3.1-8B, immediate', 'day3/raw_scores/llama3_1_8b_instruct.parquet'),
    ('Mistral-7B, immediate', 'day3/raw_scores/mistral_7b_instruct_v0_3.parquet'),
    ('Gemma-2-9B, immediate', 'day3/raw_scores/gemma_2_9b_it.parquet'),
    ('Qwen2.5-14B, RTS', 'reason_then_score_ablation/raw_rows.parquet'),
    ('Qwen3-30B-A3B, native', 'reasoning_anchor/raw_rows.parquet')])
SAVED = {'Qwen2.5-14B, RTS': ('reason_then_score_ablation/metrics_by_artifact.parquet', 'mvr_zero_r', 'mvr_practical_r'),
         'Qwen3-30B-A3B, native': ('reasoning_anchor/metrics_by_artifact.parquet', 'mvr_zero_tolerance', 'mvr_practical')}
sig = lambda m: 1 / (1 + math.exp(-m))


def mvr(m, d):
    p = [sig(v) for v in m]
    return sum(p[s + 1] - p[s] > d for s in range(len(p) - 1)) / (len(p) - 1)


def main():
    Y, fam, _ = g.truth_table(); out = collections.OrderedDict()
    for name, path in SRC.items():
        cur = g.curves_from(os.path.join(A, path), Y)
        e = {'n': len(cur)}
        for d in DELTAS:
            e['MVR_%g' % d] = sum(mvr(m, d) for m, _ in cur.values()) / len(cur)
            by = collections.defaultdict(list)
            for a, (m, _) in cur.items(): by[fam[a]].append(mvr(m, d))
            e['family_MVR_%g' % d] = {f: sum(v) / len(v) for f, v in by.items()}
        # largest single upward step per curve, on the probability scale
        ups = sorted(max(sig(m[s + 1]) - sig(m[s]) for s in range(len(m) - 1)) for m, _ in cur.values())
        e['share_curves_with_any_rise_gt_0.05'] = sum(u > .05 for u in ups) / len(ups)
        if name in SAVED:
            p, z, pr = SAVED[name]; c = minipq.read_columns(os.path.join(A, p), [z, pr])
            e['saved_MVR_0'] = sum(c[z]) / len(c[z]); e['saved_MVR_0.05'] = sum(c[pr]) / len(c[pr])
        out[name] = e
    json.dump(out, sys.stdout, indent=1)


if __name__ == '__main__':
    main()
