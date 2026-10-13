"""Post-hoc robustness of TR/FS to margin noise, with a random-ordering null.

Scale-normalized tolerance: for each curve, gaps are divided by the curve's
own range R_x = max_s M_x(s) - min_s M_x(s). This is invariant to any
positive affine map of the margins, so it compares protocols whose margin
scales differ.  TR_tau: (min_{t<j*} M(t) - M(j*)) / R_x > tau.
FS_tau: (min_P M - max_N M) / R_x > tau.  tau = 0 recovers the strict
definitions.  Null: margins permuted uniformly within each curve.
Usage: python -I tolerance_analysis.py <artifacts_dir>
"""
import sys, os, json, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid_robustness as g

A = sys.argv[1]; g.A = A
TAUS = [0.0, 0.01, 0.05, 0.10]
SRC = collections.OrderedDict([
    ('Qwen2.5-14B, immediate', 's3/raw_scores/qwen2_5_14b_instruct.parquet'),
    ('Llama-3.1-8B, immediate', 's3/raw_scores/llama3_1_8b_instruct.parquet'),
    ('Mistral-7B, immediate', 's3/raw_scores/mistral_7b_instruct_v0_3.parquet'),
    ('Gemma-2-9B, immediate', 's3/raw_scores/gemma_2_9b_it.parquet'),
    ('Qwen2.5-14B, RTS', 'reason_then_score_ablation/raw_rows.parquet'),
    ('Qwen3-30B-A3B, native', 'reasoning_anchor/raw_rows.parquet')])


def norm_gaps(m, y):
    L = len(m); js = next((s for s in range(L) if y[s] == 0), L)
    if not 1 <= js <= L - 1: return None
    R = max(m) - min(m)
    if R <= 0: return (-1.0, -1.0, js)
    tr = (min(m[:js]) - m[js]) / R
    P = [m[s] for s in range(L) if y[s] == 1]; N = [m[s] for s in range(L) if y[s] == 0]
    fs = (min(P) - max(N)) / R
    return (tr, fs, js)


def perm_null(curve, tau, key, R=200):
    m, y = curve; rng = random.Random(key); hits = 0
    for _ in range(R):
        mm = m[:]; rng.shuffle(mm); hits += norm_gaps(mm, y)[0] > tau
    return hits / R


def boot_ci(vals_by_fam, B=5000, seed=20261011):
    rng = random.Random(seed); out = []
    fams = list(vals_by_fam.values())
    for _ in range(B):
        tot = 0; n = 0
        for v in fams:
            for _ in v: tot += rng.choice(v); n += 1
        out.append(tot / n)
    out.sort(); return out[int(.025 * B)], out[int(.975 * B) - 1]


def main():
    Y, fam, _ = g.truth_table()
    res = collections.OrderedDict(); per_art = {}
    rng = random.Random(7)
    for name, p in SRC.items():
        cur = g.curves_from(os.path.join(A, p), Y)
        rows = {a: norm_gaps(m, y) for a, (m, y) in cur.items()}
        rows = {a: r for a, r in rows.items() if r is not None}
        per_art[name] = rows
        n = len(rows); entry = {'n_nontriv': n}
        for tau in TAUS:
            entry['TRR_%g' % tau] = sum(r[0] > tau for r in rows.values()) / n
            entry['FSRR_%g' % tau] = sum(r[1] > tau for r in rows.values()) / n
            # permutation null: 200 within-curve shuffles per artifact, seeded per
            # (source, artifact, tau) so every statistic reuses identical draws
            nulls = {a: perm_null(cur[a], tau, '%s|%s|%g' % (name, a, tau)) for a in rows}
            entry['nullTRR_%g' % tau] = sum(nulls.values()) / len(nulls)
            if tau in (0.0, 0.05):
                by = collections.defaultdict(list)
                for a, r in rows.items():
                    by[fam[a]].append((1.0 if r[0] > tau else 0.0) - nulls[a])
                est = sum(sum(v) for v in by.values()) / n
                entry['TRR_minus_null_%g' % tau] = [est] + list(boot_ci(by, seed=20261012))
        byfam = collections.defaultdict(list)
        for a, r in rows.items(): byfam[fam[a]].append(r)
        entry['family_TRR_0'] = {f: sum(r[0] > 0 for r in v) / len(v) for f, v in byfam.items()}
        entry['family_TRR_0.05'] = {f: sum(r[0] > 0.05 for r in v) / len(v) for f, v in byfam.items()}
        vb = collections.defaultdict(list)
        for a, r in rows.items(): vb[fam[a]].append(1.0 if r[0] > 0.05 else 0.0)
        entry['TRR_0.05_ci'] = boot_ci(vb)
        # raw (unnormalized) gaps for strictly recoverable boundaries, in logits
        raw = []
        for a in rows:
            m, y = cur[a]; js = rows[a][2]
            gap = min(m[:js]) - m[js]
            if gap > 0: raw.append(gap)
        raw.sort()
        entry['raw_gap_lt1_share'] = sum(x < 1 for x in raw) / len(raw)
        entry['raw_gap_median'] = raw[len(raw) // 2] if len(raw) % 2 else (raw[len(raw)//2 - 1] + raw[len(raw)//2]) / 2
        res[name] = entry
    # paired RTS - immediate (Qwen2.5) at tau = 0.05
    qi = per_art['Qwen2.5-14B, immediate']; qr = per_art['Qwen2.5-14B, RTS']
    common = [a for a in qi if a in qr]
    by = collections.defaultdict(list)
    for a in common: by[fam[a]].append((qr[a][0] > .05) - (qi[a][0] > .05))
    res['paired_RTS_minus_imm_tau0.05'] = {'diff': sum(sum(v) for v in by.values()) / len(common),
                                           'ci': boot_ci({k: [float(x) for x in v] for k, v in by.items()})}
    json.dump(res, sys.stdout, indent=1)


if __name__ == '__main__':
    main()
