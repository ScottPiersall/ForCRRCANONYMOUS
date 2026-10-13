"""Analyses requested in review (post-hoc, saved scores only, no inference).

A. Level s=0 (always passed by construction): share of curves whose first
   failure is s=0, and TCE / accuracy / TRR / FSRR with and without s=0
   (S3 judges and protocols; S2 held-out test).
B. Held-out recalibration on S2: intercepts fitted on the S2 CALIBRATION
   partition by the paper's objective (root of sum[sigma(M+a) - Y] = 0),
   at four granularities (global, family, level, level x family), applied
   to S2 TEST.  Reports TCE and the share of nontrivial boundaries placed
   exactly, split by TR (computed from raw test margins).
C. Family-stratified artifact bootstrap (5,000) intervals for the S2 table,
   conditional on the fitted calibration intercepts.
Usage: python -I reviewer_analyses.py <artifacts_dir>
"""
import sys, os, json, math, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minipq

A = sys.argv[1]; L = 9
sig = lambda v: 1.0 / (1.0 + math.exp(-v)) if v > -700 else 0.0
JUDGES = collections.OrderedDict([('Qwen2.5-14B', 'qwen2_5_14b_instruct'), ('Llama-3.1-8B', 'llama3_1_8b_instruct'),
                                  ('Mistral-7B', 'mistral_7b_instruct_v0_3'), ('Gemma-2-9B', 'gemma_2_9b_it')])


def truth(path):
    c = minipq.read_columns(path, ['artifact_id', 'strictness_index', 'formal_truth', 'family'])
    Y = {}; F = {}
    for a, s, y, f in zip(c['artifact_id'], c['strictness_index'], c['formal_truth'], c['family']):
        Y[(a, s)] = int(y); F[a] = f
    return Y, F


def curves(path, Y):
    c = minipq.read_columns(path, ['artifact_id', 'strictness_index', 'margin'])
    M = collections.defaultdict(dict)
    for a, s, m in zip(c['artifact_id'], c['strictness_index'], c['margin']):
        if m is not None: M[a][s] = m
    return {a: ([d[s] for s in range(L)], [Y[(a, s)] for s in range(L)]) for a, d in M.items() if len(d) == L}


def jhat(m, levels, off=None):
    off = off or [0.0] * L
    return next((s for s in levels if m[s] + off[s] < 0), L)


def stats(cur, levels):
    """TCE, accuracy, TRR, FSRR on the given ordered level subset (first failure searched within it)."""
    lo = levels[0]; n = len(cur); tce = acc = j0 = 0; tr = []; fs = []
    for m, y in cur.values():
        js = sum(y)  # first failing level index in the full grid (z+1), or L
        jh = jhat(m, levels); tce += abs(jh - js); j0 += (jh == lo)
        acc += sum((m[s] >= 0) == (y[s] == 1) for s in levels) / len(levels)
        if lo < js <= levels[-1] and js > lo:
            if js - lo >= 1 and js <= levels[-1]:
                prefix = [m[t] for t in levels if t < js]
                if prefix and js in levels:
                    tr.append(m[js] < min(prefix))
                    P = [m[s] for s in levels if y[s] == 1]; N = [m[s] for s in levels if y[s] == 0]
                    fs.append(min(P) > max(N))
    return dict(n=n, first_fail_at_lowest=j0 / n, tce=tce / n, acc=acc / n, trr=sum(tr) / len(tr), fsrr=sum(fs) / len(fs), n_nontriv=len(tr))


def fit_alpha(pairs):
    """root of sum sigma(M + a) - Y = 0 by bisection (monotone in a)."""
    tgt = sum(y for _, y in pairs)
    if tgt == 0 or tgt == len(pairs): return None
    lo, hi = -500.0, 500.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if sum(sig(m + mid) for m, _ in pairs) < tgt: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def main():
    out = collections.OrderedDict()
    # ---------------- A. level s=0 (S3 and protocols)
    Y3, F3 = truth(os.path.join(A, 's3/test_dataset.parquet'))
    s3 = collections.OrderedDict([(k + ', immediate', 's3/raw_scores/%s.parquet' % v) for k, v in JUDGES.items()])
    s3['Qwen2.5-14B, RTS'] = 'reason_then_score_ablation/raw_rows.parquet'
    s3['Qwen3-30B-A3B, native'] = 'reasoning_anchor/raw_rows.parquet'
    A_res = collections.OrderedDict()
    for name, p in s3.items():
        cur = curves(os.path.join(A, p), Y3)
        full = stats(cur, list(range(L))); no0 = stats(cur, list(range(1, L)))
        byfam = collections.defaultdict(list)
        for a, (m, y) in cur.items(): byfam[F3[a]].append(jhat(m, list(range(L))) == 0)
        # TR=0 failures whose only offending earlier level is s=0
        only0 = n0 = 0
        for m, y in cur.values():
            js = sum(y)
            if 1 <= js <= L - 1 and not m[js] < min(m[:js]):
                n0 += 1
                only0 += (m[0] <= m[js]) and all(m[t] > m[js] for t in range(1, js))
        A_res[name] = dict(all_levels=full, levels_1_8=no0, first_fail_s0_by_family={f: sum(v) / len(v) for f, v in byfam.items()},
                           tr0_caused_only_by_s0=only0, tr0_total=n0)
    out['A_level0_S3'] = A_res
    # ---------------- B. S2 held-out recalibration at four granularities
    Yc, Fc = truth(os.path.join(A, 's2/calibration_dataset.parquet'))
    Yt, Ft = truth(os.path.join(A, 's2/test_dataset.parquet'))
    P = json.load(open(os.path.join(A, 's2/calibration_parameters.json')))['parameters']
    B_res = collections.OrderedDict(); C_res = collections.OrderedDict(); boot_rows = {}
    for jname, f in JUDGES.items():
        cal = curves(os.path.join(A, 's2/raw_scores/calibration/%s.parquet' % f), Yc)
        tst = curves(os.path.join(A, 's2/raw_scores/test/%s.parquet' % f), Yt)
        fams = sorted(set(Fc.values()))
        rows = lambda keep: [(m[s], y[s]) for a, (m, y) in cal.items() for s in range(L) if keep(a, s)]
        a_glob = fit_alpha(rows(lambda a, s: True))
        a_fam = {fm: fit_alpha(rows(lambda a, s, fm=fm: Fc[a] == fm)) for fm in fams}
        a_lvl = [fit_alpha(rows(lambda a, s, l=l: s == l)) for l in range(L)]
        a_lf = {(fm, l): fit_alpha(rows(lambda a, s, fm=fm, l=l: Fc[a] == fm and s == l)) for fm in fams for l in range(L)}
        # same objective on levels 1..8 only (for the s=0 sensitivity of Q1)
        a_fam18 = {fm: fit_alpha(rows(lambda a, s, fm=fm: Fc[a] == fm and s >= 1)) for fm in fams}
        mid = [k for k in P if P[k]['short_name'] == jname][0]
        chk = dict(global_alpha_matches=abs(a_glob - P[mid]['global']['alpha']) < 1e-6,
                   family_alpha_matches=all(abs(a_fam[fm] - P[mid]['family'][fm]['alpha']) < 1e-6 for fm in fams))
        offs = {
            'none': lambda a: [0.0] * L,
            'global': lambda a: [a_glob] * L,
            'family': lambda a: [a_fam[Ft[a]]] * L,
            'level': lambda a: [x if x is not None else (500.0 if l == 0 else 0.0) for l, x in enumerate(a_lvl)],
            'level_x_family': lambda a: [(a_lf[(Ft[a], l)] if a_lf[(Ft[a], l)] is not None else (500.0 if l == 0 else 0.0)) for l in range(L)],
        }
        e = collections.OrderedDict(check=chk, n_level_alphas_undefined=sum(x is None for x in a_lvl))
        per_art = {}
        for kname, off in offs.items():
            tce = 0; fix1 = fix0 = n1 = n0 = 0
            for a, (m, y) in tst.items():
                o = off(a); js = sum(y); jh = jhat(m, list(range(L)), o); tce += abs(jh - js)
                per_art.setdefault(a, {})[kname] = abs(jh - js)
                if 1 <= js <= L - 1:
                    trx = m[js] < min(m[:js])
                    if trx: n1 += 1; fix1 += jh == js
                    else: n0 += 1; fix0 += jh == js
            e[kname] = dict(tce=tce / len(tst), exact_TR1=fix1, n_TR1=n1, exact_TR0=fix0, n_TR0=n0)
        # levels 1..8 with family shift refitted on levels 1..8
        tce18 = tce18raw = 0
        for a, (m, y) in tst.items():
            js = sum(y)
            tce18 += abs(jhat(m, list(range(1, L)), [a_fam18[Ft[a]]] * L) - js)
            tce18raw += abs(jhat(m, list(range(1, L))) - js)
        e['levels_1_8'] = dict(tce_raw=tce18raw / len(tst), tce_family_refit=tce18 / len(tst), s2_test_stats=stats(tst, list(range(1, L))))
        e['all_levels_stats'] = stats(tst, list(range(L)))
        B_res[jname] = e
        # per-artifact values for the bootstrap
        rowsA = {}
        for a, (m, y) in tst.items():
            p = [sig(v) for v in m]
            js = sum(y)
            rowsA[a] = dict(fam=Ft[a], acc=sum((m[s] >= 0) == (y[s] == 1) for s in range(L)) / L,
                            brier=sum((p[s] - y[s]) ** 2 for s in range(L)) / L,
                            tce_raw=per_art[a]['none'], tce_fam=per_art[a]['family'], tce_glob=per_art[a]['global'],
                            mvr=sum(p[s + 1] - p[s] > 0.05 for s in range(L - 1)) / (L - 1),
                            tr=(m[js] < min(m[:js])) if 1 <= js <= L - 1 else None)
        boot_rows[jname] = rowsA
    out['B_S2_heldout_calibration'] = B_res
    # ---------------- C. bootstrap CIs (paired across judges: same resampled artifacts)
    ids = sorted(boot_rows['Qwen2.5-14B'])
    byfam = collections.defaultdict(list)
    for a in ids: byfam[boot_rows['Qwen2.5-14B'][a]['fam']].append(a)
    rng = random.Random(20261012); Bn = 5000
    keys = ['acc', 'brier', 'tce_raw', 'tce_fam', 'tce_glob', 'mvr', 'trr']
    def agg(samp, j, k):
        R = boot_rows[j]
        if k == 'trr':
            v = [R[a]['tr'] for a in samp if R[a]['tr'] is not None]; return sum(v) / len(v)
        return sum(R[a][k] for a in samp) / len(samp)
    draws = collections.defaultdict(list)
    for _ in range(Bn):
        samp = [rng.choice(v) for v in byfam.values() for _ in v]
        for j in JUDGES:
            for k in keys: draws[(j, k)].append(agg(samp, j, k))
        draws['qwen_acc_minus_center'].append(agg(samp, 'Qwen2.5-14B', 'acc') - 61 / 81)
        draws['qwen_minus_llama_tce_fam'].append(agg(samp, 'Qwen2.5-14B', 'tce_fam') - agg(samp, 'Llama-3.1-8B', 'tce_fam'))
        draws['gemma_tce_fam_minus_center'].append(agg(samp, 'Gemma-2-9B', 'tce_fam') - 20 / 9)
        draws['qwen_minus_gemma_brier'].append(agg(samp, 'Qwen2.5-14B', 'brier') - agg(samp, 'Gemma-2-9B', 'brier'))
    ci = lambda v: [sorted(v)[int(.025 * Bn)], sorted(v)[int(.975 * Bn) - 1]]
    for j in JUDGES:
        C_res[j] = {k: dict(est=agg(ids, j, k), ci=ci(draws[(j, k)])) for k in keys}
    for k in ['qwen_acc_minus_center', 'qwen_minus_llama_tce_fam', 'gemma_tce_fam_minus_center', 'qwen_minus_gemma_brier']:
        base = {'qwen_acc_minus_center': agg(ids, 'Qwen2.5-14B', 'acc') - 61 / 81,
                'qwen_minus_llama_tce_fam': agg(ids, 'Qwen2.5-14B', 'tce_fam') - agg(ids, 'Llama-3.1-8B', 'tce_fam'),
                'gemma_tce_fam_minus_center': agg(ids, 'Gemma-2-9B', 'tce_fam') - 20 / 9,
                'qwen_minus_gemma_brier': agg(ids, 'Qwen2.5-14B', 'brier') - agg(ids, 'Gemma-2-9B', 'brier')}[k]
        C_res[k] = dict(est=base, ci=ci(draws[k]))
    out['C_S2_bootstrap'] = C_res
    json.dump(out, sys.stdout, indent=1)


if __name__ == '__main__':
    main()
