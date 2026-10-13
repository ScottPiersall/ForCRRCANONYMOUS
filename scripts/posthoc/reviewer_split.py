"""Supplement: per-level held-out calibration on S2, split by the cause of TR=0
(only level s=0 offends vs an interior inversion), and the same analysis on
levels 1..8 only.  Usage: python -I reviewer_split.py <artifacts_dir>"""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reviewer_analyses as R
A = sys.argv[1]; L = 9
Yc, Fc = R.truth(os.path.join(A, 's2/calibration_dataset.parquet'))
Yt, Ft = R.truth(os.path.join(A, 's2/test_dataset.parquet'))
out = collections.OrderedDict(); tot = collections.Counter()
for j, f in R.JUDGES.items():
    cal = R.curves(os.path.join(A, 's2/raw_scores/calibration/%s.parquet' % f), Yc)
    tst = R.curves(os.path.join(A, 's2/raw_scores/test/%s.parquet' % f), Yt)
    fams = sorted(set(Fc.values()))
    rows = lambda keep: [(m[s], y[s]) for a, (m, y) in cal.items() for s in range(L) if keep(a, s)]
    lf = {(fm, l): R.fit_alpha(rows(lambda a, s, fm=fm, l=l: Fc[a] == fm and s == l)) for fm in fams for l in range(L)}
    off = lambda a: [(lf[(Ft[a], l)] if lf[(Ft[a], l)] is not None else 500.0) for l in range(L)]
    e = collections.Counter()
    for a, (m, y) in tst.items():
        js = sum(y)
        if not 1 <= js <= L - 1: continue
        tr = m[js] < min(m[:js]); only0 = (not tr) and m[0] <= m[js] and all(m[t] > m[js] for t in range(1, js))
        fixed = R.jhat(m, list(range(L)), off(a)) == js
        k = 'TR1' if tr else ('TR0_only_s0' if only0 else 'TR0_interior')
        e['n_' + k] += 1; e['fixed_' + k] += fixed
        # levels 1..8: TR on the 1..8 grid (nontrivial if js>=2), per-level x family offsets for levels 1..8
        if js >= 2:
            tr8 = m[js] < min(m[1:js]); fixed8 = R.jhat(m, list(range(1, L)), off(a)) == js
            k8 = 'L18_TR1' if tr8 else 'L18_TR0'
            e['n_' + k8] += 1; e['fixed_' + k8] += fixed8
    out[j] = dict(e); tot.update(e)
out['pooled'] = dict(tot)
json.dump(out, sys.stdout, indent=1)
