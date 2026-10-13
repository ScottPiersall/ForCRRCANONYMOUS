"""New analysis 2: does TR_x tell in advance which errors held-out calibration can fix?

S2 held-out test curves; the family-level (and global) intercepts were fitted on the
separate S2 calibration partition. For each nontrivial artifact we compute TR_x from the
raw margins (Theorem 1) and whether the learned cutoff places the boundary exactly
(TCE_x = 0 after correction). Usage: python -I s2_tr_split.py <artifacts_dir>
"""
import sys, os, json, math, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minipq
A = os.path.join(sys.argv[1], 's2')
P = json.load(open(os.path.join(A, 'calibration_parameters.json')))['parameters']
FILES = collections.OrderedDict([('Qwen/Qwen2.5-14B-Instruct', 'qwen2_5_14b_instruct'),
    ('meta-llama/Llama-3.1-8B-Instruct', 'llama3_1_8b_instruct'),
    ('mistralai/Mistral-7B-Instruct-v0.3', 'mistral_7b_instruct_v0_3'), ('google/gemma-2-9b-it', 'gemma_2_9b_it')])
L = 9
t = minipq.read_columns(os.path.join(A, 'test_dataset.parquet'), ['artifact_id', 'strictness_index', 'formal_truth', 'family', 'true_first_fail_index'])
Y = {(a, s): int(y) for a, s, y in zip(t['artifact_id'], t['strictness_index'], t['formal_truth'])}
FAM = dict(zip(t['artifact_id'], t['family'])); JS = dict(zip(t['artifact_id'], t['true_first_fail_index']))

def jhat(m, a=0.0):
    return next((s for s in range(L) if m[s] + a < 0), L)

def boot(groups, B=5000, seed=20261011):
    """family-stratified bootstrap of the difference in exact-fix share, TR=1 minus TR=0"""
    rng = random.Random(seed); out = []
    fams = list(groups.values())
    for _ in range(B):
        n1 = k1 = n0 = k0 = 0
        for v in fams:
            for _ in v:
                tr, fix = rng.choice(v)
                if tr: n1 += 1; k1 += fix
                else: n0 += 1; k0 += fix
        out.append(k1 / n1 - (k0 / n0 if n0 else 0))
    out.sort(); return out[int(.025 * B)], out[int(.975 * B) - 1]

res = collections.OrderedDict()
for mid, f in FILES.items():
    c = minipq.read_columns(os.path.join(A, 'raw_scores/test', f + '.parquet'), ['artifact_id', 'strictness_index', 'margin'])
    M = collections.defaultdict(dict)
    for a, s, m in zip(c['artifact_id'], c['strictness_index'], c['margin']): M[a][s] = m
    par = P[mid]; ag = par['global']['alpha']; af = {k: v['alpha'] for k, v in par['family'].items()}
    tce = {'raw': [], 'global': [], 'family': []}
    rows = []
    for a, d in M.items():
        m = [d[s] for s in range(L)]; js = JS[a]
        tce['raw'].append(abs(jhat(m) - js)); tce['global'].append(abs(jhat(m, ag) - js))
        tce['family'].append(abs(jhat(m, af[FAM[a]]) - js))
        if 1 <= js <= L - 1:
            tr = m[js] < min(m[:js])
            rows.append((a, tr, jhat(m, af[FAM[a]]) == js, jhat(m, ag) == js, jhat(m) == js))
    sig = lambda v: 1 / (1 + math.exp(-v))
    accs = []; briers = []; mvrs = []
    for a, d in M.items():
        p = [sig(d[s]) for s in range(L)]; y = [Y[(a, s)] for s in range(L)]
        accs.append(sum((d[s] >= 0) == (y[s] == 1) for s in range(L)) / L)
        briers.append(sum((p[s] - y[s]) ** 2 for s in range(L)) / L)
        mvrs.append(sum(p[s + 1] - p[s] > 0.05 for s in range(L - 1)) / (L - 1))
    e = {'n': len(M), 'accuracy': sum(accs) / len(M), 'brier': sum(briers) / len(M), 'mvr_0.05': sum(mvrs) / len(M), 'tce_raw': sum(tce['raw']) / len(M), 'tce_global': sum(tce['global']) / len(M), 'tce_family': sum(tce['family']) / len(M)}
    r1 = [r for r in rows if r[1]]; r0 = [r for r in rows if not r[1]]
    e.update(n_nontriv=len(rows), TRR=len(r1) / len(rows), n_TR1=len(r1), n_TR0=len(r0))
    for i, k in [(2, 'family'), (3, 'global'), (4, 'raw')]:
        e['fixed_%s_TR1' % k] = sum(r[i] for r in r1) / len(r1)
        e['fixed_%s_TR0' % k] = sum(r[i] for r in r0) / len(r0) if r0 else None
        e['n_fixed_%s_TR1' % k] = sum(r[i] for r in r1); e['n_fixed_%s_TR0' % k] = sum(r[i] for r in r0)
    g = collections.defaultdict(list)
    for a, tr, fx, _, _ in rows: g[FAM[a]].append((tr, fx))
    e['diff_family_ci'] = boot(g)
    # share of all exactly fixed boundaries that come from TR=1 artifacts
    res[par['short_name']] = e
json.dump(res, sys.stdout, indent=1)
