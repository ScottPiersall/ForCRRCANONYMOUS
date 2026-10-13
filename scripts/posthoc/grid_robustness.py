"""L = 5 grid-robustness analysis for FormalCRRC (post-hoc, no new inference).

Reads saved S3 margins, recomputes TRR, FSRR, TCE, accuracy and
Pr(TR=0 | E+) on the full L = 9 grid and on the coarser grid
s in {0,2,4,6,8} (L = 5), and bootstraps paired differences.
Usage: python -I grid_robustness.py <artifacts_dir>
"""
import sys, os, json, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minipq

A = sys.argv[1]
FINE = list(range(9))
COARSE = [0, 2, 4, 6, 8]          # fixed before any L = 5 result was computed


def truth_table():
    c = minipq.read_columns(os.path.join(A, 's3/test_dataset.parquet'),
                            ['artifact_id', 'strictness_index', 'formal_truth', 'family', 'true_first_fail_index'])
    Y = {}; fam = {}; jstar = {}
    for a, s, y, f, j in zip(c['artifact_id'], c['strictness_index'], c['formal_truth'], c['family'], c['true_first_fail_index']):
        Y[(a, s)] = int(y); fam[a] = f; jstar[a] = j
    return Y, fam, jstar


def curves_from(path, Y, extra_filter=None):
    cols = ['artifact_id', 'strictness_index', 'margin']
    if extra_filter: cols += [extra_filter[0]]
    c = minipq.read_columns(path, cols)
    M = collections.defaultdict(dict)
    for k in range(len(c['artifact_id'])):
        if extra_filter and c[extra_filter[0]][k] not in extra_filter[1]: continue
        a = c['artifact_id'][k]; s = c['strictness_index'][k]; m = c['margin'][k]
        if m is None: continue
        M[a][s] = m
    out = {}
    for a, d in M.items():
        if len(d) == 9:
            out[a] = ([d[s] for s in FINE], [Y[(a, s)] for s in FINE])
    return out


def metrics(m, y):
    L = len(m)
    yhat = [1 if v >= 0 else 0 for v in m]
    jhat = next((s for s in range(L) if yhat[s] == 0), L)
    js = next((s for s in range(L) if y[s] == 0), L)
    nontriv = 1 <= js <= L - 1
    tr = fs = None
    if nontriv:
        tr = m[js] < min(m[:js])
        P = [m[s] for s in range(L) if y[s] == 1]; N = [m[s] for s in range(L) if y[s] == 0]
        fs = min(P) > max(N)
    eplus = (yhat[L - 1] == 0 and yhat[0] == 1)
    return dict(tce=abs(jhat - js), acc=sum(a == b for a, b in zip(yhat, y)) / L,
                nontriv=nontriv, tr=tr, fs=fs, eplus=eplus)


def summarize(curves, idx):
    rows = {a: metrics([m[s] for s in idx], [y[s] for s in idx]) for a, (m, y) in curves.items()}
    nt = [r for r in rows.values() if r['nontriv']]
    ep = [r for r in nt if r['eplus']]
    return rows, dict(
        n=len(rows), n_nontriv=len(nt),
        trr=sum(r['tr'] for r in nt) / len(nt),
        fsrr=sum(r['fs'] for r in nt) / len(nt),
        unrec=sum(not r['tr'] for r in nt),
        tce=sum(r['tce'] for r in rows.values()) / len(rows),
        acc=sum(r['acc'] for r in rows.values()) / len(rows),
        p_tr0_eplus=(sum(not r['tr'] for r in ep) / len(ep)) if ep else None,
        n_eplus=len(ep))


def boot_diff(r9, r5, fam, key, B=5000, seed=20261010):
    """Paired, family-stratified artifact bootstrap of (L5 - L9) for a rate over nontrivial artifacts."""
    ids = [a for a in r9 if r9[a]['nontriv']]
    by = collections.defaultdict(list)
    for a in ids: by[fam[a]].append(a)
    rng = random.Random(seed); ds = []
    for _ in range(B):
        samp = [rng.choice(v) for v in by.values() for _ in v]
        ds.append(sum(r5[a][key] - r9[a][key] for a in samp) / len(samp))
    ds.sort(); return ds[int(0.025 * B)], ds[int(0.975 * B) - 1]


def main():
    Y, fam, jstar = truth_table()
    sources = collections.OrderedDict([
        ('Qwen2.5-14B, immediate', curves_from(os.path.join(A, 's3/raw_scores/qwen2_5_14b_instruct.parquet'), Y)),
        ('Llama-3.1-8B, immediate', curves_from(os.path.join(A, 's3/raw_scores/llama3_1_8b_instruct.parquet'), Y)),
        ('Mistral-7B, immediate', curves_from(os.path.join(A, 's3/raw_scores/mistral_7b_instruct_v0_3.parquet'), Y)),
        ('Gemma-2-9B, immediate', curves_from(os.path.join(A, 's3/raw_scores/gemma_2_9b_it.parquet'), Y)),
        ('Qwen2.5-14B, RTS', curves_from(os.path.join(A, 'reason_then_score_ablation/raw_rows.parquet'), Y)),
        ('Qwen3-30B-A3B, native', curves_from(os.path.join(A, 'reasoning_anchor/raw_rows.parquet'), Y)),
    ])
    out = {}
    for name, cur in sources.items():
        r9, s9 = summarize(cur, FINE); r5, s5 = summarize(cur, COARSE)
        ci_tr = boot_diff(r9, r5, fam, 'tr'); ci_fs = boot_diff(r9, r5, fam, 'fs')
        out[name] = dict(L9=s9, L5=s5, d_trr_ci=ci_tr, d_fsrr_ci=ci_fs,
                         coarse_nontriv_same=all(r9[a]['nontriv'] == r5[a]['nontriv'] for a in r9))
    json.dump(out, sys.stdout, indent=1)


if __name__ == '__main__':
    main()
