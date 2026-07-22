import numpy as np, glob
p=glob.glob('MOTIVATION_EXPERIMENTS/query_recompute_overlap_32q_preprocess/details/*/preprocesskv_example000_10passages_scores.npz')[0]
d=np.load(p, allow_pickle=True)
print(p)
print(d.files)
for k in d.files:
    a=d[k]
    preview = a[:3] if a.ndim == 1 else ''
    print(k, a.shape, a.dtype, preview)
