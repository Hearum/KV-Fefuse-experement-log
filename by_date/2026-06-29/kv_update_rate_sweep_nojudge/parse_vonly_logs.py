import re, pathlib
files={
'0.15':'MOTIVATION_EXPERIMENTS/reflect_pipeline_strict_v_only_clean/run.log',
'0.3':'MOTIVATION_EXPERIMENTS/kv_update_rate_sweep_nojudge/v_only_rate0p3.log',
'0.5':'MOTIVATION_EXPERIMENTS/kv_update_rate_sweep_nojudge/v_only_rate0p5.parallel.gpu1.log',
'0.8':'MOTIVATION_EXPERIMENTS/kv_update_rate_sweep_nojudge/v_only_rate0p8.parallel.gpu7.log',
}
for r,p in files.items():
    path=pathlib.Path(p)
    if not path.exists():
        print(r,'missing',p)
        continue
    txt=path.read_text(errors='ignore')
    counts=[int(x) for x in re.findall(r'prompt eval count:\s+(\d+) token', txt)]
    selects=re.findall(r'selected_doc_tokens[^\n]*', txt)
    print('\nrate',r,'file',p,'prompt_count_n',len(counts))
    if counts:
        print(' first20', counts[:20])
        print(' avg', sum(counts)/len(counts), 'min', min(counts), 'max', max(counts))
