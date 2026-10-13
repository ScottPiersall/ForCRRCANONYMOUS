# Redactions for anonymous review

This archive is the anonymized review release. Author-identifying strings were
replaced by placeholders (`<USER>`, `<CLUSTER_HOST>`, `<CLUSTER_HOME>`,
`<SLURM_ACCOUNT>`, `<PARTITION>`, "Anonymous Authors"). No scientific value,
score, prompt, dataset row, or analysis output was changed.

**Effect on integrity hashes.** 25 files changed. 4 of them are frozen study
artifacts (`model_provenance.json` and `preflight.json` for the RTS and Qwen3
studies). The only change in those four files is the cluster cache path in
`snapshot_path`; the model revision hashes themselves are unchanged. For each
changed file, the table gives the SHA-256 before and after redaction. A study
integrity manifest that lists one of these files will show the "before" hash;
compare against that column. `REDACTIONS.json` has the full 64-character hashes.



