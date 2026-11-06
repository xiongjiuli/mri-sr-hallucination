# Model weights

The trained checkpoints (`rrdb_l1.pt`, `rrdb_l1_mkda.pt`, `rrdb_dsc_mkda.pt`,
`rrdb_perc.pt`, `rrdb_dsc_mkda_unc.pt`) are **not** committed (about 12 MB
each). They can be regenerated with:

```bash
bash scripts/train_all.sh
```

All results in the paper can also be inspected without the weights via the
per-slice metrics and COP JSON files in `results/`.
