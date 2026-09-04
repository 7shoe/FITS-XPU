# Downstream adaptation

Reserved for loading pretrained architecture weights and adapting to a specific
downstream dataset. Implementation is deferred; no adaptation command is enabled.

- `FITS_eval/`: the seven FITS forecasting benchmarks.
- `ts_chronicle_eval/`: the separate TS Chronicle evaluation protocol.

Future adaptation outputs belong in
`/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/FITS_checkpoints/adapt/<MODEL>/`.
Load the pretraining `config.json` together with `checkpoint.pth`. FITS weights
depend on context, horizon and cutoff; changing their dimensions requires an
explicit transfer rule, not a silent partial state load. Channel-shared weights
can be reused across different numbers of channels. Keep adaptation checkpoints
separate from the pretrained weights and dataset-from-scratch benchmarks.
