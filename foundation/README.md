# Foundation-model experiments

This lane is separate from the paper's dataset-by-dataset FITS benchmarks.
Corpus pretraining is a new experimental use of the architecture, not a
reproduction of the FITS paper or a claim of established transfer performance.

- [pretrain](pretrain/README.md): curated v1/v2/v3 loaders, resumable trainer,
  and experiment-parallel Aurora launch.
- [adapt](adapt/README.md): reserved downstream integration, with `FITS_eval`
  and `ts_chronicle_eval` benchmark directories. Adaptation is not implemented.

Architectures remain shared in `models/<MODEL>/`; pretraining adapters are
registered in `pretrain/models.py`. FITS is implemented. DIFFS, ANG, and ROUTER
must implement their architecture and adapter before the launcher accepts them.
No changes to the existing dataset-specific training objective are required.
