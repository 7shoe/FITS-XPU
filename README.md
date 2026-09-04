# FITS: Modeling Time Series with 10k parameters (ICLR 2024 Spotlight)

This is the official implementation of FITS. Please run the scripts in scripts\FITS for results. Scripts without `_best` are for ablation study and grid search for parameters. Scripts with `_best` are for multiple run on the optimal parameters.

See updates here: [Update](#update)

## Aurora experiment-parallel benchmarks

Tiny models are run as independent experiments, one process per XPU tile. From
an interactive Aurora compute node, this command detects the 12 flat tiles and
runs up to 12 dataset/horizon/seed combinations concurrently:

```bash
bash scripts/launch_benchmark.sh FITS train \
  ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic
```

The same command with `test` evaluates completed checkpoints in parallel. The
pool uses no `mpiexec` or model-level DDP. It sets a distinct flat
`ZE_AFFINITY_MASK` in every child, binds four CPU cores per child, staggers
starts by two seconds, and defaults to zero DataLoader subprocesses per
experiment. Tune the operational concurrency without changing experiment
hyperparameters:

```bash
MODEL_MAX_PARALLEL=6 MODEL_LAUNCH_STAGGER_SECONDS=3 \
  bash scripts/launch_benchmark.sh FITS train ETTh1 ETTh2
```

Scheduler evidence is written below
`logs/FITS/aurora/benchmark_runs/<run-id>/`: `manifest.json`, `attempts.csv`,
`events.jsonl`, `summary.json`, per-attempt logs, XPU health snapshots, raw
`xpu-smi` telemetry, and a telemetry peak/counter summary. A Level Zero failure
quarantines its tile and retries the task once elsewhere. An unhandled signal
or critical health report stops new admissions and drains already-running
experiments.

Aurora `ze_peak` safety is primarily a teardown concern. On SIGINT/SIGTERM the
pool starts no new work and sends no force signals. FITS records the signal,
stops between batches, synchronizes submitted XPU work, releases references,
and exits through normal Python teardown. Do not use `qdel -W force`; allow at
least 60 seconds of scheduler kill grace. Monitoring is useful evidence but
cannot by itself prove that the next PBS prologue will pass.

See [the experiment-pool runbook](docs/aurora_experiment_pool.md) for resume,
failure, and tuning controls.

The full FITS reproduction (train, test, and five-seed summary) is available as
one resumable command:

```bash
bash scripts/FITS/aurora/run_full_reproduction.sh
```

## Aurora multi-tile training

`run_longExp_F.py` supports single-node distributed training with one XCCL/DDP
process per PyTorch-visible XPU tile. Aurora's default FLAT hierarchy exposes 12
tiles per node:

```bash
module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset ZE_AFFINITY_MASK
export OMP_NUM_THREADS=4
export MASTER_ADDR="$(hostname | cut -d. -f1).hsn.cm.aurora.alcf.anl.gov"
export MASTER_PORT=29500
export CPU_BIND="verbose,list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"

mpiexec --pmi=pmix --envall -n 12 --ppn 12 --cpu-bind=${CPU_BIND} \
  python -u run_longExp_F.py --distributed --is_training 1 ...
```

Distributed mode currently supports `FITS` and `Real_FITS`, is training-only,
and deliberately rejects test or prediction execution. Evaluate the saved
rank-zero checkpoint later with an ordinary `--is_training 0` invocation.

The runner's existing augmentation rules are applied to `--batch_size` first;
the resulting value is the DDP **global** batch and must be divisible by the
world size. The runner prints both global and per-rank values before training.
`--ddp_num_workers` controls loader workers per rank and defaults to zero to
avoid creating 120 workers on one node.

Do not set a job-wide `ZE_AFFINITY_MASK` for this mode. Rank placement comes
from `PALS_LOCAL_RANKID`, and ambiguous or incompatible launch configurations
fail before model construction.

For a single-XPU interactive-node smoke test, selected-result configurations,
and separate checkpoint evaluation, see
[scripts/FITS/aurora/README.md](scripts/FITS/aurora/README.md). The historical
author ablation scripts are preserved separately in
[`scripts/FITS/legacy_author_ablation_scripts/`](scripts/FITS/legacy_author_ablation_scripts/).

## Also see our exciting new work! 

Wanna see something beyond FITS? Check: 

"Beyond Trend and Periodicity: Guiding Time Series Forecasting with Textual Cues" [Paper](https://arxiv.org/abs/2405.13522) [Code](https://github.com/VEWOXIC/TGTSF) [Dataset](https://github.com/VEWOXIC/Weather-Captioned)

Check out our new time series forecasting pipeline: [Code](https://github.com/VEWOXIC/Universal-Cross-Modal-Time-Series-Forecasting-Pipeline). A DLinear-like framework that support multiple features such as multimodal dataset support, better model/dataset/task management, more readable data pipeline and torch-lightning capatability! 

## Update
- We add a notebook for interpretability. We analyze FITS on synthetic datasets to show its capability of modeling sinusodial waves. 
- We add a model Real_FITS which use two linear layer to simulate the complex multiplication. This model can achieve the same result of FITS. Real_FITS can be used on devices that do not support complex number calculation (e.g. RTX4090). 
- We add a onnx implementation of FITS with the architecture of Real_FITS. ONNX is an open format built to represent machine learning models. It can be directly deploy on embedded system devices such as STM32. As far as we know, there is compatability issue on Cube AI with onnx opset17. 
- **All the training scripts are updated!**
- Files for anomaly detection are uploaded! Please check the instruction [here](./AD/runAD.md)
- ⚠ **We find a long standing bug in our code which may affect a wide range of research work. Please check the Important Notice section for more information.** We have been actively fixing this bug and rerun all our experiments as well as the baseline models we compared with. 
- We have updated the final results of FITS in this repo. Also, the arxiv version of paper is updated. 
- The experiment scripts are updated and logs for FITS are updated. 
- FITS is accepted by **ICLR 2024 as Spotlight presentation**!!! We will update the new results in camera ready version.
- 2024-09-19 FITS reaches 100 GITHUB STARS!!! Thanks for the support and recognition of you guys! 🎉🎉🎉


## 🚨 Important Update: 2023-12-25 🎄

We've identified a significant bug in our code, originally found in Informer (AAAI 2021 Best Paper), thanks to [Luke Nicholas Darlow](https://lukedarlow.com/) from the University of Edinburgh. This issue has implications for a broad spectrum of research on time series forecasting, including but not limited to:

- PatchTST (ICLR 2023) - [Link to affected code](https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/data_provider/data_factory.py)
- ~~TimesNet (ICLR 2023) - [Link to affected code](https://github.com/thuml/Time-Series-Library/blob/main/data_provider/data_factory.py)~~ (Note: We later find TimesNet uses batch_size=1 during testing. Thus, it is not impacted by this issue.)
- DLinear (AAAI 2022 reported version) - [Link to affected code](https://github.com/cure-lab/LTSF-Linear/commit/6fe4c28ff36b4228792f2bbe513e807577e4a57e)
- Informer (AAAI 2021 Best Paper) - [Link to affected code](https://github.com/zhouhaoyi/Informer2020/blob/main/exp/exp_informer.py)
- Autoformer (NIPS 2021 reported version) - [Link to affected code](https://github.com/thuml/Autoformer/commit/d9100709b04e3e8361170794eba4f47b1afb217f)
- Fedformer (ICML 2022) - [Link to affected code](https://github.com/MAZiqing/FEDformer/blob/master/data_provider/data_factory.py)
- FiLM (ICLR 2023) - [Link to affected code](https://github.com/tianzhou2011/FiLM/blob/main/data_provider/data_factory.py)
- ~~iTransformer (ICLR 2024 score: 8886) - [Link to affected code](https://github.com/thuml/iTransformer/blob/main/data_provider/data_factory.py)~~ (Note: We later find iTransformer uses batch_size=1 during testing. Thus, it is not impacted by this issue.)

Efforts are underway to correct this bug, and we will update our Arxiv submission and this repository with the revised results. A bug fix method will also be released to assist the community in addressing this issue in their work.

### Description of the Bug:

The bug stems from an incorrect implementation in the data loader. Specifically, the test dataloader uses `drop_last=True`, which may exclude a significant portion of test data, particularly with large batch sizes, leading to unfair model comparisons.

### Solution:

To fix this issue in codebases using LSTF-Linear's architecture:

1. In [data_factory.py](./data_provider/data_factory.py) within the data_provider folder (usually on line 19), change:

    ```python
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    ```

    To:

    ```python
    if flag == 'test':
        shuffle_flag = False
        drop_last = False #True
        batch_size = args.batch_size
        freq = args.freq
    ```

2. In your experiment script (e.g., [./exp/exp_main.py](./exp/exp_main_F.py)), modify the following (around line 290):

    From:
    ```python
    preds = np.array(preds)
    trues = np.array(trues)
    inputx = np.array(inputx) # some times there is not this line, it does not matter
    ```

    To:
    ```python
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputx = np.concatenate(inputx, axis=0) # if there is not that line, ignore this
    ```

    If you do not do this, it will generate an error during testing because of the dimension 0 (batch_size) is not aligned. Maybe this is why everyone is dropping the last batch. But concatenate them on the 0 axis (batch_size) can solve this problem. 

3. Run the officially provided scripts!

## Result Update

The best result is in bold and the second best is in italic. The results are reported in terms of MSE. ~~This is still preliminary results for FITS. We are rerunning the parameter search, ablation study and multi-runs for the final results. The final results will be updated in the paper.~~ Following are our final results. We have reported these results in the ICLR final version.

| Model     | ETTh1-96  | ETTh1-192 | ETTh1-336 | ETTh1-720 | ETTh2-96  | ETTh2-192 | ETTh2-336 | ETTh2-720 | ETTm1-96  | ETTm1-192 | ETTm1-336 | ETTm1-720 | ETTm2-96  | ETTm2-192 | ETTm2-336 | ETTm2-720 |
| --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| PatchTST  | 0.385     | *0.413*   | *0.44*    | *0.456*   | *0.274*   | *0.338*   | *0.367*   | *0.391*   | **0.292** | **0.33**  | **0.365** | *0.419*   | *0.163* | *0.219*   | *0.276*   | *0.368*   |
| Dlinear   | 0.384     | 0.443     | 0.446     | 0.504     | 0.282     | 0.350     | 0.414     | 0.588     | *0.301*   | *0.335*   | 0.371     | 0.426     | 0.171     | 0.237     | 0.294     | 0.426     |
| FedFormer | *0.375* | 0.427     | 0.459     | 0.484     | 0.340     | 0.433     | 0.508     | 0.480     | 0.362     | 0.393     | 0.442     | 0.483     | 0.189     | 0.256     | 0.326     | 0.437     |
| TimesNet  | 0.384     | 0.436     | 0.491     | 0.521     | 0.340     | 0.402     | 0.452     | 0.462     | 0.338     | 0.374     | 0.410     | 0.478     | 0.187     | 0.249     | 0.321     | 0.408     |
| FITS      | **0.372** | **0.404** | **0.427** | **0.424** | **0.271** | **0.331** | **0.354** | **0.377** | 0.303     | 0.337     | *0.366*   | **0.415** | **0.162** | **0.216** | **0.268** | **0.348** |
| IMP       | 0.003     | 0.009     | 0.013     | 0.032     | 0.003     | 0.007     | 0.013     | 0.014     | -0.011    | -0.007    | -0.001    | 0.004     | 0.001     | 0.003     | 0.008     | 0.020     |

| Model     | Weather-96 | Weather-192 | Weather-336 | Weather-720 | Electricity-96 | Electricity-192 | Electricity-336 | Electricity-720 | Traffic-96 | Traffic-192 | Traffic-336 | Traffic-720 |
| --------- | ---------- | ----------- | ----------- | ----------- | -------------- | --------------- | --------------- | --------------- | ---------- | ----------- | ----------- | ----------- |
| PatchTST  | *0.151*    | *0.195*     | *0.249*     | *0.321*     | **0.129**      | **0.149**       | *0.166*         | 0.210           | **0.366**  | **0.388**   | **0.398**   | *0.457*     |
| Dlinear   | 0.174      | 0.217       | 0.262       | 0.332       | 0.140          | 0.153           | 0.169           | *0.204*       | 0.413      | 0.423       | 0.437       | 0.466       |
| Fedformer | 0.246      | 0.292       | 0.378       | 0.447       | 0.188          | 0.197           | 0.212           | 0.244           | 0.573      | 0.611       | 0.621       | 0.630       |
| TimesNet  | 0.172      | 0.219       | 0.280       | 0.365       | 0.168          | 0.184           | 0.198           | 0.220           | 0.593      | 0.617       | 0.629       | 0.640       |
| **FITS**  | **0.143**  | **0.186**   | **0.236**   | **0.307**   | *0.134*        | **0.149**       | **0.165**       | **0.203**       | *0.385*    | *0.397*     | *0.410*     | **0.448**   |
| IMP       | 0.008      | 0.009       | 0.013       | 0.014       | -0.005         | 0.000           | 0.001           | 0.001           | -0.019     | -0.009      | -0.012      | 0.009       |

## Analysis

The discovered bug predominantly impacts results on smaller datasets like ETTh1 and ETTh2. Interestingly, for other datasets, certain models, such as PatchTST on ETTm1, demonstrate enhanced performance. FITS still maintains its good enough and comparable-to-sota performance.

## Datasets

The experiment runners default to the shared FITS data directory:

```text
/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/FITS_data/
```

`run_longExp_F.py` stores FITS checkpoints on Lustre by default:

```text
/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/FITS_checkpoints/FITS/
```

FITS test arrays, plots, and mergeable metrics are stored separately under:

```text
/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/FITS_results/FITS/
```

The model-lane convention for FITS and future DIFFS, ANG, and ROUTER work is
documented in [docs/model_lanes.md](docs/model_lanes.md).

Download the four ETT datasets there:

```bash
fits_data_dir="/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/FITS_data"
mkdir -p "$fits_data_dir"

for name in ETTh1 ETTh2 ETTm1 ETTm2; do
  curl -fL --retry 3 \
    -o "$fits_data_dir/${name}.csv" \
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/${name}.csv"
done
```

Download the remaining FITS forecasting benchmarks:

```bash
curl -fL --retry 3 \
  -o "$fits_data_dir/weather.csv" \
  'https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv?download=true'

curl -fL --retry 3 \
  -o "$fits_data_dir/electricity.csv" \
  'https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/electricity/electricity.csv?download=true'

curl -fL --retry 3 \
  -o "$fits_data_dir/traffic.csv" \
  'https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/traffic/traffic.csv?download=true'
```

The FITS scripts use this directory by default. Override it for another data
location with `--root_path /path/to/data/`.

### Replication

- We have uploaded the training logs for community review. Additionally, we've provided logs for other baseline models. It's important to note that these logs were generated using their respective official codebases, not the versions in this repository.

- ~~We will update the training scripts of FITS very soon.~~
- We Have update the training scripts. 

- For fairness, we have conducted baseline runs using freshly cloned codebases with the original hyperparameters. (Note: Avoid using versions from this repository.) TimesNet, which is unaffected by this issue, was not re-run and is mentioned here only for reference.

- We encourage the community to apply the provided bug fix and re-conduct their experiments.

(A minor note: The only change we made in hyperparameters was reducing the learning rate for DLinear on ETTh2 from 0.05 to 0.005, resulting in improved outcomes.)

(A word of caution: Training PatchTST, particularly on datasets like traffic and electricity, can be extremely time-consuming.)

(We failed to reproduce the FiLM result since it takes over 40GB GPU memory and over **2 hour per epoch** on an A800. Further, the provided scripts seems to have flaws, i.e. the 'modes1' parameter is set to 1032 in ETTh1 instead of the '32' in others, the train_epoch is 1 in ETTh2 which may result in a downgraded performance. Thus, we exclude FiLM in the following analysis since we can not ensure a fair comparison.)

## 🚨 Another potential information leakage in previous AD works

In previous anomaly detection works, anomaly threshold is calculated based on the test_set, see affected code in [Anomaly Transformer](https://github.com/thuml/Anomaly-Transformer/blob/b0ee470c8012bff857fb600462aec6209c4a18d9/solver.py#L254). Such setting may violate the assumption that the test_set should be unavailable before deploying the model. Such method may cause information leakage and cherrypicked result on the test_set. 

As claimed in the paper, FITS directly uses the validation set for threshold selecting as indicated in [code](https://github.com/VEWOXIC/FITS/blob/3b64eaa6c66618013a0120bfb260485259a52cc4/AD/solver_recon.py#L244). 

However, we still compare FITS with the results reported in their original paper which may have potential information leakage. And we encourage the community to reevaluate the affected methods for further reference. XD


## Notice

- ~~**FITS benefits from large batch size. Our latest version uses batch size of 128.** Some results are not updated due to limited time.~~
- ~~Please run the scripts for ETT datasets with _fin.~~

## Acknowledgement

We thank ***Luke Darlow*** from the University of Edinburgh who find the bug.
