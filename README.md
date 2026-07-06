## Steps to replicate experiments in paper

1. Create and activate the environment:

```bash
conda create -n strat_usage_repro python=3.12.4 pip -y
conda activate strat_usage_repro
```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install nbconvert jupyter
```

The camera-ready artifact includes the linear Census, Amazon, and MovieLens
experiments plus the Census two-layer MLP extension. The MLP path requires
`torch`, which is included in `requirements.txt`.

3. Download MovieLens raw data:

```bash
mkdir -p dataset
curl -L https://files.grouplens.org/datasets/movielens/ml-10m.zip -o /tmp/ml-10m.zip
unzip -q /tmp/ml-10m.zip -d dataset
```

4. Prepare MovieLens dataset:

```bash
python movieLens_data_preparation.py
```

5. Run batch experiments:

```bash
python run_batch.py
```

`run_batch.py` uses the `CONFIG_PATHS` list near the top of the file. Edit that
list to select the configs you want to run. Generated result payloads are written
under `results/`; cached ERM or pretrained models are written under `cache/`.

### Camera-ready linear kappa experiments

Run the three kappa configs used for the camera-ready robustness figure:

```bash
python -c "from pathlib import Path; from run_batch import RESULTS_ROOT, run_config_file; run_config_file(Path('configs/census_good_kappa.yaml'), RESULTS_ROOT.resolve())"
python -c "from pathlib import Path; from run_batch import RESULTS_ROOT, run_config_file; run_config_file(Path('configs/amazon_good_kappa.yaml'), RESULTS_ROOT.resolve())"
python -c "from pathlib import Path; from run_batch import RESULTS_ROOT, run_config_file; run_config_file(Path('configs/movielens_good_kappa.yaml'), RESULTS_ROOT.resolve())"
```

### Census two-layer MLP experiments

The Census NN path uses `dataset: census_nn`. It keeps the same Census
preprocessing, train/test split, preference clustering, rankings, and probing
semantics as the linear Census experiments, but replaces each logistic learner
with a two-layer ReLU MLP. There is no validation split; hyperparameters are
fixed by the YAML configs for reproducibility.

Fast smoke test:

```bash
python test_census_nn_small.py
python -c "from pathlib import Path; from run_batch import RESULTS_ROOT, run_config_file; run_config_file(Path('configs/census_nn_smoke.yaml'), RESULTS_ROOT.resolve())"
```

Reviewer-scale bad and good outcomes:

```bash
python -c "from pathlib import Path; from run_batch import RESULTS_ROOT, run_config_file; run_config_file(Path('configs/census_nn_bad.yaml'), RESULTS_ROOT.resolve())"
python -c "from pathlib import Path; from run_batch import RESULTS_ROOT, run_config_file; run_config_file(Path('configs/census_nn_good.yaml'), RESULTS_ROOT.resolve())"
```

### Camera-ready figures

After the kappa and Census NN runs finish, generate deterministic camera-ready
PDFs:

```bash
python generate_camera_ready_figures.py
```

Outputs:

- `final_figs/good_kappa.pdf`
- `final_figs/census_nn_camera_ready_joint.pdf`

If any required result is missing, the script fails with the missing config path
so the corresponding experiment can be run first.

### Notebooks

Open and run all cells in:

- `census_experiments_clean.ipynb`
- `amazon_experiments_clean.ipynb`
- `movielens_experiments_clean.ipynb`
- `census_nn_experiments_clean.ipynb`

Generated directories such as `results/`, `cache/`, `final_figs/`, and
`__pycache__/` should not be committed.
