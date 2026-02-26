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

5. Run batch experiments (one bad config per dataset, as configured in `run_batch.py`):

```bash
python run_batch.py
```

6. Open and run all cells in:

- `census_experiments_clean.ipynb`
- `amazon_experiments_clean.ipynb`
- `movielens_experiments_clean.ipynb`
