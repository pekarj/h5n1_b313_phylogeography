# Phylogeography of B3.13

Code and analysis files accompanying the manuscript *Dairy cattle movement networks govern the spread and persistence of H5N1 B3.13 in the United States*.

## System requirements

### Software dependencies

- Python 3.13 (conda environment specified in `environment.yml`)
- BEAST X v10.5.0 with BEAGLE library
- Key Python packages: numpy, scipy, pandas, matplotlib, geopandas, baltic, dendropy, networkx, statsmodels, scikit-learn (full list in `environment.yml`)

### Operating systems

Tested on macOS 15 (Tahoe) and Ubuntu 22.04 LTS. Expected to work on any system with conda support.

### Hardware

No non-standard hardware required. BEAST analyses benefit from GPU acceleration via BEAGLE but can run on CPU. Simulation analyses were run on a multi-core server but can run on a standard desktop with longer runtimes.

## Installation

```bash
conda env create -f environment.yml
conda activate h5n1_phylogeography
```

Typical install time: 10–15 minutes on a standard desktop with broadband internet.

## Demo

### Simulations

To run a minimal demonstration of the simulation pipeline:

```bash
cd simulations
python run_simulations.py --config config.yaml --abc_results results/abc_accepted.csv --scenarios npi_day35 --num_networks 5 --num_sims 1 --seed 42
```

This runs the federal order baseline scenario with 5 networks (instead of 1,000) and 1 simulation per network per parameter set. Expected runtime: a few minutes on a standard desktop. Output: scenario summary statistics and per-simulation results in `outputs/scenarios/`.

To aggregate results:

```bash
python aggregate.py --input_dir outputs/scenarios --scenarios npi_day35
```

### Notebooks

Notebooks can be run in order (`fig1` through `fig5`, then `sup_*`). Each notebook expects pre-computed BEAST outputs (summary trees and log files) and simulation results. Pre-computed BEAST summary trees are provided in `beast/`.

### BEAST

BEAST XMLs require genome sequences to be inserted (accession numbers are listed as comments in each XML). Run with:

```bash
beast -beagle_auto <xml_file>
```

Typical runtime: days to weeks depending on dataset size and hardware.

## Instructions for use

### Reproducing simulation analyses

1. Obtain USAMMv3 dairy cattle movement networks (Sellman et al.)
2. Place network files in `simulations/data/networks/`
3. Run ABC calibration: `python abc_calibration.py --config config.yaml`
4. Run all scenarios: `python run_simulations.py --config config.yaml`
5. Aggregate results: `python aggregate.py --input_dir outputs/scenarios`

Full reproduction with 1,000 networks and all scenarios requires a multi-core server and approximately 1–2 weeks of compute time depending on available resources.

### Reproducing figures

Run notebooks in `notebooks/` using the conda environment. Notebooks read BEAST summary trees from `beast/` and simulation results from `simulations/results/`.

## Repository structure

### `beast/`

BEAST X input files and summary trees for all phylogenetic and phylogeographic analyses. Genome sequences have been removed from XMLs; each is replaced with a comment referencing the corresponding accession.

- **`phylogenetic/`** — Bayesian phylogenetic inference
- **`phylogeographic/`** — Discrete phylogeographic analyses with GLM:
  - `timehomogeneous_allCorrelates_bothMvmts` — Time-homogeneous with BSSVS, all candidate predictors
  - `timeinhomogeneous_allCorrelates_bothMvmts` — Time-inhomogeneous (weekly epochs) with BSSVS, all candidate predictors
  - `timeinhomogeneous_correlateSet2_weeklyEpochs_3era_3rateScalar` — Three-era parameterization with BSSVS per era
  - `timeinhomogeneous_correlateSet2enforced_weeklyEpochs_3era_3rateScalar` — Three-era parameterization with inclusion probabilities fixed at 1.0 (primary analysis)
- **`host/`** — Host transition analyses (FIT) with downsampling sensitivity:
  - `host_only_1358` — Full dataset (1,358 tips)
  - `host_only_1358_downsample_551c` — Downsampled to ~50% of cattle sequences
  - `host_only_1358_downsample_275c` — Downsampled to ~25% of cattle sequences
  - `host_only_1358_downsample_137c` — Downsampled to ~12.5% of cattle sequences
  - `host_only_1358*geographyFIT.empirical` — Location phylogeography analysis using the empirical trees from `host_only_1358`
- **`county/`** — County-level phylogeographic analysis 

### `notebooks/`

Jupyter notebooks generating the main and supplementary figures.

**Main figures:**
| Notebook | Figure |
|----------|--------|
| `fig1_host_phylogeny.ipynb` | Fig. 1 — Multi-host analysis |
| `fig2_phylogeography.ipynb` | Fig. 2 — State-level phylogeography |
| `fig3_persistence.ipynb` | Fig. 3 — Within-state persistence |
| `fig4_network_emergence.ipynb` | Fig. 4 — Network properties and outbreak emergence |
| `fig5_npi_counterfactuals.ipynb` | Fig. 5 — NPI counterfactual scenarios |

**Supplementary figures:**
| Notebook | Content |
|----------|---------|
| `sup_phylogeography_glm.ipynb` | GLM coefficient and rate scalar panels |
| `sup_phylogeography_transitions.ipynb` | Transition heatmaps and polygon-arrow maps by era |
| `sup_persistence_background.ipynb` | Persistence distributions and predictor collinearity |
| `sup_persistence_sensitivity.ipynb` | Leave-one-out and California subsampling sensitivity |
| `sup_simulation_county_tree.ipynb` | County-level tree |
| `sup_simulation_validation.ipynb` | Concordance, temporal ordering, start-date sensitivity |
| `sup_simulation_abc_posterior.ipynb` | ABC posterior scatter plots |
| `sup_epicentre_risk_map.ipynb` | All-county multi-state spread risk maps |
| `sup_policy_targeted_containment.ipynb` | Targeted containment heatmaps |
| `sup_policy_containment_perimeters.ipynb` | Containment perimeter maps |
| `sup_policy_network_perimeter.ipynb` | Network perimeter analysis |
| `sup_movement_network_maps.ipynb` | USAMMv3 network maps |
| `sup_tables.ipynb` | Supplementary tables |

### `simulations/`

County-level SIR epidemic simulation code calibrated via ABC-SMC.

- `sir_model.py` — Core SIR simulation engine with network-based transmission and NPI layers
- `abc_calibration.py` — ABC-SMC calibration
- `run_simulations.py` — Config-driven scenario script (NPI timing/strength sweeps, containment, alternative origins)
- `aggregate.py` — Aggregation of per-simulation transmission logs into summary statistics
- `config.yaml` — Scenario definitions, prior bounds, observed data, and model parameters
- `computed_network_perimeters.json` — Precomputed network-based containment perimeters
- `results/abc_accepted.csv` — 75 ABC-SMC accepted parameter sets used for downstream simulations

Simulation results (375,000 simulations per scenario) can be regenerated using the code and config provided. USAMMv3 dairy cattle movement networks are available from the original manuscript.

## Software

- [BEAST X v10.5.0](https://beast.community/)
- Python environment specified in `environment.yml`

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
