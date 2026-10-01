# Phylogeography of B3.13

Code and analysis files accompanying the manuscript *Dairy cattle movement networks govern the spread and persistence of H5N1 B3.13 in the United States*.

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
  - `host_only_corrected_1358taxa` — Full dataset (1,358 tips)
  - `host_only_1358_downsample_551c` — Downsampled to ~50% of cattle sequences
  - `host_only_1358_downsample_275c` — Downsampled to ~25% of cattle sequences
  - `host_only_1358_downsample_137c` — Downsampled to ~12.5% of cattle sequences
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

Simulation results (375,000 simulations per scenario) can be regenerated using the code and config provided. USAMMv3 dairy cattle movement networks are available from the original mansucript.

## Software

- [BEAST X v10.5.0](https://beast.community/)
- Python environment specified in `environment.yml`