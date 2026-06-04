# Data-Driven Symbolic Equation Discovery of Multi-Robot Dynamics

This repository presents a physics-informed machine learning pipeline for the analysis of multi-robot trajectory data and the discovery of interpretable mathematical models describing their dynamics.

The workflow combines:
* Visualization and analysis
* Analytical modeling based on physical priors.
* Feature extraction
* Feature-based clustering of robots
* Trajectory preprocessing
* Hyperparameter optimization with **Optuna**
* Symbolic regression via **EPDE**,
* Physics-informed trajectory reconstruction **TEDEOUS**.

## Project Goal

The goal of this work is to develop a software framework for identifying accurate, compact, and interpretable systems of ordinary differential equations (ODEs) governing the motion of robots based on experimental trajectory data.

---
##  Data Description

* **Pickle file** contains raw experimental data extracted from video tracking.
* For each robot and timestep:
  * robot ID,
  * 2D coordinates `(x, y)`.
* Supported robot shapes:
  * circle
  * oval

---
## Components

`AllTrajectoriesFigure`
* Spatial Y(X) trajectories
* Temporal X(t), Y(t) trajectories
* Animation (for circular robots)
* Cluster visualization (spatial, temporal and pair plots)

`ColorManager`
* Unified color mapping for robots, particles, Fourier transform reconstructions and clusters.

`ParticleTrajectoryFitting`
* Estimates rotation via FFT
* Fits analytical particle-like model
* Parameter optimization via Differential Evolution

`FeatureCollection`

Extracts:
* Fourier features (dominant frequencies, amplitudes, reconstruction error),
* particle-model parameters,
* kinematic features (velocity, curvature),
* exploratory analysis (correlations, distributions).

`Clusterer`
* Feature scaling (StandardScaler)
* Clustering (KMeans)
* Cluster analysis

`DataProcessor`
* Extracts coordinates
* Normalizes trajectories (MinMax scaling)

`data_analysis.ipynb`
* Load experimental data
* Visualize motion
* Fit particle model
* Extract features
* Determine number of clusters (elbow method)
* Perform clustering

`discovery_science.ipynb`
* Splits trajectories into segments
* Runs EPDE for symbolic regression
* Uses Optuna for hyperparameter tuning
* Outputs differential equations

`system_analysis.ipynb`
* Parses discovered equations
* Computes statistics (mean, variance)
* Comparison of the opening time of equations at different levels

`solver.ipynb`
* Runs TEDEouS for reconstruction of trajectory
* Uses Optuna for hyperparameter tuning

---
## Hyperparameter optimization

For each trajectory segment of `n_parts` independently:

* **Optuna** optimizes EPDE parameters:

  * polynomial window,
  * smoothing sigma,
  * boundary,
  * population size.

* **Optuna** optimizes TEDEOUS parameters:

  * the number and frequency of Fourier transform embeddings,
  * the number of layers
  
* Objective balances:

  * equation residuals,
  * model complexity,
  * reconstruction stability.
---

## Repository Structure

```
.
├── data_analysis.ipynb
├── discovery_science_mezo.ipynb
├── system_analysis.ipynb
├── solver.ipynb
├── colors_manager.py
├── all_trajectories_figure.py
├── particle_trajectory_fitting.py
├── feature_collection.py
├── clusterer.py
├── data_process.py
├── ode_external_tokens.py
│   (additional force tokens for EPDE)
├── {circle|oval}/
│   ├── data/
│       └── {circle_data_00_330_[30_bots_PWM_10_15cw_15ccw_D_41cm].MP4.pickle | oval_data_[30_bots_PWM_1_exp_1].pickle}
│   ├── figures/
│       ├── pair_plot.pdf
│       ├── elbow_plot.pdf
│       ├── histograms.pdf
│       ├── heatmap.pdf
│       ├── X(t)_Y(t)_trajectories_of_cluster_{c}.html
│       ├── Y(X)_clusters_trajectories.html
│       ├── X(t)_Y(t)_trajectories_{ids}.html
│       └── Y(X)_trajectories_{ids}.html
│   ├── fit_all_particle_results.json 
│   ├── particle_coordinates.npz
│   ├── robots_features.csv
│   ├── levels_robots_ids.json
│   ├── EPDE_output_micro/
│       ├── robot_{id}/
│           └── {n}_parts/
│               └── {with|without}_force/
│                   ├── part_{nk}_system_best_params.json
│                   ├── part_{nk}_system_history_plot.html
│                   ├── part_{nk}_system_importances_plots.html
│                   ├── system_{nk}.csv
│                   └── part_{nk}_obj_func_and_equations.txt
│   └── EPDE_output_meso
│       ├── robot_{ids}
│           └── ...
├── requirements.txt
└── README.md
```
---
## Installation
```
git clone <repo>
cd <repo>

pip install -r requirements.txt
```
---

## Dependencies

Core libraries:
* numpy, pandas, scipy, matplotlib, seaborn

Visualization:
* matplotlib
* plotly

Machine Learning
* scikit-learn

Optimization & Discovery
* epde
* tedeous
* optuna

Utility
* dill (pickle)
* pathlib
* json

---
## Research Outcomes
* Developed a unified pipeline for data-driven discovery of dynamical systems
* Demonstrated partial recovery of analytical motion laws via EPDE
* Showed effectiveness of physics-informed priors at the micro-level
* Identified limitations of priors at the meso-level
* Established that cluster-based modeling provides improved stability of identified systems
* Constructed feature space incorporating FFT-based characteristics of trajectories

---

## Limitations and Assumptions

* Robot ID < 100
* Only `oval` and `circle` supported
* Particle model applies mainly to circular robots
* the model chosen was the motion of a charged particle in an electromagnetic field with mass m = 60 and charge q = 1.2
* derivatives up to second order
* EPDE and TEDEouS sensitive to noise and hyperparameters





