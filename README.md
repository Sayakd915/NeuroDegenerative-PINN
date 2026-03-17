# Neurodegenerative Disease Progression Forecasting using Physics-Informed Neural Networks

This repository provides a comprehensive suite of deep learning models designed to forecast the longitudinal progression of various neurodegenerative diseases. By integrating **Physics-Informed Neural Networks (PINNs)** and **Graph Neural Networks (GNNs)**, these models bridge the gap between purely data-driven sequence modeling and established deterministic biological and physical laws.

The framework supports applying physical constraints through Ordinary Differential Equations (ODEs) and Network Diffusion Heat Equations over structural brain connectomes, improving generalization on sparse, highly irregular clinical datasets. 

---

## Core Architectures and Physics Integration

The codebase explores multiple architectures tuned for specific diseases and biological kinetics:

### 1. Data-Driven Baseline (LSTM)
* **Model**: `BaselineLSTM`
* **Purpose**: A standard Recurrent Neural Network baseline used to demonstrate the limitations of purely data-driven interpolations on sparse clinical visits. 
* **Target**: Predicting Normalized Whole Brain Volume (`nWBV`) or UPDRS clinical scores.

### 2. Atrophy & Kinetic Physics-Informed Neural Networks (PINNs)
* **Models**: `AtrophyPINN`, `KineticPINN`
* **Purpose**: Overcomes data scarcity by penalizing the network when predictions violate simplified first-order kinetics.
* **Physics Loss**: 
  - **Atrophy** (e.g. Alzheimer's): Modeled as an exponential decay ODE: $\frac{dV}{dt} = -kV$
  - **Kinetics** (e.g. Parkinson's UPDRS): Modeled as linear growth: $\frac{dP}{dt} = kP$

### 3. Coupled CJD Biological PINN
* **Model**: `CoupledCJDPINN`
* **Purpose**: Designed specifically for Creutzfeldt-Jakob Disease (CJD) incorporating the **Nucleated Polymerization Model**.
* **Physics Loss**: Models the coupled kinetics of normal prior protein ($M$ / PrPC) and misfolded disease burden ($P$ / PrPSc):
  - $\frac{dM}{dt} = \lambda - \gamma M - k M P$
  - $\frac{dP}{dt} = k M P - \mu P$
* Highlights dynamic learning of the transmission rate constant ($k$) directly from patient demographics and genetics.

### 4. Hierarchical Graph PINN (PI-GNN)
* **Model**: `HeirarchicalGraphPINN`
* **Purpose**: A spatiotemporal graph neural network running on the brain's structural connectome representing Population Co-Atrophy.
* **Physics Loss**: Adapts the **Network Diffusion Heat Equation** over the graph Laplacian ($L$) to guide multi-regional predictions over time:
   $\frac{dV}{dt} = -k V + \alpha L V$

---

## Datasets

The repository natively supports multiple modalities of medical data:
1. **OASIS-2**: Longitudinal MRI statistics tracking dementia and Alzheimer's disease progression.
2. **AMP-PD**: Longitudinal clinical assessments mapping Parkinson's progression via UPDRS scales.
3. **TADPOLE / FreeSurfer**: Morphometric statistics used to extract regional brain volumes and build Graph structures using PyTorch Geometric. 
4. **Synthetic CJD Biomolecular Data**: Generable biomolecular tracking for Prion diseases leveraging randomized synthetic kinetics via `synthetic_datagen.py`.

---

## Repository Structure

```text
├── data/                            # Datasets and generated graph outputs (.csv, .pt)
├── env/                             # Python virtual environment (if utilized)
├── figures/                         # Output target for generated PDFs and visualizations
├── models/                          # Saved model state dicts (*.pth)
├── output/                          # Standard out logs for training runs
├── src/                             # Core library
│   ├── dataset.py                   # PyTorch Datasets for longitudinal tabular data handling and varied length padding
│   ├── graph_dataset.py             # PyTorch Geometric dataset for brain connectomes and Graph Laplacian creation
│   ├── model.py                     # All neural network architectures, ODE definitions, and Physics Loss definitions
│   ├── plot_graph_pinn.py           # Visualization for spatiotemporal trajectories, laplacian heatmaps, and harmonized losses
│   ├── plot_results.py              # Visualizations for tabular PINNs and baseline trajectory comparisons
│   └── synthetic_datagen.py         # ODE-driven nucleated polymerization data generator (CJD simulation)
├── train_baseline.py                # Training pipeline for the purely data-driven standard LSTM
├── train_cjd_pinn.py                # Coupled ODE training loop mapped to Prion disease
├── train_graph_pinn.py              # Connectome diffusion training pipeline targeting regional brain volumes
├── train_pinn.py                    # Independent PINN training targeting Alzheimer's (OASIS) and Parkinson's (AMP-PD)
└── requirements.txt                 # Project dependencies
```

---

## Installation

Ensure you have Python 3.9+ installed.

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd Neurodegenerative-PINN
   ```

2. Create a virtual environment (Optional but Recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Generating Synthetic Data
Before training the CJD models, you can generate synthetic biomolecular test cases:
```bash
python src/synthetic_datagen.py
```

### Training Models
Execute the dedicated pipeline files from the root of the repository to train and automatically plot evaluation trajectories:

**1. Baseline LSTM Pipeline:**
```bash
python train_baseline.py
```

**2. Standard PINN Pipeline (OASIS-2 & AMP-PD):**
```bash
python train_pinn.py
```

**3. Coupled Epidemic PINN Pipeline (CJD):**
```bash
python train_cjd_pinn.py
```

**4. Spatiotemporal Graph PINN Pipeline:**
```bash
python train_graph_pinn.py
```

Visualizations generated during the training phase (e.g. structural graph adjacencies, true vs predicted trajectories, and physics loss convergence profiles) will be saved natively as PDFs and PNGs in the root or `figures/` directory.

---

## Dependencies
- `torch` >= 2.0.0
- `torch_geometric` >= 2.3.0
- `pandas` >= 2.0.0
- `numpy` >= 1.24.0
- `scikit-learn` >= 1.2.0
- `matplotlib` >= 3.7.0
- `seaborn` >= 0.12.2

## 📝 License
This codebase is distributed under the terms of the included `LICENSE`.
