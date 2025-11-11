# RF Inversion Pipeline

A comprehensive machine learning pipeline for geophysical inversion using Random Forest to predict subsurface density and susceptibility models from gravity and magnetic data.

## 📋 Overview

This pipeline implements a four-stage workflow for geophysical inversion:

1. **Synthetic Data Generation** - Create training datasets using forward modeling
2. **Training** - Build Random Forest classifier from synthetic training data
3. **Testing** - Validate model performance on test datasets  
4. **Deployment** - Apply trained model to field data for prediction

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Required packages (install via pip):
  ```bash
  pip install numpy scipy scikit-learn imbalanced-learn faiss-cpu joblib pyyaml tqdm matplotlib pandas
  ```

### Installation & Setup

1. **Download the project files** and ensure all Python scripts are in the same directory

2. **Run the pipeline** (recommended for first-time users):
   ```bash
   python run_rf_pipeline.py
   ```

The pipeline will automatically:
- ✅ Create necessary directories (`constants/`, `training_data/`, `output/`, etc.)
- ⚙️ Generate a default configuration file if none exists
- 📊 Create synthetic training data using default parameters
- 🤖 Train and test the Random Forest model
- 🎯 Deploy the model on field data

### Manual Setup (Advanced Users)

1. **Create directory structure**:
   ```bash
   mkdir -p constants training_data output logs pipeline_logs field_data
   ```

2. **Add your field data** (optional for first run):
   - Place `GRV_2D_Data.txt` in `field_data/` - Gravity observations
   - Place `RTP_2D_Data.txt` in `field_data/` - Magnetic observations

3. **Customize configuration** (optional):
   - Modify `constants/input_config.yaml` for your specific use case
   - The pipeline will create this file with default values if it doesn't exist

4. **Run individual scripts**:
   ```bash
   python generate_synthetic_data.py  # Creates config + synthetic data
   python train_rf.py                 # Trains model
   python test_rf.py                  # Validates model  
   python deploy_rf.py                # Applies to field data
   ```

## 🏗️ Project Structure

```
project/
├── run_rf_pipeline.py          # Main pipeline script (RECOMMENDED)
├── generate_synthetic_data.py  # Synthetic data + config generation
├── train_rf.py                 # Model training script
├── test_rf.py                  # Model testing script  
├── deploy_rf.py                # Field deployment script
├── constants/
│   ├── input_config.yaml       # Configuration (auto-generated if missing)
│   ├── Gkernelsp.npz           # Gravity kernels (generated)
│   ├── Mkernelsp.npz           # Magnetic kernels (generated)
│   └── rf_inversion.joblib     # Trained model (generated)
├── training_data/              # Synthetic training datasets (.npy)
├── output/                     # Results and predictions
├── field_data/                 # Field observations (optional)
│   ├── GRV_2D_Data.txt         # Gravity field data
│   └── RTP_2D_Data.txt         # Magnetic field data
├── logs/                       # Execution logs
└── pipeline_logs/              # Pipeline execution logs
```

## ⚙️ Configuration

The pipeline uses `constants/input_config.yaml` for all parameters. This file is **automatically created with default values** if it doesn't exist.

### Default Configuration Values

When no config file exists, the pipeline creates one with these defaults:

```yaml
# Data parameters
Ndatapoints: 32
gravity_data_std: 0.2      # mGal
rtp_data_std: 1.0          # nT
gravity_data_path: 'field_data/GRV_2D_Data.txt'
rtp_data_path: 'field_data/RTP_2D_Data.txt'

# Model space parameters  
CX: 64
CY: 64 
CZ: 64
Z0: 0
ZEND: 10000
Pad_Length: 5000

# Physical properties
rho_sed: [0.0]
rho_salt: [-0.13]
rho_base: [0.4]

# Training parameters
NKEEP: 50
Nmodels: 1000000
Kmin: 6
Kmax: 50
max_num_training_model: 10000
k_list: [3, 9, 15, 21, 27, 33, 39, 45, 51]

# Magnetic field parameters
Inc: 1.5708    # 90° in radians
Dec: 0.0
Fe: 43314      # nT
```

### Customizing Configuration

After the first run, modify `constants/input_config.yaml` for your specific needs:

- **Grid resolution**: Adjust `CX`, `CY`, `CZ` for model complexity
- **Physical properties**: Update density values for your geology
- **Data noise**: Modify `gravity_data_std` and `rtp_data_std` for your data quality
- **Training size**: Adjust `Nmodels` and `max_num_training_model` based on available compute resources

## 📊 Pipeline Stages

### 1. Synthetic Data Generation (`generate_synthetic_data.py`)

- **Input**: Field data geometry (or default parameters)
- **Process**:
  - Creates default configuration file if none exists
  - Generates gravity and magnetic forward modeling kernels
  - Creates synthetic subsurface models using Voronoi tessellation
  - Simulates observed data with added noise
  - Uses multi-threading for efficient generation
- **Output**: 
  - `constants/input_config.yaml` - Configuration file
  - Multiple `.npy` training files in `training_data/`
  - Forward modeling kernels in `constants/`

### 2. Training Phase (`train_rf.py`)

- **Input**: Synthetic training models + configuration
- **Process**:
  - Loads pre-computed kernels and configuration
  - Applies data balancing to handle class imbalance
  - Trains Random Forest classifier with time-series cross-validation
  - Splits data into training/test sets
- **Output**: 
  - `constants/rf_inversion.joblib` - Trained model
  - `training_data/train.yaml`, `training_data/test.yaml` - Data splits

### 3. Testing Phase (`test_rf.py`)

- **Input**: Trained model + test datasets + configuration
- **Process**:
  - Loads and validates model performance
  - Generates comprehensive evaluation metrics
  - Produces visualizations (confusion matrices, reliability diagrams)
  - Computes calibration metrics
- **Output**:
  - Performance metrics and classification reports
  - Model validation plots and statistics
  - Reliability diagrams and confusion matrices

### 4. Deployment Phase (`deploy_rf.py`)

- **Input**: Trained model + field observation data + configuration
- **Process**:
  - Applies model to real field data
  - Generates density and susceptibility models
  - Creates final visualizations and 3D models
  - Compares with known geological structures
- **Output**:
  - Predicted density/susceptibility models
  - Field data fit plots and 3D visualizations
  - Comparison with regularization-based methods

## 🔄 Workflow Options

### First-Time Users
```bash
python run_rf_pipeline.py
```
*Let the pipeline handle everything automatically*

### Experienced Users
1. Review/modify `constants/input_config.yaml`
2. Run pipeline: `python run_rf_pipeline.py`

### Advanced Users
1. Customize configuration
2. Run scripts individually for debugging
3. Modify individual script parameters as needed

## 🐛 Troubleshooting

### Common Issues

1. **First Run Takes Long Time**:
   - Synthetic data generation is computationally intensive
   - Subsequent runs will be much faster as data is reused

2. **Memory Errors**:
   - Reduce `Nmodels` in configuration
   - Use smaller grid sizes (`CX`, `CY`, `CZ`)
   - Increase `NKEEP` to reduce file count

3. **Missing Field Data**:
   - Pipeline will use default parameters
   - Add your field data to `field_data/` for realistic results

4. **Configuration Changes**:
   - Delete `training_data/` folder to regenerate data with new parameters
   - Delete `constants/` folder to reset everything

### Logs and Debugging

- **Pipeline logs**: `pipeline_logs/rf_pipeline_*.log`
- **Individual script logs**: `logs/` folder
- **Configuration**: `constants/input_config.yaml`

## 📈 Outputs

The pipeline generates:

- **Configuration**: `constants/input_config.yaml` (auto-created)
- **Synthetic Data**: Training datasets in `training_data/`
- **Model Files**: Trained classifier and kernels in `constants/`
- **Performance Metrics**: Accuracy, F1 scores, calibration metrics
- **Visualizations**: Confusion matrices, reliability diagrams, 3D models
- **Predictions**: Density/susceptibility models in `output/`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License.

---

*For detailed technical documentation of individual scripts, refer to the source code comments and module docstrings.*