import streamlit as st

st.title("📄 Model & Feature Documentation")
st.markdown("""
# ExoVision Kepler FPT: Documentation

This page provides documentation for the main features and models available in the ExoVision Kepler FPT application.
""")

st.header("🌌 Exoplanet Predictor")
st.markdown("""
**Purpose:** Predict exoplanet characteristics and classification (confirmed/candidate/false positive) using astrophysical parameters.

**Features:**
- Manual input for single planet prediction
- Batch prediction via CSV upload (K2, KOI, TESS catalogs)
- Data standardization and preprocessing
- Probability and class output for each planet
- Saves prediction history for user

**Required Columns for CSV:**
| Column         | Description                        | Unit         |
|---------------|------------------------------------|--------------|
| koi_kepmag    | Kepler-band brightness             | mag          |
| pl_radj       | Planet radius                      | R_J          |
| pl_orbper     | Orbital period                     | days         |
| pl_trandur    | Transit duration                   | hours        |
| depth         | Transit depth                      | fraction     |
| st_teff       | Stellar effective temperature      | K            |
| st_logg       | Stellar surface gravity            | dex          |
| st_rad        | Stellar radius                     | R_Sun        |

**Optional Columns:** koi_impact, pl_insol, pl_eqt, st_dist

**How to Use:**
1. Go to the Exoplanet Predictor page.
2. Use the Simple tab for manual input, or Advanced tab for CSV upload.
3. For CSV, select the data type (KOI/K2/TESS) and upload your file.
4. Review results, download processed CSV, and view prediction history.

---

## 🚀 Training Guide for Model V1

This section explains how to train **Model V1** using the prepared metadata features.


#### 📊 Step 1: Prepare Metadata Features

Your dataset should include **15 key features**:

1. `koi_kepmag`: Kepler-band brightness (magnitude) – unit: mag
2. `pl_radj`: Planet radius – unit: R<sub>J</sub> (Jupiter radius, converted if needed)
3. `koi_impact`: Impact parameter – dimensionless
4. `pl_trandur`: Transit duration – unit: hours
5. `depth`: Transit depth – unit: fraction (normalized from ppm or %)
6. `pl_orbper`: Orbital period – unit: days
7. `st_teff`: Stellar effective temperature – unit: K
8. `st_logg`: Stellar surface gravity – unit: dex (log10(cm/s²))
9. `st_rad`: Stellar radius – unit: R<sub>☉</sub> (Solar radius)
10. `pl_insol`: Insolation flux – unit: F<sub>Earth</sub> (Earth flux ratio)
11. `pl_eqt`: Equilibrium temperature – unit: K
12. `st_dist`: Stellar distance – unit: parsec (pc)
13. `density_proxy`: Derived proxy (1 / pl_radj³) – unitless
14. `habitability_proxy`: Derived proxy (pl_orbper * 0.7 / st_teff) – unitless
15. `transit_shape_proxy`: Derived proxy (depth / pl_trandur) – fraction/hour

---

#### 🛠 Step 2: Clone Repository

```bash
git clone https://github.com/KienPC1234/ExoVision-Kepler-FPT
cd ExoVision-Kepler-FPT
```

---

#### 📦 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

#### 📂 Step 4: Prepare Dataset

* Place your dataset as a CSV file at:

  ```
  data/merged_processed.csv
  ```
* Or modify and run the preprocessing script:

```bash
python ModelTrainer/modelV1/data_preprocess.py
```

This will generate the processed dataset.

---

#### 🤖 Step 5: Train the Model

Run the model builder script:

```bash
python ModelTrainer/modelV1/model_builder.py
```

---

#### 📥 Step 6: Load and Use the Model

Check the usage example in:

```
ModelTrainer/modelV1/model_loader.py
```

This file demonstrates how to load the trained model and use it for predictions.

✅ You are now ready to train and test **Model V1** with exoplanet metadata!

---
""")

st.header("💡 Exoplanet Flux Prediction")
st.markdown("""
**Purpose:** Predict the existence of exoplanets from flux (light curve) time series data using deep learning (PatchTST model).

**Features:**
- Upload multiple files per celestial body (supports .fits, .tbl, .csv, .tsv)
- Merge and preprocess light curve data
- Predict exoplanet existence for each body
- Visualize flux light curves
- Download prediction results as CSV
- Saves prediction history for user

**File Requirements:**
- Supported formats: `.fits`, `.tbl`, `.csv`, `.tsv`
- Required columns: `TIME` and one of `SAP_FLUX`, `PDCSAP_FLUX`, `FLUX` (case-insensitive)
- Optional: `KEPID` (if missing, will use -1)

**How to Use:**
1. Go to the Exoplanet Flux Prediction page.
2. Use Tab 1 to upload files for multiple bodies, or Tab 2 for a preprocessed table.
3. Click "Predict All Bodies" or "Predict" to run predictions.
4. View results, download CSV, and see prediction history.

---

## 🌌 Model V2: Time-Series (Lightcurves) Training

### 📊 Step 1: Prepare Lightcurve Dataset

Two options:

1. Run preprocessing:

   ```bash
   python /home/sysadmin/TrainAI/ModelTrainer/modelV2/data_preprocess.py
   ```
2. Or prepare a parquet dataset at:

   ```
   /home/sysadmin/TrainAI/data/koi_lightcurves.parquet
   ```

   Required columns:

   ```python
   ['kepid', 'time', 'flux', 'label']
   ```

---

### 🛠 Step 2: Clone Repository

```bash
git clone https://github.com/KienPC1234/ExoVision-Kepler-FPT
cd ExoVision-Kepler-FPT
```

---

### 📦 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### ⚙ Step 4: Configuration

Training configs are defined in:

```
ModelTrainer/modelV2/model_builder.py
```

Example `CONFIG`:

```python
CONFIG = {
    'file_path': Path('data/koi_lightcurves.parquet'),
    'cols': ['kepid', 'time', 'flux', 'label'],
    'chunk_rows': 200_000,

    # Windowing / reduction
    'max_seq_len': 1024,
    'window_step': 512,
    'downsample_factor': 1,

    # Training
    'batch_size': 256,
    'num_epochs': 10,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'test_size': 0.2,
    'random_state': 42,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_workers': 0,
    'binary_classification': True,

    # PatchTST-specific params
    'model_arch': 'PatchTST',
    'model_params': {
        'seq_len': 1024,
        'patch_len': 16,
        'd_model': 128,
        'n_layers': 3,
        'n_heads': 8,
        'd_ff': 256,
        'dropout': 0.1,
        'revin': False,
        'pred_dim': None,
        'subtract_last': False,
    },

    'save_fname': 'best_patchtst',
}
```

---

### 🤖 Step 5: Train the Model

```bash
python ModelTrainer/modelV2/model_builder.py
```

Model checkpoints will be saved automatically.

---

### 📥 Step 6: Use the Model

After training, run the preprocessing or load directly with `SingletonModel`:

```python
class SingletonModel:
    def predict(self, flux: list[float], id: any):
        ""
        Predict on a single flux list and id.
        Returns:
            (predicted_class: int, probability: float)
        where probability is % confidence for class 1.
        ""
```

This allows quick diagnosis on new lightcurve sequences.

---
""")

st.header("🛠️ Technical Details")
st.markdown("""
- **Exoplanet Predictor** uses a machine learning model (TFNNClassifier) trained on astrophysical parameters from NASA catalogs.
- **Flux Prediction** uses a deep learning PatchTST model for time series classification of light curves.
- All predictions and results are saved to user history for review and download.
""")