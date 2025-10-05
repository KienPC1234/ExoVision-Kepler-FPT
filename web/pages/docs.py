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
""")

st.header("🛠️ Technical Details")
st.markdown("""
- **Exoplanet Predictor** uses a machine learning model (TFNNClassifier) trained on astrophysical parameters from NASA catalogs.
- **Flux Prediction** uses a deep learning PatchTST model for time series classification of light curves.
- All predictions and results are saved to user history for review and download.
""")