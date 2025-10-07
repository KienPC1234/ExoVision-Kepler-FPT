import streamlit as st
import pandas as pd
import numpy as np
import sys
from web.helper import streamlit_custom_components as scc
from ModelTrainer.modelV1.model_loader import ModelLoader, TFNNClassifier
from ModelTrainer.modelV1.data_preprocess import *
from datetime import datetime
from web.db import connect_db
from web.db.models.users import User

st.set_page_config(page_title="Exoplanet Predictor", layout="wide")
sys.modules['__main__'].TFNNClassifier = TFNNClassifier

@st.cache_resource
def get_loader():
    return ModelLoader()

@st.cache_resource
def get_db():
    return connect_db()

st.markdown(
    """
# 🌌 Exoplanet Predictor

Predict exoplanet characteristics either by **manual input** or by **uploading CSV** from K2, KOI, or TESS catalogs.

Use the **Simple** tab for a quick planet check or **Advanced** tab for batch processing.

---
"""
)

with st.expander("ℹ️ How to obtain K2, KOI, or TESS data"):
    st.markdown("""
**Data Sources:**  
| Mission | Link |
|---------|------|
| KOI | [Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu) |
| K2 | [STScI K2 Archive](https://archive.stsci.edu/k2) |
| TESS | [TESS TOI Tables](https://exoplanetarchive.ipac.caltech.edu/docs/tessmission.html) |

**Required Columns:**  
`pl_radj`, `pl_orbper`, `pl_trandur`, `depth`, `st_teff`, `st_logg`, `st_rad`, `koi_kepmag`  

**Optional Columns:**  
`koi_impact`, `pl_insol`, `pl_eqt`, `st_dist`  

**Column Descriptions & Units:**  
| Column | Description | Unit |
|--------|------------|------|
| koi_kepmag | Kepler-band brightness | mag |
| pl_radj | Planet radius | R_J (Jupiter radii) |
| koi_impact | Impact parameter | dimensionless |
| pl_trandur | Transit duration | hours |
| depth | Transit depth | fraction (normalized from ppm or %) |
| pl_orbper | Orbital period | days |
| st_teff | Stellar effective temperature | K |
| st_logg | Stellar surface gravity | dex (log10(cm/s²)) |
| st_rad | Stellar radius | R_Sun |
| pl_insol | Insolation flux | F_Earth |
| pl_eqt | Planet equilibrium temperature | K |
| st_dist | Stellar distance | pc |
""")

with st.expander("📽️ Click to watch the tutorial", expanded=False):
    st.write("This video explains how to use the Exoplanet Predictor app step by step.")
    st.video("https://youtu.be/0TWm4pI2Cqs")

# ----------------------------
# Tabs
# ----------------------------
tab = st.tabs(["Simple", "Advanced"])

# ----------------------------
# TAB 1: Simple
# ----------------------------
with tab[0]:
    # --- Load model ---
    loader = get_loader()
    db = get_db()
    st.header("Simple Input - Enter Single Planet Data")

    # --- Field definitions ---
    REQUIRED_FIELDS = [
        ('koi_kepmag', 'Kepler-band brightness (mag)'),
        ('pl_radj', 'Planet radius (R_J)'),
        ('pl_orbper', 'Orbital period (days)'),
        ('pl_trandur', 'Transit duration (hours)'),
        ('depth', 'Transit depth (fraction)'),
        ('st_teff', 'Stellar effective temperature (K)'),
        ('st_logg', 'Stellar surface gravity (dex)'),
        ('st_rad', 'Stellar radius (R_Sun)'),
    ]

    OPTIONAL_FIELDS = [
        ('koi_impact', 'Impact parameter (dimensionless)'),
        ('pl_insol', 'Insolation flux (F_Earth)'),
        ('pl_eqt', 'Planet equilibrium temperature (K)'),
        ('st_dist', 'Stellar distance (pc)')
    ]

    # --- Input form ---
    manual_data = {}
    cols = st.columns(2)
    for i, (field, label) in enumerate(REQUIRED_FIELDS):
        col = cols[i % 2]
        manual_data[field] = col.number_input(f"{label} *", value=0.0, step=0.01, format="%.5f")

    for i, (field, label) in enumerate(OPTIONAL_FIELDS):
        col = cols[i % 2]
        val = col.text_input(f"{label}", "")
        try:
            manual_data[field] = float(val) if val.strip() else np.nan
        except ValueError:
            manual_data[field] = np.nan

    # --- Predict button ---
    if st.button("Check Planet"):
        # --- Impute numeric fields ---
        temp_df = pd.DataFrame([manual_data])
        numeric_cols = temp_df.select_dtypes(include=[np.number]).columns.tolist()

        if loader.num_imputer is not None and numeric_cols:
            try:
                transformed = loader.num_imputer.transform(temp_df[numeric_cols])
                temp_df[numeric_cols] = pd.DataFrame(transformed, columns=numeric_cols)
            except Exception:
                temp_df[numeric_cols] = temp_df[numeric_cols].fillna(temp_df[numeric_cols].median())
        else:
            temp_df[numeric_cols] = temp_df[numeric_cols].fillna(temp_df[numeric_cols].median())

        # Ensure no NaN left
        temp_df[numeric_cols] = temp_df[numeric_cols].fillna(temp_df[numeric_cols].median())
        imputed_data = temp_df.to_dict('records')[0]

        # --- Predict ---
        X = loader.prepare_input(imputed_data)
        out = loader.predict(X)

        # --- Display results ---
        st.subheader("Model Prediction")
        col1, col2 = st.columns(2)
        col1.metric("Predicted Class", out['class'], "1 = Confirmed/Candidate")
        col2.metric("Probability", f"{out['probability']:.4f}")

        # --- Save history to DB ---
        username = st.session_state.get("auth_user", "anonymous")
        user: User = db.get_user(username)
        if user:
            result_markdown = f"Class: {out['class']}, Probability: {out['probability']:.4f}"
            db.add_prediction_record(
                user=user,
                type="manual_input",
                name="Single Planet Prediction",
                result_markdown=result_markdown,
                user_data_path=None,
                output_filename=None,
                timestamp=datetime.now()
            )

        # --- Embed iframe and pass JSON directly ---
        json_data = {**imputed_data, 'disposition': out['class']}
        scc.embed_iframe(
            url=f"https://iframe.fptoj.com/iframe/plant_preview/",
            json_data=json_data,
            height=600
        )

# ----------------------------
# TAB 2: Advanced
# ----------------------------
with tab[1]:
    st.header("Advanced Input - Upload CSV (KOI / K2 / TESS / Standardized)")

    csv_type = st.selectbox(
        "Select Data type", 
        [
            "KOI (Kepler Objects of Interest - Raw CSV format)",
            "K2 (K2 Mission - Raw CSV format)", 
            "TESS (Transiting Exoplanet Survey Satellite - Raw CSV format)",
            "Standardized Data (Pre-processed CSV with exactly 12 columns as per guidelines)"
        ],
        format_func=lambda x: x  # Use the full descriptive label for display
    )
    
    # Map descriptive label back to internal type for processing
    type_mapping = {
        "KOI (Kepler Objects of Interest - Raw CSV format)": "koi",
        "K2 (K2 Mission - Raw CSV format)": "k2", 
        "TESS (Transiting Exoplanet Survey Satellite - Raw CSV format)": "tess",
        "Standardized Data (Pre-processed CSV with exactly 12 columns as per guidelines)": "standardized"
    }
    internal_type = type_mapping[csv_type]
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file:
        df_csv = pd.read_csv(uploaded_file)
        # Add original index column
        df_csv['original_index'] = df_csv.index
        
        # Do not fillna globally, handle per row and in prepare_input
        
        # Apply standardization based on type
        if internal_type == "koi":
            df_csv = standardize_koi(df_csv)
        elif internal_type == "k2":
            df_csv = standardize_k2(df_csv)
        elif internal_type == "tess":
            df_csv = standardize_tess(df_csv)
        elif internal_type == "standardized":
            # For standardized data, skip standardization but validate the 12 columns
            st.info("Assuming uploaded CSV is already standardized. Validating columns...")
            pass  # No standardization needed
        
        # Drop noise features
        df_csv = drop_noise_features(df_csv)
        
        # Drop disposition if present (for prediction)
        if 'disposition' in df_csv.columns:
            df_csv = df_csv.drop('disposition', axis=1)
        
        # Define the 12 expected columns
        expected_cols = [
            'koi_kepmag', 'pl_radj', 'koi_impact', 'pl_trandur', 'depth',
            'pl_orbper', 'st_teff', 'st_logg', 'st_rad', 'pl_insol',
            'pl_eqt', 'st_dist'
        ]
        
        # For standardized data, check if all 12 columns are present
        if internal_type == "standardized":
            missing_expected = [col for col in expected_cols if col not in df_csv.columns]
            if missing_expected:
                st.error(f"For Standardized Data, all 12 expected columns must be present. Missing: {missing_expected}. Please ensure your CSV matches the guidelines.")
                st.stop()  # Halt processing if validation fails
            else:
                st.success("Standardized data validation passed! All 12 columns detected.")
        else:
            # For raw types, add missing expected columns as NaN
            for col in expected_cols:
                if col not in df_csv.columns:
                    df_csv[col] = np.nan
                    st.warning(f"Column '{col}' was missing in CSV and added as NaN.")
        
        # Reorder to have expected columns first (optional, for clarity)
        df_csv = df_csv.reindex(columns=expected_cols + [col for col in df_csv.columns if col not in expected_cols], fill_value=np.nan)
        
        # Required columns for processing a row
        required_cols = ['pl_radj', 'pl_orbper', 'pl_trandur', 'depth', 'st_teff', 'st_logg', 'st_rad', 'koi_kepmag']
        
        # Check if all required columns are now present (after standardization)
        missing_required = [c for c in required_cols if c not in df_csv.columns]
        if missing_required:
            st.error(f"Missing required columns after standardization: {missing_required}. Cannot process.")
        else:
            loader = get_loader()
            
            # Initialize prediction columns
            df_csv['predicted_class'] = np.nan
            df_csv['probability'] = np.nan
            df_csv['status'] = 'skipped: missing params'
            
            # Process row by row
            processed_count = 0
            skipped_count = 0
            real_count = 0
            fake_count = 0
            
            total_rows = len(df_csv)
            progress_bar = st.progress(0)
            
            for idx, row in df_csv.iterrows():
                row_dict = row.to_dict()
                # Remove prediction columns if present
                row_dict.pop('predicted_class', None)
                row_dict.pop('probability', None)
                row_dict.pop('status', None)
                row_dict.pop('original_index', None)  # Exclude index from input
                
                # Check required fields: present and valid (not NaN and >0 where makes sense)
                missing_params = [col for col in required_cols if pd.isna(row_dict.get(col)) or row_dict.get(col, 0) <= 0]
                if not missing_params:
                    try:
                        X = loader.prepare_input(row_dict)
                        out = loader.predict(X)
                        df_csv.at[idx, 'predicted_class'] = out['class']
                        df_csv.at[idx, 'probability'] = out['probability']
                        df_csv.at[idx, 'status'] = 'processed'
                        processed_count += 1
                        if out['class'] == 1:
                            real_count += 1
                        else:
                            fake_count += 1
                    except Exception as e:
                        st.warning(f"Error processing row {idx}: {e}")
                        skipped_count += 1
                else:
                    df_csv.at[idx, 'status'] = f"skipped: missing {', '.join(missing_params)}"
                    skipped_count += 1
                
                # Update progress bar
                progress_bar.progress((idx + 1) / total_rows)
            
            st.info(f"Total rows: {total_rows}")
            st.info(f"Processed: {processed_count} rows, Skipped: {skipped_count} rows")
            if processed_count > 0:
                real_pct = (real_count / processed_count) * 100
                fake_pct = (fake_count / processed_count) * 100
                st.metric("Confirmed/Candidate Planets (%)", f"{real_pct:.1f}%")
                st.metric("False Positives (%)", f"{fake_pct:.1f}%")
            
            st.subheader("Processed Data Preview")
            preview_cols = ['original_index', 'status', 'predicted_class', 'probability'] + expected_cols
            st.dataframe(df_csv[preview_cols].head())
            
            # Save history to file and DB
            username = st.session_state.get("auth_user", "anonymous")
            user: User = db.get_user(username)
            if user:
                result_markdown = f"Total rows: {total_rows}, Processed: {processed_count}, Skipped: {skipped_count}, Confirmed: {real_count}, False Positives: {fake_count}"
                import uuid
                import os
                from datetime import datetime
                os.makedirs("userdata", exist_ok=True)
                filename = str(uuid.uuid4()) + ".csv"
                full_path = os.path.join("userdata", filename)
                df_csv.to_csv(full_path, index=False)
                db.add_prediction_record(
                    user=user,
                    type="csv_upload",
                    name=f"{internal_type.upper()} CSV Predictions ({total_rows} rows)",
                    result_markdown=result_markdown,
                    user_data_path="userdata",
                    output_filename=filename,
                    timestamp=datetime.now()
                )
            
            # Download updated CSV
            csv_out = df_csv.to_csv(index=False).encode('utf-8')
            st.download_button("Download Processed CSV with Predictions", csv_out, "exoplanet_predictions.csv", "text/csv")