import sys
import os
import uuid
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

from web.helper import streamlit_custom_components as scc
from ModelTrainer.modelV1.model_loader import ModelLoader, TFNNClassifier
from ModelTrainer.modelV1.data_preprocess import (
    standardize_koi,
    standardize_k2,
    standardize_tess,
    drop_noise_features,
)
from web.db import connect_db
from web.db.models.users import User
from web.helper.translator import translate_text  # translation helper

# --- language helper ---
lang = st.session_state.get("preferences", {}).get("lang", "en")
def t(text: str) -> str:
    """Translate text using user's preference (wrapper)."""
    try:
        return translate_text(text, lang)
    except Exception:
        # Fail gracefully and return original
        return text

# --- page config (call after lang available) ---
st.set_page_config(page_title=t("Exoplanet Predictor"), layout="wide")
sys.modules["__main__"].TFNNClassifier = TFNNClassifier

@st.cache_resource
def get_loader():
    return ModelLoader()

@st.cache_resource
def get_db():
    return connect_db()


# ---- Intro ----
st.markdown(
    t(
"""
# 🌌 Exoplanet Predictor

Predict exoplanet characteristics either by **manual input** or by **uploading CSV** from K2, KOI, or TESS catalogs.

Use the **Simple** tab for a quick planet check or **Advanced** tab for batch processing.

---
""")
)

with st.expander(t("ℹ️ How to obtain K2, KOI, or TESS data")):
    st.markdown(
        t(
            """
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
    )

with st.expander(t("📽️ Click to watch the tutorial"), expanded=False):
    st.write(t("This video explains how to use the Exoplanet Predictor app step by step."))
    st.video("https://youtu.be/0TWm4pI2Cqs")

# ---- Tabs ----
tab = st.tabs([t("Simple"), t("Advanced")])

# ---- TAB 1: Simple ----
with tab[0]:
    loader = get_loader()
    db = get_db()
    st.header(t("Simple Input - Enter Single Planet Data"))

    REQUIRED_FIELDS = [
        ("koi_kepmag", "Kepler-band brightness (mag)"),
        ("pl_radj", "Planet radius (R_J)"),
        ("pl_orbper", "Orbital period (days)"),
        ("pl_trandur", "Transit duration (hours)"),
        ("depth", "Transit depth (fraction)"),
        ("st_teff", "Stellar effective temperature (K)"),
        ("st_logg", "Stellar surface gravity (dex)"),
        ("st_rad", "Stellar radius (R_Sun)"),
    ]

    OPTIONAL_FIELDS = [
        ("koi_impact", "Impact parameter (dimensionless)"),
        ("pl_insol", "Insolation flux (F_Earth)"),
        ("pl_eqt", "Planet equilibrium temperature (K)"),
        ("st_dist", "Stellar distance (pc)"),
    ]

    manual_data = {}
    cols = st.columns(2)

    for i, (field, label) in enumerate(REQUIRED_FIELDS):
        col = cols[i % 2]
        manual_data[field] = col.number_input(
            t(f"{label} *"), value=0.0, step=0.01, format="%.5f"
        )

    for i, (field, label) in enumerate(OPTIONAL_FIELDS):
        col = cols[i % 2]
        val = col.text_input(t(label), "")
        try:
            manual_data[field] = float(val) if val.strip() else np.nan
        except ValueError:
            manual_data[field] = np.nan

    if st.button(t("Check Planet")):
        temp_df = pd.DataFrame([manual_data])
        numeric_cols = temp_df.select_dtypes(include=[np.number]).columns.tolist()

        if loader.num_imputer is not None and numeric_cols:
            try:
                transformed = loader.num_imputer.transform(temp_df[numeric_cols])
                temp_df[numeric_cols] = pd.DataFrame(transformed, columns=numeric_cols)
            except Exception:
                temp_df[numeric_cols] = temp_df[numeric_cols].fillna(
                    temp_df[numeric_cols].median()
                )
        else:
            temp_df[numeric_cols] = temp_df[numeric_cols].fillna(
                temp_df[numeric_cols].median()
            )

        # Ensure no NaN left in numeric cols
        temp_df[numeric_cols] = temp_df[numeric_cols].fillna(
            temp_df[numeric_cols].median()
        )
        imputed_data = temp_df.to_dict("records")[0]

        # Predict
        try:
            X = loader.prepare_input(imputed_data)
            out = loader.predict(X)
        except Exception as e:
            st.error(t(f"Error during prediction: {e}"))
            out = {"class": None, "probability": 0.0}

        st.subheader(t("Model Prediction"))
        col1, col2 = st.columns(2)
        col1.metric(t("Predicted Class"), out["class"], t("1 = Confirmed/Candidate"))
        col2.metric(t("Probability"), f"{out['probability']:.4f}")

        # Save history to DB
        user: User = db.get_user(st.user.email)
        if user:
            result_markdown = f"Class: {out['class']}, Probability: {out['probability']:.4f}"
            db.add_prediction_record(
                user=user,
                type="manual_input",
                name="Single Planet Prediction",
                result_markdown=result_markdown,
                user_data_path=None,
                output_filename=None,
                timestamp=datetime.now(),
            )

        # Embed iframe and pass JSON directly
        json_data = {**imputed_data, "disposition": out["class"]}
        scc.embed_iframe(
            url="https://iframe.fptoj.com/iframe/plant_preview/",
            json_data=json_data,
            height=600,
        )

# ---- TAB 2: Advanced ----
with tab[1]:
    st.header(t("Advanced Input - Upload CSV (KOI / K2 / TESS / Standardized)"))

    # options use internal keys and translated labels for display
    csv_label_map = {
        "koi": t("KOI (Kepler Objects of Interest - Raw CSV format)"),
        "k2": t("K2 (K2 Mission - Raw CSV format)"),
        "tess": t("TESS (Transiting Exoplanet Survey Satellite - Raw CSV format)"),
        "standardized": t(
            "Standardized Data (Pre-processed CSV with exactly 12 columns as per guidelines)"
        ),
    }

    csv_choice = st.selectbox(
        t("Select Data type"),
        options=list(csv_label_map.keys()),
        format_func=lambda k: csv_label_map[k],
    )
    internal_type = csv_choice

    uploaded_file = st.file_uploader(t("Choose CSV file"), type="csv")

    if uploaded_file:
        try:
            df_csv = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(t(f"Failed to read CSV: {e}"))
            st.stop()

        df_csv["original_index"] = df_csv.index

        # Standardize based on chosen internal type
        if internal_type == "koi":
            df_csv = standardize_koi(df_csv)
        elif internal_type == "k2":
            df_csv = standardize_k2(df_csv)
        elif internal_type == "tess":
            df_csv = standardize_tess(df_csv)
        elif internal_type == "standardized":
            # skip standardization, validate later
            pass

        # Drop noise features if helper exists
        try:
            df_csv = drop_noise_features(df_csv)
        except Exception:
            # If drop_noise_features fails, keep going
            pass

        # Drop disposition column if present (we're predicting)
        if "disposition" in df_csv.columns:
            df_csv = df_csv.drop("disposition", axis=1)

        expected_cols = [
            "koi_kepmag",
            "pl_radj",
            "koi_impact",
            "pl_trandur",
            "depth",
            "pl_orbper",
            "st_teff",
            "st_logg",
            "st_rad",
            "pl_insol",
            "pl_eqt",
            "st_dist",
        ]

        # Validate standardized
        if internal_type == "standardized":
            missing_expected = [col for col in expected_cols if col not in df_csv.columns]
            if missing_expected:
                st.error(
                    t(
                        "For Standardized Data, all 12 expected columns must be present. Missing: "
                        + str(missing_expected)
                        + ". "
                        + "Please ensure your CSV matches the guidelines."
                    )
                )
                st.stop()
            else:
                st.success(t("Standardized data validation passed! All 12 columns detected."))
        else:
            # Add missing expected columns as NaN and warn (translated)
            for col in expected_cols:
                if col not in df_csv.columns:
                    df_csv[col] = np.nan
                    st.warning(t(f"Column '{col}' was missing in CSV and added as NaN."))

        # Reorder columns (expected first)
        df_csv = df_csv.reindex(
            columns=expected_cols + [c for c in df_csv.columns if c not in expected_cols],
            fill_value=np.nan,
        )

        # Check required columns after standardization
        required_cols = [
            "pl_radj",
            "pl_orbper",
            "pl_trandur",
            "depth",
            "st_teff",
            "st_logg",
            "st_rad",
            "koi_kepmag",
        ]
        missing_required = [c for c in required_cols if c not in df_csv.columns]
        if missing_required:
            st.error(
                t(
                    "Missing required columns after standardization: "
                    + str(missing_required)
                    + ". Cannot process."
                )
            )
            st.stop()

        # Prepare loader and DB
        loader = get_loader()
        db = get_db()

        # Initialize prediction columns
        df_csv["predicted_class"] = np.nan
        df_csv["probability"] = np.nan
        df_csv["status"] = "skipped: missing params"

        total_rows = len(df_csv)
        processed_count = 0
        skipped_count = 0
        real_count = 0
        fake_count = 0

        progress_bar = st.progress(0)
        # Use enumerate over iterrows to get sequential progress
        for i, (idx, row) in enumerate(df_csv.iterrows()):
            row_dict = row.to_dict()
            # remove helper columns
            for k in ("predicted_class", "probability", "status", "original_index"):
                row_dict.pop(k, None)

            # Check required fields presence/validity
            missing_params = [
                col
                for col in required_cols
                if (pd.isna(row_dict.get(col)) or (isinstance(row_dict.get(col), (int, float)) and row_dict.get(col) <= 0))
            ]

            if not missing_params:
                try:
                    X = loader.prepare_input(row_dict)
                    out = loader.predict(X)
                    df_csv.at[idx, "predicted_class"] = out["class"]
                    df_csv.at[idx, "probability"] = out["probability"]
                    df_csv.at[idx, "status"] = "processed"
                    processed_count += 1
                    if out["class"] == 1:
                        real_count += 1
                    else:
                        fake_count += 1
                except Exception as e:
                    st.warning(t(f"Error processing row {idx}: {e}"))
                    df_csv.at[idx, "status"] = t(f"skipped: error processing row {idx}")
                    skipped_count += 1
            else:
                df_csv.at[idx, "status"] = t(f"skipped: missing {', '.join(missing_params)}")
                skipped_count += 1

            # update progress (i is 0-based)
            progress_bar.progress(min(1.0, float(i + 1) / max(1, total_rows)))

        # Summary
        st.info(t(f"Total rows: {total_rows}"))
        st.info(t(f"Processed: {processed_count} rows, Skipped: {skipped_count} rows"))

        if processed_count > 0:
            real_pct = (real_count / processed_count) * 100
            fake_pct = (fake_count / processed_count) * 100
            st.metric(t("Confirmed/Candidate Planets (%)"), f"{real_pct:.1f}%")
            st.metric(t("False Positives (%)"), f"{fake_pct:.1f}%")

        st.subheader(t("Processed Data Preview"))
        preview_cols = ["original_index", "status", "predicted_class", "probability"] + expected_cols
        st.dataframe(df_csv[preview_cols].head())

        # Save to user history and file
        user: User = db.get_user(st.user.email)
        if user:
            result_markdown = t(
                f"Total rows: {total_rows}, Processed: {processed_count}, Skipped: {skipped_count}, Confirmed: {real_count}, False Positives: {fake_count}"
            )
            os.makedirs("userdata", exist_ok=True)
            filename = str(uuid.uuid4()) + ".csv"
            full_path = os.path.join("userdata", filename)
            try:
                df_csv.to_csv(full_path, index=False)
                db.add_prediction_record(
                    user=user,
                    type="csv_upload",
                    name=f"{internal_type.upper()} CSV Predictions ({total_rows} rows)",
                    result_markdown=result_markdown,
                    user_data_path="userdata",
                    output_filename=filename,
                    timestamp=datetime.now(),
                )
            except Exception as e:
                st.error(t(f"Failed to save results: {e}"))

        # Download updated CSV
        try:
            csv_out = df_csv.to_csv(index=False).encode("utf-8")
            st.download_button(
                t("Download Processed CSV with Predictions"),
                csv_out,
                "exoplanet_predictions.csv",
                "text/csv",
            )
        except Exception as e:
            st.error(t(f"Failed to prepare download: {e}"))
