import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from astropy.io import fits, ascii
from ModelTrainer.modelV2.model_loader import SingletonModel
from datetime import datetime
from web.db import connect_db
from web.db.models.users import User
from web.helper.translator import t
import os, uuid

# Load the singleton model with caching
@st.cache_resource
def get_model():
    return SingletonModel()

@st.cache_resource
def get_db():
    return connect_db()

model = get_model()

# -----------------------------
# Helper to read flux data
# -----------------------------
def read_flux_file(file):
    fname = Path(file.name)
    ext = fname.suffix.lower()
    df = None

    if ext in ['.fits', '.fit']:
        try:
            hdul = fits.open(file)
        except Exception as e:
            st.warning(t(f"Failed to open FITS file {fname}: {e}"))
            return None

        for hdu in hdul:
            # Skip PrimaryHDU which has no columns
            if not hasattr(hdu, 'columns'):
                continue
            upper_names = [n.upper() for n in hdu.columns.names]
            if 'TIME' in upper_names and any(f in upper_names for f in ['SAP_FLUX','PDCSAP_FLUX','FLUX']):
                time_idx = upper_names.index('TIME')
                flux_upper = next(f for f in ['SAP_FLUX','PDCSAP_FLUX','FLUX'] if f in upper_names)
                flux_idx = upper_names.index(flux_upper)
                # Extract data and convert to native endianness (NumPy 2.0 compatible)
                time_arr = hdu.data.field(time_idx)
                flux_arr = hdu.data.field(flux_idx)
                if time_arr.dtype.byteorder != '=':
                    time_data = time_arr.byteswap().view(time_arr.dtype.newbyteorder('='))
                else:
                    time_data = time_arr
                if flux_arr.dtype.byteorder != '=':
                    flux_data = flux_arr.byteswap().view(flux_arr.dtype.newbyteorder('='))
                else:
                    flux_data = flux_arr
                df = pd.DataFrame({
                    'time': time_data,
                    'flux': flux_data
                })
                break
        hdul.close()
    elif ext in ['.tbl', '.txt']:
        try:
            df = ascii.read(file).to_pandas()
        except Exception as e:
            st.warning(t(f"Failed to read table file {fname}: {e}"))
            return None
    elif ext in ['.csv', '.tsv']:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.warning(t(f"Failed to read CSV {fname}: {e}"))
            return None
    else:
        st.warning(t(f"Unsupported file type: {ext}"))
        return None

    if df is None:
        return None

    # standardize column names to UPPERCASE
    df = df.rename(columns={c: c.upper() for c in df.columns})
    
    # Map flux column if not exactly 'FLUX'
    if 'TIME' in df.columns:
        flux_cols = ['SAP_FLUX', 'PDCSAP_FLUX', 'FLUX']
        flux_col = next((col for col in flux_cols if col in df.columns), None)
        if flux_col:
            df['FLUX'] = df[flux_col]
        if 'FLUX' in df.columns:
            # Keep only necessary columns
            if 'KEPID' not in df.columns:
                df['KEPID'] = '-1'
            df = df[['KEPID', 'TIME', 'FLUX']].copy()
            # Rename back to lower for consistency
            df = df.rename(columns={'KEPID': 'kepid', 'TIME': 'time', 'FLUX': 'flux'})
            return df
    st.warning(t(f"File {fname} missing required columns ['TIME','FLUX'] (case insensitive)"))
    return None

# -----------------------------
# Helper to predict single body with explicit padding
# -----------------------------
def predict_body(merged_df, kepid):
    # merged_df: ['kepid','time','flux'] - already merged and sorted
    # Drop time, use flux only
    flux = merged_df['flux'].values.astype(np.float32)
    # Remove non-finite
    flux = flux[np.isfinite(flux)]
    if len(flux) == 0:
        return 0, 0.0
    
    # Explicit padding if short (though sliding_windows handles it, ensure here)
    seq_len = model.max_seq_len
    if len(flux) < seq_len:
        padded_flux = np.zeros(seq_len, dtype=np.float32)
        padded_flux[:len(flux)] = flux
        flux = padded_flux
    
    # Call single predict
    pred_class, percent = model.predict(flux.tolist(), kepid)
    return pred_class, percent

# -----------------------------
# Streamlit page
# -----------------------------
st.set_page_config(page_title=t("Exoplanet Flux Predictor"), layout="wide")
st.title(t("Exoplanet Flux Predictor using PatchTST"))
st.markdown(
    t(
"""
This page allows you to predict exoplanet transit probabilities from flux time series.

**Instructions:**
- Supported file formats: `.fits`, `.tbl`, `.csv`, `.tsv`.
- Required columns: `TIME` and one of `SAP_FLUX`, `PDCSAP_FLUX`, `FLUX` (case insensitive).  
- Optional: `KEPID`. If missing, a default value -1 will be assigned. Supports string IDs.
- Tab 1: Upload files for multiple celestial bodies. Files per body are merged, sorted by time, and processed as one sequence.
- Tab 2: Upload preprocessed file with columns ['kepid','time','flux'] for quick prediction.
- Output: Simple existence message per body.
"""
    ),
    unsafe_allow_html=True,
)

with st.expander(t("📽️ Click to watch the tutorial"), expanded=False):
    st.write(t("This video explains how to use the Exoplanet Flux Predictor app step by step."))
    st.video("https://youtu.be/0_FebaOdt38")

tab1, tab2 = st.tabs([t("Upload flux files for multiple bodies"), t("Upload preprocessed table")])

# -----------------------------
# Tab 1: Multiple bodies support with merging, single global predict
# -----------------------------
with tab1:
    st.subheader(t("Upload flux files for multiple celestial bodies"))
    
    # Use session state for dynamic addition and storage
    if 'num_bodies' not in st.session_state:
        st.session_state.num_bodies = 1
    if 'body_data' not in st.session_state:
        st.session_state.body_data = {}  # {body_id: {'kepid': str, 'files': list, 'merged_df': df}}
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []  # List of {'body_id': int, 'kepid': str, 'class': int, 'status': str}
    
    # Button to add more bodies
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button(t("+ Add Celestial Body"), key="add_body"):
            st.session_state.num_bodies += 1
            st.rerun()
    with col2:
        st.info(t(f"Current number of bodies: {st.session_state.num_bodies}"))
    
    # Button to remove last body
    if st.session_state.num_bodies > 1:
        if st.button(t("- Remove Last Body"), key="remove_body"):
            st.session_state.num_bodies -= 1
            if st.session_state.num_bodies < len(st.session_state.body_data):
                st.session_state.body_data.pop(st.session_state.num_bodies, None)
            st.rerun()
    
    # Body inputs
    for i in range(st.session_state.num_bodies):
        with st.expander(t(f"Celestial Body {i+1}"), expanded=(i==0)):
            kepid_input = st.text_input(t(f"Enter KEPLER ID for Body {i+1} (optional, will use -1 if empty; supports strings)"), key=f"kepid_{i}")
            uploaded_files = st.file_uploader(t(f"Select files for Body {i+1}"), type=['fits','fit','tbl','csv','tsv'], accept_multiple_files=True, key=f"uploader_{i}")
            
            # Store files and kepid in session
            if uploaded_files:
                st.session_state.body_data[i] = {
                    'kepid': kepid_input or '-1',
                    'files': uploaded_files,
                    'merged_df': None
                }
    
    # Global Predict All button
    if st.button(t("Predict All Bodies"), key="predict_all", use_container_width=True):
        st.session_state.predictions = []
        success_count = 0
        db = get_db()
        for body_id, data in st.session_state.body_data.items():
            if data.get('files'):
                body_dfs = []
                filenames = []
                for f in data['files']:
                    df = read_flux_file(f)
                    if df is not None:
                        body_dfs.append(df)
                        filenames.append(f.name)
                
                if body_dfs:
                    # Merge all dfs for this body
                    merged_df = pd.concat(body_dfs, ignore_index=True)
                    # Sort by time
                    merged_df = merged_df.sort_values('time').reset_index(drop=True)
                    # Set single kepid
                    merged_df['kepid'] = data['kepid']
                    
                    # Predict
                    pred_class, percent = predict_body(merged_df, data['kepid'])
                    status = t("Exists!") if pred_class == 1 else t("Does not exist")
                    
                    st.session_state.predictions.append({
                        'body_id': body_id + 1,
                        'kepid': data['kepid'],
                        'class': pred_class,
                        'status': status
                    })
                    
                    with st.expander(t(f"Body {body_id + 1} ({data['kepid']}): {status}")):
                        st.write(t(f"Probability: {percent:.2f}%"))
                        st.write(t(f"Merged {len(filenames)} files: {', '.join(filenames)}"))
                        
                        # Visualize flux data with matplotlib
                        if 'time' in merged_df.columns and 'flux' in merged_df.columns:
                            st.subheader(t("Flux Light Curve"))
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(merged_df['time'], merged_df['flux'], linewidth=0.5, alpha=0.7)
                            ax.set_xlabel(t('Time'))
                            ax.set_ylabel(t('Flux'))
                            ax.set_title(t(f'Light Curve for KEPLER ID: {data["kepid"]}'))
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            plt.close(fig)  # Close to free memory
                        else:
                            st.warning(t("Flux data columns 'time' and 'flux' not found for plotting."))
                    
                    success_count += 1
                else:
                    st.warning(t(f"No valid files for Body {body_id + 1}"))
            else:
                st.warning(t(f"No files for Body {body_id + 1}"))
        
        if success_count > 0:
            st.success(t(f"Predicted {success_count} bodies!"))

            user: User = db.get_user(st.user.email)
            if user:
                real_count = len([p for p in st.session_state.predictions if p['class'] == 1])
                fake_count = len([p for p in st.session_state.predictions if p['class'] == 0])
                result_markdown = t(f"Predicted {success_count} bodies: {real_count} exist, {fake_count} do not exist")
                os.makedirs("userdata", exist_ok=True)
                filename = str(uuid.uuid4()) + ".csv"
                full_path = os.path.join("userdata", filename)
                results_df = pd.DataFrame(st.session_state["predictions"])
                results_df = results_df[['body_id', 'kepid', 'class', 'status']]
                try:
                    results_df.to_csv(full_path, index=False)
                    db.add_prediction_record(
                        user=user,
                        type="flux_upload",
                        name=f"{t('Flux Predictions')} ({success_count} bodies)",
                        result_markdown=result_markdown,
                        user_data_path="userdata",
                        output_filename=filename,
                        timestamp=datetime.now()
                    )
                except Exception as e:
                    st.error(t(f"Failed to save flux prediction results: {e}"))
        else:
            st.warning(t("No predictions made!"))
        
    if st.session_state.get("predictions"):
        # Convert predictions to DataFrame
        results_df = pd.DataFrame(st.session_state["predictions"])
        # Select columns
        results_df = results_df[['body_id', 'kepid', 'class', 'status']]
        csv_bytes = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=t("Download All Predictions CSV"),
            data=csv_bytes,
            file_name="all_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning(t("Predict all bodies first!"))

# -----------------------------
# Tab 2
# -----------------------------
with tab2:
    st.subheader(t("Upload preprocessed table ['kepid','time','flux']"))
    pre_file = st.file_uploader(t("Select preprocessed CSV/TBL/TXT"), type=['csv','tbl','txt'])
    kepid_input = st.text_input(t("Enter KEPLER ID (optional, supports strings)"), value='-1')
    if st.button(t("Predict"), key="predict_single"):
        if pre_file:
            df = read_flux_file(pre_file)
            if df is not None:
                # For single file, sort by time if not already
                df = df.sort_values('time').reset_index(drop=True)
                df['kepid'] = kepid_input
                pred_class, percent = predict_body(df, kepid_input)
                status = t("Exists!") if pred_class == 1 else t("Does not exist")
                st.success(t(f"Status: {status}"))
                st.info(t(f"Probability: {percent:.2f}%"))
                st.info(t(f"Source: {pre_file.name}"))
                
                # Simple CSV for single
                results_df = pd.DataFrame([{
                    'body_id': 1,
                    'kepid': kepid_input,
                    'class': pred_class,
                    'status': status
                }])
                csv_bytes = results_df.to_csv(index=False).encode()
                st.download_button(t("Download result CSV"), data=csv_bytes, file_name="prediction_result.csv")
            else:
                st.warning(t("No valid file loaded!"))
        else:
            st.warning(t("No file uploaded!"))
