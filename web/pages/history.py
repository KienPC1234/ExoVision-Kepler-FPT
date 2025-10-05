import os
import streamlit as st
from web.db import connect_db
from web.db.wrapper import DBWrapper
from web.db.models import PredictRecord

def display_prediction(record: PredictRecord):
    """Display a single prediction record in an expander"""
    with st.expander(f"{record.type} - {record.name} ({record.timestamp.strftime('%Y-%m-%d %H:%M:%S')})"):
        st.markdown(record.result_markdown)
        
        if record.has_output_file and record.user_data_path and record.output_filename:
            full_output_path = os.path.join(record.user_data_path, record.output_filename)
            
            try:
                if os.path.exists(full_output_path):
                    with open(full_output_path, 'rb') as f:
                        st.download_button(
                            "📥 Download Results",
                            f,
                            file_name=record.output_filename,
                            mime="application/octet-stream"
                        )
                else:
                    st.warning("Output file not found!")
            except (IOError, OSError) as e:
                st.error(f"Error accessing output file!")

def main():
    st.title("📊 Prediction History")

    # DB & user
    db = connect_db()
    user = db.get_user(st.session_state["auth_user"])
    predictions = user.predictions

    if not predictions:
        st.info("No predictions found. Try making some predictions first!")
        return

    # Sort predictions by timestamp (newest first)
    sorted_predictions = sorted(predictions, key=lambda x: x.timestamp, reverse=True)

    # --- Filters & Sorting ---
    with st.expander("Filters & Sorting", expanded=True):
        col1, col2 = st.columns([2, 3])
        with col1:
            selected_type = st.selectbox(
                "Filter by type",
                options=["All"] + sorted(set(p.type for p in predictions))
            )
        with col2:
            date_order = st.radio(
                "Sort order",
                ["Newest first", "Oldest first"],
                horizontal=True
            )

    # Apply filters
    filtered_predictions = sorted_predictions
    if selected_type != "All":
        filtered_predictions = (p for p in filtered_predictions if p.type == selected_type)
    if date_order == "Oldest first":
        filtered_predictions = reversed(list(filtered_predictions))

    # --- Display predictions ---
    st.markdown("### Predictions")
    for record in filtered_predictions:
        display_prediction(record)

    # --- Sidebar summary ---
    st.sidebar.markdown("### History Summary")
    st.sidebar.metric("Total Predictions", len(predictions))

    prediction_types = {}
    for p in predictions:
        prediction_types[p.type] = prediction_types.get(p.type, 0) + 1

    st.sidebar.markdown("#### By Type")
    for ptype, count in prediction_types.items():
        st.sidebar.text(f"{ptype}: {count}")

main()