import os
import pandas as pd
import streamlit as st
from web.db import connect_db
from web.db.models import PredictRecord
from web.helper.translator import t


def display_prediction(record: PredictRecord):
    """Display a single prediction record in an expander, translated"""
    title = f"{record.type} - {record.name} ({record.timestamp.strftime('%Y-%m-%d %H:%M:%S')})"
    title_translated = t(title)

    with st.expander(title_translated):
        content = record.result_markdown
        content_translated = t(content)
        st.markdown(content_translated)

        if record.has_output_file and record.user_data_path and record.output_filename:
            full_output_path = os.path.join(record.user_data_path, record.output_filename)

            try:
                if os.path.exists(full_output_path):
                    with open(full_output_path, "rb") as f:
                        st.download_button(
                            t("📥 Download Results"),
                            f,
                            file_name=record.output_filename,
                            mime="application/octet-stream",
                        )
                else:
                    st.warning(t("⚠️ Output file not found!"))
            except (IOError, OSError):
                st.error(t("Error accessing output file!"))


def main():
    # --- Translated UI titles ---
    st.title(t("📊 Prediction History"))

    # --- DB & user ---
    db = connect_db()
    user = db.get_user(st.user.email)
    predictions = user.predictions

    if not predictions:
        st.info(t("No predictions found. Try making some predictions first!"))
        return

    # --- Filters & Sorting ---
    date_orders = [
        t("Newest first"),
        t("Oldest first"),
    ]
    with st.expander(t("⚙️ Filters & Sorting"), expanded=True):
        col1, col2 = st.columns([2, 3])
        with col1:
            selected_type = st.selectbox(
                t("Filter by type"),
                options=["All"] + sorted(set(p.type for p in predictions)),
            )
        with col2:
            date_order = st.radio(
                t("Sort order"),
                date_orders,
                horizontal=True,
            )

    # --- Apply filters ---
    sorted_predictions = sorted(predictions, key=lambda x: x.timestamp, reverse=True)
    filtered_predictions = sorted_predictions

    if selected_type != "All":
        filtered_predictions = [p for p in filtered_predictions if p.type == selected_type]

    if date_order == date_orders[1]:
        filtered_predictions = list(reversed(filtered_predictions))

    # --- Display predictions ---
    st.markdown(f"### {t('Predictions')}")
    for record in filtered_predictions:
        display_prediction(record)

    # --- Sidebar summary ---
    st.sidebar.markdown(f"### {t('History Summary')}")
    st.sidebar.metric(
        t("Total Predictions"),
        len(predictions),
    )

    # --- Summarization by type ---
    prediction_types = {}
    for p in predictions:
        prediction_types[p.type] = prediction_types.get(p.type, 0) + 1

    st.sidebar.markdown(f"#### {t('By Type')}")

    summary_df = pd.DataFrame(
        [{"Type": ptype, "Count": count} for ptype, count in prediction_types.items()]
    ).sort_values(by="Count", ascending=False).reset_index(drop=True)

    st.sidebar.dataframe(summary_df, use_container_width=True, hide_index=True)


main()
