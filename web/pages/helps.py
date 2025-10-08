import streamlit as st
from web.helper.translator import t

# --- Page title ---
st.title(t("❓ Help & User Guide"))

# --- Intro section ---
st.markdown(
    t(
        """
# Welcome to the Exoplanet Discovery Platform Help Center

This page provides guidance on using all features of the web application, including navigation, data input, prediction tools, and troubleshooting.
""")
)

# --- Getting Started ---
st.header(t("🔑 Getting Started"))
st.markdown(
    t(
        """
1. **Sign Up / Log In:**
    - Create an account or log in with your credentials.
    - Use the sidebar to navigate between pages after logging in.
2. **Sidebar Navigation:**
    - Home: Overview and platform introduction  
    - Exoplanet Predictor: Predict planet characteristics from parameters or CSV  
    - Exoplanet Flux Prediction: Predict planet existence from light curve data  
    - History: View and download your past predictions  
    - Models Docs: Read about models, features, and file requirements  
    - About: Learn about the project and team  
    - Help: Access this help page anytime
""")
)

# --- Exoplanet Predictor ---
st.header(t("🌌 Exoplanet Predictor"))
st.markdown(
    t(
        """
- **Simple Tab:** Enter planet parameters manually for a quick prediction.  
- **Advanced Tab:** Upload a CSV file (KOI, K2, or TESS) for batch predictions.  
- **Required columns:** See the Docs page for details.  
- **Results:** View predicted class and probability. Download processed CSVs and see your prediction history.
""")
)

# --- Flux Prediction ---
st.header(t("💡 Exoplanet Flux Prediction"))
st.markdown(
    t(
        """
- **Tab 1:** Upload multiple files per celestial body (supports .fits, .tbl, .csv, .tsv). Merge and predict for each body.  
- **Tab 2:** Upload a preprocessed table with columns ['kepid','time','flux'] for single prediction.  
- **Results:** View existence status, probability, and download results as CSV. Visualize light curves.
""")
)

# --- Prediction History ---
st.header(t("📊 Prediction History"))
st.markdown(
    t(
        """
- View all your past predictions, filter by type, and download results.  
- Sidebar summary shows total predictions and breakdown by type.
""")
)

# --- Documentation ---
st.header(t("📄 Model Documentation"))
st.markdown(
    t(
        """
- See the Docs page for detailed information on model architecture, input requirements, and technical details.
""")
)

# --- Troubleshooting ---
st.header(t("🛠️ Troubleshooting & Tips"))
st.markdown(
    t(
        """
- **File Upload Issues:** Ensure your files match the required formats and columns (see Docs).  
- **Prediction Errors:** Check for missing or invalid values in your input data.  
- **Session Problems:** If the app behaves unexpectedly, try refreshing the page or logging out and back in.  
- **Contact Support:** For persistent issues, use the contact information on the About page.
""")
)

# --- FAQ ---
st.header(t("🙋 Frequently Asked Questions (FAQ)"))
st.markdown(
    t(
        """
**Q: What file formats are supported?**  
- See Docs for each tool. Generally, .csv, .fits, .tbl, and .tsv are supported.  

**Q: How do I reset my password?**  
- Currently, contact support via the About page for password resets.  

**Q: Can I use my own data?**  
- Yes! You can upload your own CSVs or light curve files as long as they match the required format.  

**Q: Where are my results saved?**  
- All predictions are saved in your user history and can be downloaded as CSV files.
""")
)
