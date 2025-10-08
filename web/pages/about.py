import streamlit as st
from web.helper.translator import t


# Page configuration
st.set_page_config(page_title="About Kepler FPT", page_icon="🌌")

# English base content
about_text = """
We are **Kepler FPT** — a passionate team of Vietnamese developers who love **coding, exploring, and discovering new worlds** beyond our own 🌍🚀  

Born from **NASA’s Space Apps Challenge 2025**, our mission is to harness the power of **Artificial Intelligence (AI)** and **Machine Learning (ML)** to uncover **new exoplanets** hidden in data from NASA missions like **Kepler**, **K2**, and **TESS**.  

With curiosity as our compass and innovation as our rocket fuel, we’re turning raw data into cosmic insights ✨  
"""

quote_text = "> ✨ *“Exploring the unknown, one dataset at a time.”*"
contact_text = "📧 **Contact us:** [admin@fptoj.com](mailto:admin@fptoj.com)"

about_text = t(about_text)
quote_text = t("“Exploring the unknown, one dataset at a time.”")
quote_text = f"> ✨ *{quote_text}*"
contact_text = t("📧 **Contact us:** [admin@fptoj.com](mailto:admin@fptoj.com)")

# UI rendering
st.title(t("🌌 About Kepler FPT"))

st.markdown(about_text)
st.divider()

st.subheader(t("👩‍🚀 Our Crew"))

members = [
    "🌟 **Hà Trí Kiên**  \n*Team Leader* 🇻🇳",
    "🪐 **Vũ Hoà Vượng**  \n*Member* 🇻🇳",
    "🌕 **Dương Hoàng Kỳ Anh**  \n*Member* 🇻🇳",
    "🌌 **Phùng Thiện Bảo**  \n*Member* 🇻🇳",
    "💫 **Đinh Thảo Nhi**  \n*Member* 🇻🇳",
]

cols = st.columns(len(members))

for i, en_text in enumerate(members):
    with cols[i]:
        st.markdown(t(en_text))

st.divider()
st.markdown(quote_text)
st.divider()
st.markdown(contact_text)