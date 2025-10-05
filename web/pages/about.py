import streamlit as st

st.set_page_config(page_title="About Kepler FPT", page_icon="🌌")

st.title("🌌 About Kepler FPT")

st.markdown("""
We are **Kepler FPT** — a passionate team of Vietnamese developers who love **coding, exploring, and discovering new worlds** beyond our own 🌍🚀  

Born from **NASA’s Space Apps Challenge 2025**, our mission is to harness the power of **Artificial Intelligence (AI)** and **Machine Learning (ML)** to uncover **new exoplanets** hidden in data from NASA missions like **Kepler**, **K2**, and **TESS**.  

With curiosity as our compass and innovation as our rocket fuel, we’re turning raw data into cosmic insights ✨  
""")

st.divider()

st.subheader("👩‍🚀 Our Crew")

cols = st.columns(5)

with cols[0]:
    st.markdown("🌟 **Hà Trí Kiên**  \n*Team Leader* 🇻🇳")
with cols[1]:
    st.markdown("🪐 **Vũ Hoà Vượng**  \n*Member* 🇻🇳")
with cols[2]:
    st.markdown("🌕 **Dương Hoàng Kỳ Anh**  \n*Member* 🇻🇳")
with cols[3]:
    st.markdown("🌌 **Phùng Thiện Bảo**  \n*Member* 🇻🇳")
with cols[4]:
    st.markdown("💫 **Đinh Thảo Nhi**  \n*Member* 🇻🇳")

st.divider()

st.markdown("> ✨ *“Exploring the unknown, one dataset at a time.”*")
