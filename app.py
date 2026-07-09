"""Lightweight Streamlit demo UI -- for mentor/portfolio presentations.

Run:
   app.py

Requires the FastAPI backend running separately:
    uvicorn src.retinocare.api.main:app --reload
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000/predict"
HEALTH_URL = "http://localhost:8000/health"

st.set_page_config(page_title="RetinoCare AI", page_icon="👁️")
st.title("RetinoCare AI — Diabetic Retinopathy Screening Assistant")
st.caption("Screening support tool — not a medical diagnosis.")

# --- Backend connectivity check ---
try:
    health = requests.get(HEALTH_URL, timeout=3)
    backend_ok = health.status_code == 200
except requests.exceptions.RequestException:
    backend_ok = False

if not backend_ok:
    st.warning(
        "Backend API not reachable at `localhost:8000`. Start it with:\n\n"
        "`uvicorn src.retinocare.api.main:app --reload`",
        icon="⚠️",
    )

uploaded_file = st.file_uploader("Upload a fundus image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, width=300)

    if st.button("Analyze image", disabled=not backend_ok):
        with st.spinner("Running classification and generating recommendation..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files, timeout=60)

                if response.status_code == 200:
                    result = response.json()

                    st.subheader(f"Severity: {result['severity_label']}")
                    st.progress(result["confidence"])
                    st.caption(f"Model confidence: {result['confidence']:.1%}")

                    st.markdown("### Agent Recommendation")
                    st.write(result["agent_recommendation"])

                    if result["sources"]:
                        st.caption("Sources: " + ", ".join(result["sources"]))

                    st.info(result["disclaimer"], icon="ℹ️")
                else:
                    st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach the backend: {exc}")